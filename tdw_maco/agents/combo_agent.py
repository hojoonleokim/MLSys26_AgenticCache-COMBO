import os
import json
import pickle
import random
import time
from openai import AzureOpenAI, OpenAI
import numpy as np
from PIL import Image
from combo.proposer import Proposer
from combo.cwm import CWM
from combo.ranker import Ranker
from combo.intent import IntentTracker
import base64
from utils.utils import get_ego_topdown, get_overlay_ego_topdown
from copy import deepcopy
from multiprocessing import Pool, Manager
import re
_worker_vlm = None

def _pool_init_vlm(vlm_args):
    """Initialize VLM instance in worker process"""
    global _worker_vlm
    from combo.proposer import Proposer
    
    # Unpack VLM arguments
    task, max_tokens, debug_mode, num_propose, temperature, proposer_lm_id, lm_source, cot, agent_id, run_id, experiment_name, output_dir = vlm_args
    
    # Initialize Proposer instance
    _worker_vlm = Proposer(
        task=task,
        max_tokens=max_tokens,
        debug_mode=debug_mode,
        num_propose=num_propose,
        temperature=temperature,
        lm_id=proposer_lm_id,
        lm_source=lm_source,
        cot=cot,
        agent_id=agent_id,
        run_id=run_id,
        experiment_name=experiment_name,
        output_dir=output_dir
    )

def _pool_reset_vlm(recipe, agents_name):
    """Reset VLM instance in worker process"""
    global _worker_vlm
    import os
    worker_pid = os.getpid()
    print(f"[Worker {worker_pid}] Resetting VLM with recipe={recipe is not None}")
    if _worker_vlm is not None:
        _worker_vlm.reset(recipe, agents_name)
        print(f"[Worker {worker_pid}] Reset completed")
    else:
        print(f"[Worker {worker_pid}] WARNING: _worker_vlm is None!")

def _pool_reset_vlm_wrapper(args):
    """Wrapper for pool.map which only accepts single argument"""
    recipe, agents_name = args
    return _pool_reset_vlm(recipe, agents_name)

def async_vlm_worker(vlm_args, done_flag, result_plan, result_info):
    """Worker function that runs VLM planning in a separate process"""
    try:
        global _worker_vlm
        
        # Unpack arguments
        img_path, agent_id, save_path, progress, available_plans, steps, action_history, episode, use_future_prompt = vlm_args
        
        # Use the global VLM instance to run proposer
        if _worker_vlm is None:
            raise RuntimeError("Worker VLM not initialized. Call _pool_init_vlm first.")
        
        proposes = _worker_vlm.run(
            img_path, agent_id, save_path, progress, 
            available_plans, steps, action_history, episode, use_future_prompt
        )

        # Store results in shared memory
        result_plan['proposes'] = proposes
        result_info['success'] = True
        
        # Set done flag
        done_flag.value = True
        
    except Exception as e:
        print(f"Error in async VLM worker: {e}")
        import traceback
        traceback.print_exc()
        result_plan['proposes'] = None
        result_info['error'] = str(e)
        result_info['success'] = False
        done_flag.value = True

# Function to encode the image
def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class COMBOAgent:
    """
    The High Level Planner of COMBO, consisting of the following components:
        - Action Proposer VLM: a Vision Language Model that takes in the current observation and propose an action
        - Intent Tracker VLM: a Belief Model that takes in the observation history and infer about other agent's actions
        - Compositional World Model: a Video Model that takes in the current observation and the joint action, simulates the world dynamics
        - Outcome Evaluator VLM: a Ranker that takes in the image outcome and rank it
    """
    def __init__(
        self,
        agent_id, task, logger, output_dir='results',
        max_tokens: int = 512,
        debug_mode: bool = False,
        num_propose: int = 2,
        temperature: float = 0,
        proposer_lm_id: str = "gpt-4",
        belief_lm_id: str = "gpt-4",
        ranker_lm_id: str = "gpt-4",
        lm_source: str = "openai",
        only_propose: bool = False,
        no_belief: bool = False,
        plan_horizon: int = 1,
        plan_beam: int = 1,
        cot: bool = False,
        guidance_weight: int = 10,
        history_horizon: int = 3,
        run_id: str = "baseline",
        experiment_name: str = "train"
    ):

        self.progress = 0
        self.agents_name = None
        self.num_agents = 2
        self.agent_id = agent_id
        self.agent_type = 'combo_agent'
        self.task = task
        self.recipe = None
        self.logger = logger
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
        self.steps = 0
        self.guidance_weight = guidance_weight
        self.history_horizon = history_horizon
        self.run_id = run_id
        self.experiment_name = experiment_name
        chat_base = os.path.join(output_dir, task, proposer_lm_id)
        self.record_dir = f"{chat_base}/{agent_id}_chat.jsonl"
        os.makedirs(chat_base, exist_ok=True)
        self.only_propose = only_propose
        self.no_belief = no_belief
        self.plan_horizon = plan_horizon
        self.plan_beam = plan_beam
        self.num_propose = num_propose

        self.temperature = temperature
        self.lm_source = lm_source
        self.output_dir = output_dir
        self.action_history = []
        self.object_in_hand = None
        # self.obs_history = []
        self.top_down_history = None
        
        # Initialize async VLM processing components (한번만 생성, reset에서 재사용)
        self.proposer_lm_id = proposer_lm_id
        self.cot = cot
        self.manager = Manager()
        
        # Target pool - 1개의 VLM 쿼리를 관리 (sequential processing)
        self.target_done_flags = [self.manager.Value('b', False)]
        self.target_async_plans = [self.manager.dict()]
        self.target_async_infos = [self.manager.dict()]
        self.target_async_results = [None]
        self.target_running = [False]
        
        # Create process pool with initializer (한번만 생성, 이후 reset에서 재사용)
        # Pool 생성 시 worker process에서 Proposer 인스턴스가 _pool_init_vlm을 통해 자동 생성됨
        self.target_vlm_args = (task, max_tokens, debug_mode, num_propose, temperature, 
                   proposer_lm_id, lm_source, cot, agent_id, run_id, experiment_name, output_dir)
        self.target_pool = None
        self.target_plan = {}  # {issue_timestep: plan_info}
        
        # Track which timestep each slot was issued at
        self.target_issue_timesteps = [None]
        
        # Initialize proposer for reparsing actions
        self.proposer = Proposer(
            task=task,
            max_tokens=max_tokens,
            debug_mode=debug_mode,
            num_propose=num_propose,
            temperature=temperature,
            lm_id=proposer_lm_id,
            lm_source=lm_source,
            cot=cot,
            agent_id=agent_id,
            run_id=run_id,
            experiment_name=experiment_name,
            output_dir=output_dir
        )
        
        if self.only_propose:
            self.cwm = CWM(
                serve="server" in lm_source,
                device="cuda:0",
                target_size=(128, 128),
                sample_per_seq=8,
                guidance_weight=guidance_weight,
                model_id=None,
                inpainting_model_id="modl-100.pt",
                superres_model=None,
            )
            return
        self.cwm = CWM(
            serve="server" in lm_source,
            device="cuda:0",
        	target_size = (128, 128),
        	sample_per_seq=8,
        	guidance_weight=guidance_weight,
        )
        self.ranker = Ranker(
            task=task,
            max_tokens=max_tokens,
            debug_mode=debug_mode,
            temperature=temperature,
            lm_id=ranker_lm_id,
            lm_source=lm_source,
            cot=cot,
        )
        if self.no_belief:
            return
        self.intenttracker = IntentTracker(
            task=task,
            max_tokens=max_tokens,
            debug_mode=debug_mode,
            temperature=temperature,
            lm_id=belief_lm_id,
            lm_source=lm_source,
            cot=cot,
        )

    def reset(self, obs, info, output_dir='results'):
        self.agents_name = info["agents_name"]
        self.num_agents = len(self.agents_name)
        self.output_dir = os.path.join(output_dir, f"{self.agent_id}")
        self.steps = 0
        self.action_history = [{"type": None, "prompt": "wait"}]
        self.object_in_hand = None
        # self.obs_history = [None for _ in range(self.history_horizon)]
        self.top_down_history = [None for _ in range(self.history_horizon)]
        
        # Save task-specific metadata
        if self.task == 'cook':
            self.recipe = info["recipe"]
            self.progress = 0
            self.reachable_bins = info["reachable_bins"][int(self.agent_id)]
            self.dest_plates = info["dest_plates"]
        else:  # game task
            self.recipe = None
            self.progress = None  # Game task doesn't have progress
            self.puzzle_id2piece_id = info["puzzle_id2piece_id"].copy()
            self.reachable_bins = info["reachable_bins"][int(self.agent_id)]
            self.reachable_region = info["reachable_region"][int(self.agent_id)]  # [x1, z1, x2, z2]
            self.annotation_dict = info["annotation_dict"].copy()
            self.semantic_name_dict = info["semantic_name_dict"].copy()
            # puzzle_ids will be determined in act() based on proximity

        # Reset async query state by closing and recreating the VLM pool
        # This ensures all pending queries from previous episode are completely terminated
        
        # Reset target pool
        if hasattr(self, 'target_pool') and self.target_pool is not None:
            try:
                self.target_pool.terminate()
                self.target_pool.join()
            except Exception as e:
                print(f"Warning: Error terminating target pool: {e}")
        
        # Recreate the pool for the new episode
        self.target_pool = Pool(
            processes=1,
            initializer=_pool_init_vlm,
            initargs=(self.target_vlm_args,)
        )
        
        # Reset VLM in ALL worker processes with current episode info
        try:
            # Use map to ensure all workers get the reset
            self.target_pool.map(_pool_reset_vlm_wrapper, [(self.recipe, self.agents_name)] * 1)
        except Exception as e:
            print(f"Warning: Could not reset worker VLM: {e}")

        # Reset the proposer instance for reparsing
        self.proposer.reset(self.recipe, self.agents_name)
        # if not self.only_propose:
        #     self.ranker.reset(self.recipe, self.agents_name)

        # if not self.only_propose and not self.no_belief:
        #     self.intenttracker.reset(self.recipe, self.agents_name)

        # Reset all async-related state
        self.target_async_results[0] = None
        self.target_done_flags[0].value = False
        self.target_running[0] = False
        self.target_async_plans[0].clear()
        self.target_async_infos[0].clear()
        self.target_issue_timesteps[0] = None
        
        # Clear plan dictionary
        self.target_plan.clear()

    def write_log_to_file(self,log_message, file_name=None):
        file_name = self.record_dir
        with open(file_name, 'a') as file:  
            file.write(log_message + '\n')
    
    def cleanup(self):
        """Clean up multiprocessing resources"""
        from multiprocessing import TimeoutError
        
        # Collect all running results with timeout before closing
        if hasattr(self, 'target_pool') and self.target_pool is not None:
            if self.target_running[0] and self.target_async_results[0] is not None:
                try:
                    # Wait up to 30 seconds for this result
                    self.target_async_results[0].get(timeout=30)
                    # Extract and store the result
                    proposes = self.target_async_plans[0].get('proposes', None)
                    issue_timestep = self.target_issue_timesteps[0]
                    if issue_timestep is not None and proposes is not None:
                        self.target_plan[issue_timestep] = proposes
                        print(f"Collected target pool result during cleanup (step {issue_timestep})")
                except TimeoutError:
                    print(f"Warning: Timeout waiting for target pool. Force closing.")
                except Exception as e:
                    print(f"Warning: Error collecting target pool result: {e}")
            
            self.target_pool.close()
            self.target_pool.join()
            self.target_pool = None  

    def submit_async_vlm_query(self, img_path, agent_id, save_path, progress, available_plans, action_history, episode, target_step, use_future_prompt=False):
        """Submit an async VLM query to the target pool
        
        Args:
            target_step: The step number to generate plan for.
            use_future_prompt: If True, use get_future_proposal_prompt instead of get_proposal_prompt
        """
        pool = self.target_pool
        done_flags = self.target_done_flags
        async_plans = self.target_async_plans
        async_infos = self.target_async_infos
        async_results = self.target_async_results
        running = self.target_running
        issue_timesteps = self.target_issue_timesteps
        
        if pool is None:
            raise RuntimeError("Target pool not initialized")
        
        # Use the single slot (slot 0)
        slot_idx = 0
        if running[0]:
            print(f"Warning: Target pool is busy. Query not submitted.")
            return False
        
        # Pack arguments for worker
        vlm_args = (img_path, agent_id, save_path, progress, available_plans, target_step, action_history, episode, use_future_prompt)
        
        # Reset shared state for this slot
        done_flags[slot_idx].value = False
        async_plans[slot_idx].clear()
        async_infos[slot_idx].clear()
        
        # Record the timestep this query was issued at
        issue_timesteps[slot_idx] = target_step
        
        # Submit async task
        async_results[slot_idx] = pool.apply_async(
            async_vlm_worker,
            (vlm_args, done_flags[slot_idx], async_plans[slot_idx], async_infos[slot_idx])
        )
        running[slot_idx] = True
        return True
        
    def check_async_vlm_result(self):
        """Check async VLM query and write completed result to self.target_plan"""
        # Check if the single slot is completed
        if self.target_running[0] and self.target_done_flags[0].value:
            issue_step = self.target_issue_timesteps[0]
            
            # Get the result
            try:
                self.target_async_results[0].get(timeout=0.1)
                proposes = self.target_async_plans[0].get('proposes', None)
                
                # Write to self.target_plan
                # proposes is a list of dicts: [{'action': [...], 'raw_response': '...'}]
                # Store both action and raw response
                self.target_plan[issue_step] = {
                    'action': proposes[0]['action'],
                    'raw_response': proposes[0]['raw_response']
                }
                
                # Clear this slot
                self.target_async_results[0] = None
                self.target_running[0] = False
                self.target_done_flags[0].value = False
                self.target_issue_timesteps[0] = None
                self.target_async_plans[0].clear()
                self.target_async_infos[0].clear()
            except Exception as e:
                print(f"Error retrieving target pool result: {e}")
    
    def wait_for_llm_result(self):
        """Wait for LLM result by polling until slot is available"""
        if self.target_async_results[0] is not None:
            while True:
                self.check_async_vlm_result()
                if self.target_async_results[0] is None:
                    break
                time.sleep(1)  # Short sleep to avoid busy waiting
        else:
            self.check_async_vlm_result()

    def act(self, obs, episode, step_num):
        self.steps = step_num
        current_step = self.steps
        output_dir = os.path.join(self.output_dir, f"step_{current_step}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save obs and episode for send_futures() method
        self._last_obs = obs
        self._last_episode = episode
        
        # Log previous step result (only if there was a previous action)
        if len(self.action_history) > 0:
            if obs['rejected']:
                print(f"Agent {self.agent_id}: ❌ FAILED ({obs.get('rejected_reason', 'Unknown reason')})")
                self.action_history[-1] = {"type": None, "prompt": "wait"}
            else:
                print(f"Agent {self.agent_id}: ✅ SUCCESS")
        
        # Update object_in_hand based on previous action
        if len(self.action_history) > 0 and self.action_history[-1]["prompt"].startswith("pick up"):
            if self.task == 'cook':
                obj_name = self.action_history[-1]["prompt"].split("pick up the ")[1]
                # Remove location information if present
                if " from the " in obj_name:
                    obj_name = obj_name.split(" from the ")[0]
                self.object_in_hand = obj_name
            else:  # game task - need to convert semantic name to object ID
                picked_name = self.action_history[-1]["prompt"].split("pick up ")[1]
                # Remove location information if present
                if " from the " in picked_name:
                    picked_name = picked_name.split(" from the ")[0]
                # Find object ID from semantic name
                for obj_id, semantic_name in self.semantic_name_dict.items():
                    if semantic_name.replace('_', ' ') == picked_name:
                        self.object_in_hand = obj_id
                        break
        
        if len(self.action_history) > 0 and self.action_history[-1]["prompt"].startswith("place"):
            if self.task == 'cook' and "plate" in self.action_history[-1]["prompt"]:
                assert self.object_in_hand == self.recipe[self.agent_id][self.progress], f"seems not right! object_in_hand: {self.object_in_hand}, recipe: {self.recipe}, progress: {self.progress}"
                self.progress += 1 # self.recipe[self.agent_id][self.progress] is the next one to be placed on the plate
            self.object_in_hand = None

        # step1: estimate world state

        reconstructed_imgs = []
        for i, ego_history_path in enumerate(obs['ego_histories']):
            cur_top_down = get_ego_topdown(self.task, obs['camera_matrix'][i], ego_history_path, ego_history_path.replace("img", "depth"))
            # Image.fromarray(cur_top_down).save(
            #     os.path.join(output_dir, ego_history_path.split("/")[-1].replace("img", "reconstructed")))
            reconstructed_imgs.append(cur_top_down)
        overlay = get_overlay_ego_topdown(reconstructed_imgs)
        overlay_path = os.path.join(output_dir, "overlay_top_down.png")
        Image.fromarray(overlay).save(overlay_path)
        inpainting_top_down = self.cwm.run([[overlay_path, None, "", output_dir, "inpainting_top_down"]])[0]
        # Remove intermediate overlay image (only inpainted result is needed)
        if os.path.exists(overlay_path):
            os.remove(overlay_path)

        # step2: get available actions based on task
        if self.task == 'cook':
            available_plans = get_available_action_prompts_cook(
                obs=obs,
                agent_id=self.agent_id,
                recipe=self.recipe,
                reachable_bins=self.reachable_bins,
                dest_plates=self.dest_plates,
                object_in_hand=self.object_in_hand
            )
        elif self.task == 'game':  # game task
            available_plans = get_available_action_prompts_game(
                obs=obs,
                agent_id=self.agent_id,
                puzzle_id2piece_id=self.puzzle_id2piece_id,
                reachable_bins=self.reachable_bins,
                annotation_dict=self.annotation_dict,
                semantic_name_dict=self.semantic_name_dict,
                object_in_hand=self.object_in_hand
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Calculate progress for cook task
        progress_info = None

        # Submit VLM query only if available_plans has meaningful actions
        # Skip if available_plans is empty or only contains "wait"
        has_meaningful_actions = not (len(available_plans) == 1 and "wait" in available_plans[0].lower())
        

        self.wait_for_llm_result()
        if current_step not in self.target_plan:
            self.submit_async_vlm_query([inpainting_top_down], self.agent_id, output_dir, self.progress, available_plans, self.action_history, episode, target_step=current_step)
    
        elif self.target_plan[current_step]['action'][0] not in available_plans:
            reparsed = self.proposer.parse_action_from_response(self.target_plan[current_step]['raw_response'], available_plans)
            is_alphabet_format = bool(re.search(r'Next Action\s*:\s*[A-Z](?:\s|$)', self.target_plan[current_step]['raw_response'], re.IGNORECASE))
            if reparsed not in available_plans or is_alphabet_format:
                # Reparsing failed - remove target plan and submit new query
                print(f"⚠️  Agent {self.agent_id}: Could not reparse action, resubmitting query")
                del self.target_plan[current_step]  # Remove so we wait for new result
                # Submit new query for next attempt if meaningful actions available
                if has_meaningful_actions:
                    self.submit_async_vlm_query([inpainting_top_down], self.agent_id, output_dir, self.progress, available_plans, self.action_history, episode, target_step=current_step)
                else:
                    # No meaningful actions, set to wait
                    self.target_plan[current_step] = {'action': ['wait'], 'raw_response': ''}
            else:
                self.target_plan[current_step]['action'][0] = reparsed
                print(f"Agent {self.agent_id}: Successfully reparsed to '{reparsed}'")
        # self.submit_async_vlm_query([inpainting_top_down], self.agent_id, output_dir, self.progress, available_plans, self.action_history, episode, target_step=current_step+1, use_future_prompt=True)

        return None
    
    def send_futures(self):
        """
        Send future prediction query for next step.
        Should be called after current action is determined.
        """
        if not hasattr(self, '_last_obs') or not hasattr(self, '_last_episode'):
            print(f"Agent {self.agent_id}: No observation data for future prediction")
            return
        
        obs = self._last_obs
        episode = self._last_episode
        current_step = self.steps
        output_dir = os.path.join(self.output_dir, f"step_{current_step}")
        
        # Get inpainting_top_down from saved location
        inpainting_path = os.path.join(output_dir, "inpainting_top_down.png")
        if not os.path.exists(inpainting_path):
            print(f"Agent {self.agent_id}: Inpainting image not found for future prediction")
            return
        
        # Get available_plans for current step (will be similar for next step)
        if self.task == 'cook':
            available_plans = get_available_action_prompts_cook(
                obs=obs,
                agent_id=self.agent_id,
                recipe=self.recipe,
                reachable_bins=self.reachable_bins,
                dest_plates=self.dest_plates,
                object_in_hand=self.object_in_hand
            )
        elif self.task == 'game':
            available_plans = get_available_action_prompts_game(
                obs=obs,
                agent_id=self.agent_id,
                puzzle_id2piece_id=self.puzzle_id2piece_id,
                reachable_bins=self.reachable_bins,
                annotation_dict=self.annotation_dict,
                semantic_name_dict=self.semantic_name_dict,
                object_in_hand=self.object_in_hand
            )
        else:
            print(f"Agent {self.agent_id}: Unknown task for future prediction")
            return
        

        self.submit_async_vlm_query(
            [inpainting_path], 
            self.agent_id, 
            output_dir, 
            self.progress, 
            available_plans, 
            self.action_history, 
            episode, 
            target_step=current_step+1, 
            use_future_prompt=True
        )
        print(f"Agent {self.agent_id}: Sent future prediction query for step {current_step+1}")
    
    def convert_actions_prompt(self, actions, agent_id):
        prompt = []
        for i, agent_name in enumerate(self.agents_name):
            if str(i) in actions:
                if i != agent_id and self.conflict(actions[str(agent_id)]["prompt"], actions[str(i)]["prompt"]):
                    prompt.append(f"{agent_name} wait.")
                else:
                    prompt.append(f"{agent_name} {actions[str(i)]['prompt']}.")
            else:
                prompt.append(f"{agent_name} wait.")
        return prompt

    def conflict(self, action1, action2):
        if action1 == "wait" or action2 == "wait":
            return False
        if "place" in action1:
            object, loc = action1.split(" onto the ")
            object = object.split("the ")[1]
        else:
            object = action1.split("the ")[1]
            loc = None
        if "place" in action2:
            object2, loc2 = action2.split(" onto the ")
            object2 = object2.split("the ")[1]
        else:
            object2 = action2.split("the ")[1]
            loc2 = None
        if object == object2:
            print("conflict, same object")
            return True
        if loc == "cutting board" and loc2 == "cutting board":
            print("conflict, same location cutting board")
            return True
        return False


# ============================================
# Constants for Cook task
# ============================================
BIN_DIST_THRESHOLD = 0.1
BOARD_DIST_THRESHOLD = 0.2
PLATE_DIST_THRESHOLD = 0.25
WHOLE_TO_SLICE = {"whole_cheese": "cheese_slice", "whole_tomato": "tomato_slice", "whole_onion": "onion_slice"}

# Cook task place types
PLACE_ON_THE_CUTTING_BOARD = 10
PLACE_ON_THE_PLATE = 11
PLACE_IN_THE_PRIVATE_REGION_TOP_LEFT = 12
PLACE_IN_THE_PRIVATE_REGION_TOP_RIGHT = 13
PLACE_IN_THE_PRIVATE_REGION_BOTTOM_LEFT = 14
PLACE_IN_THE_PRIVATE_REGION_BOTTOM_RIGHT = 15

# ============================================
# Constants for Game task
# ============================================
BIN_THRESHOLD_GAME = 0.2  # Pieces near puzzle box center, bins at edges (region check prevents cross-agent access)
IN_PUZZLE_THRESHOLD = 0.1

# Game task place types (matching GamePlanAgent)
PLACE_INTO_PUZZLE = 1
PLACE_ON_THE_LEFT_BORDER = 2
PLACE_ON_THE_RIGHT_BORDER = 3
PLACE_INSIDE_PRIVATE_AREA_LEFT = 4
PLACE_INSIDE_PRIVATE_AREA_RIGHT = 5

def l2_dist(pos1, pos2):
    if isinstance(pos1, dict):
        pos1 = np.array([pos1["x"], pos1["y"], pos1["z"]])
    if isinstance(pos2, dict):
        pos2 = np.array([pos2["x"], pos2["y"], pos2["z"]])
    if isinstance(pos1, list):
        pos1 = np.array(pos1)
    if isinstance(pos2, list):
        pos2 = np.array(pos2)
    return np.linalg.norm(pos1 - pos2)


def get_available_action_prompts_cook(obs, agent_id, recipe, reachable_bins, dest_plates, object_in_hand):
    """
    Generate and return a list of available action prompts for Cook task.
    
    Args:
        obs: Observation dictionary
        agent_id: Agent ID
        recipe: Recipe list for both agents
        reachable_bins: List of reachable bin positions
        dest_plates: List of destination plate positions
        object_in_hand: Currently held object name (string) or None
        
    Returns: 
        List of prompt strings
    """
    # Get item on cutting board
    board_name = "wood_board"
    board_pos = None
    for obj in obs["objects"]:
        if obj["name"] == board_name:
            board_pos = obj["pos"]
            break
    
    object_on_cutting_board_name = None
    if board_pos is not None:
        for obj in obs["objects"]:
            if obj["name"] is not None and obj['name'] != "wood_board" and obj['name'] != "vk0007_steak_knife" and l2_dist(obj['pos'], board_pos) < BOARD_DIST_THRESHOLD:
                object_on_cutting_board_name = obj["name"]
                break
    
    # Check goal progress
    num = [0, 0]
    for agent_idx in range(2):
        plate_pos = dest_plates[agent_idx]
        last_height = -5
        for i in range(len(recipe[agent_idx])):
            name = recipe[agent_idx][i]
            objs = []
            for obj in obs["objects"]:
                if obj["name"] == name and l2_dist(obj["pos"], plate_pos) < PLATE_DIST_THRESHOLD and obj["pos"][1] > last_height:
                    objs.append(obj)
            if len(objs) == 0:
                num[agent_idx] = i
                break
            if len(objs) > 1:
                objs = sorted(objs, key=lambda x: x["pos"][1])
            last_height = objs[0]["pos"][1]
        if len(objs) > 0:
            num[agent_idx] = len(recipe[agent_idx])
    
    # Get objects in bins (just names)
    obj_in_bins = [None for _ in range(len(reachable_bins))]
    for obj in obs["objects"]:
        if obj["name"] is None:
            continue
        for i in range(len(reachable_bins)):
            if l2_dist(obj["pos"], reachable_bins[i]) < BIN_DIST_THRESHOLD:
                obj_in_bins[i] = obj["name"]
                break
    
    # Generate possible actions
    possible_actions = []
    
    if object_in_hand is None:
        # Pick actions from bins and cutting board
        pick_actions = []
        for i in range(len(reachable_bins)):
            if obj_in_bins[i] is not None:
                prompt = f"pick up the {obj_in_bins[i]} from the private region"
                pick_actions.append(prompt)
        
        if object_on_cutting_board_name is not None:
            prompt = f"pick up the {object_on_cutting_board_name} from the cutting board"
            pick_actions.append(prompt)
        
        # Shuffle pick actions
        random.shuffle(pick_actions)
        possible_actions.extend(pick_actions)
        
        # Cut action (not shuffled)
        if object_on_cutting_board_name is not None and object_on_cutting_board_name in WHOLE_TO_SLICE.keys():
            prompt = f"cut the {object_on_cutting_board_name}"
            possible_actions.append(prompt)
    else:
        # Place action to plate
        if num[int(agent_id)] < len(recipe[int(agent_id)]) and object_in_hand == recipe[int(agent_id)][num[int(agent_id)]]:
            prompt = f"place the {object_in_hand} onto the plate"
            possible_actions.append(prompt)
            
        # Place action to cutting board
        place_actions = []
        if object_on_cutting_board_name is None:
            prompt = f"place the {object_in_hand} onto the cutting board"
            place_actions.append(prompt)

        # Place actions to bins
        for i in range(len(reachable_bins)):
            if obj_in_bins[i] is None:
                place_names = ["top left corner", "top right corner", "bottom left corner", "bottom right corner"]
                prompt = f"place the {object_in_hand} onto the {place_names[i]} of the private region"
                place_actions.append(prompt)
        # Shuffle place actions
        random.shuffle(place_actions)
        possible_actions.extend(place_actions)
    
    possible_actions.append("wait")
    return possible_actions


def get_available_action_prompts_game(obs, agent_id, puzzle_id2piece_id, reachable_bins, semantic_name_dict, annotation_dict, object_in_hand):
    """
    Generate and return a list of available action prompts for Game task.
    
    Args:
        obs: Observation dictionary
        agent_id: Agent ID
        puzzle_id2piece_id: Dictionary mapping puzzle IDs to their piece IDs
        reachable_bins: List of reachable bin positions (4 bins: [0]=border1, [1]=private_left, [2]=private_right, [3]=border2)
        semantic_name_dict: Dictionary mapping object IDs to semantic names
        annotation_dict: Dictionary mapping object IDs to possible positions on puzzle
        object_in_hand: Currently held object ID (int) or None
        
    Returns: 
        List of prompt strings
    """
    # Build id2pos mapping
    id2pos = {obj['id']: obj['pos'] for obj in obs["objects"]}
    
    # Find agent's own puzzle box (closest to reachable_bins[2])
    agent_puzzle_id = None
    for puz_id in puzzle_id2piece_id.keys():
        if puz_id in id2pos and l2_dist(id2pos[puz_id], reachable_bins[2]) < 0.6:
            agent_puzzle_id = puz_id
            break
    
    # Get objects in bins (object IDs)
    can_reach_ids = [None for _ in range(len(reachable_bins))]
    for i, pos in enumerate(reachable_bins):
        for puzzle_key in puzzle_id2piece_id.keys():
            for piece_id in puzzle_id2piece_id[puzzle_key]:
                if piece_id in id2pos and l2_dist(id2pos[piece_id], pos) < BIN_THRESHOLD_GAME:
                    can_reach_ids[i] = piece_id
                    break

    # Generate possible actions
    possible_actions = []
    
    if object_in_hand is None:
        # Pick actions from reachable bins
        pick_actions = []
        pick_locations = [
            "right border of the reachable region",
            "private region right to the puzzle box",
            "private region left to the puzzle box",
            "left border of the reachable region"
        ]
        for i, piece_id in enumerate(can_reach_ids):
            if piece_id is not None:
                semantic_name = semantic_name_dict[piece_id].replace('_', ' ')
                prompt = f"pick up {semantic_name} from the {pick_locations[i]}"
                pick_actions.append(prompt)
        # Shuffle pick actions
        random.shuffle(pick_actions)
        possible_actions.extend(pick_actions)
    else:
        # Place actions to bins - check each bin individually
        semantic_name = semantic_name_dict[object_in_hand].replace('_', ' ')
        
        # All possible place location names
        place_locations = [
            "right border of the reachable region",
            "private region right to the puzzle box",
            "private region left to the puzzle box",
            "left border of the reachable region"
        ]
        
        # Add place action for each empty bin
        for i in range(len(reachable_bins)):
            if can_reach_ids[i] is None:  # This bin is empty
                prompt = f"place {semantic_name} onto the {place_locations[i]}"
                possible_actions.append(prompt)
        
        # Place actions to puzzle box (only for agent's own puzzle)
        if agent_puzzle_id is not None and object_in_hand in puzzle_id2piece_id[agent_puzzle_id]:
            # Generate a prompt for each possible semantic position
            puzzle_place_actions = []
            for semantic_pos in annotation_dict[object_in_hand]:
                semantic_position = semantic_pos.replace('_', ' ')
                prompt = f"place {semantic_name} onto the {semantic_position} of the puzzle box"
                puzzle_place_actions.append(prompt)
            # Shuffle puzzle box place actions
            random.shuffle(puzzle_place_actions)
            possible_actions.extend(puzzle_place_actions)
    
    possible_actions.append("wait")
    return possible_actions


