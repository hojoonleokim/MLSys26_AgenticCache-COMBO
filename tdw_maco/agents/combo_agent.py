import os
import json
import pickle
import random
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
from agents.action_cache import ActionCache

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
    if _worker_vlm is not None:
        _worker_vlm.reset(recipe, agents_name)

def async_vlm_worker(vlm_args, done_flag, result_plan, result_info):
    """Worker function that runs VLM planning in a separate process"""
    try:
        global _worker_vlm
        
        # Unpack arguments
        img_path, agent_id, save_path, progress, available_plans, steps, action_history, episode = vlm_args
        
        # Use the global VLM instance to run proposer
        if _worker_vlm is None:
            raise RuntimeError("Worker VLM not initialized. Call _pool_init_vlm first.")
        
        proposes = _worker_vlm.run(
            img_path, agent_id, save_path, progress, 
            available_plans, steps, action_history, episode
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
        self.output_dir = None
        self.action_history = []
        self.object_in_hand = None
        # self.obs_history = []
        self.top_down_history = None
        
        # Initialize async VLM processing components (한번만 생성, reset에서 재사용)
        self.proposer_lm_id = proposer_lm_id
        self.cot = cot
        self.manager = Manager()
        self.vlm_done_flag = self.manager.Value('b', False)  # Boolean flag managed by Manager
        self.async_plan = self.manager.dict()
        self.async_info = self.manager.dict()
        self.async_result = None
        self.vlm_running = False  # Track if VLM is currently running
        
        # Create process pool with initializer (한번만 생성, 이후 reset에서 재사용)
        # Pool 생성 시 worker process에서 Proposer 인스턴스가 _pool_init_vlm을 통해 자동 생성됨
        self.vlm_args = (task, max_tokens, debug_mode, num_propose, temperature, 
                   proposer_lm_id, lm_source, cot, agent_id, run_id, experiment_name, output_dir)
        self.vlm_pool = None
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
        self.real_action_history = []
        # Initialize action cache for this episode with recipe/puzzle info
        if self.task == 'cook':
            self.action_cache = ActionCache(task=self.task, agent_id=self.agent_id, recipe=info["recipe"])
        else:  # game
            self.action_cache = ActionCache(task=self.task, agent_id=self.agent_id)
        
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
        if hasattr(self, 'vlm_pool') and self.vlm_pool is not None:
            try:
                # Terminate the pool immediately without waiting for pending tasks
                self.vlm_pool.terminate()
                self.vlm_pool.join()
            except Exception as e:
                print(f"Warning: Error terminating VLM pool: {e}")
        
        # Recreate the pool for the new episode
        # Use class variable for VLM args
        self.vlm_pool = Pool(
            processes=1,
            initializer=_pool_init_vlm,
            initargs=(self.vlm_args,)
        )
        
        # Reset VLM in the new worker process with current episode info
        try:
            self.vlm_pool.apply(_pool_reset_vlm, (self.recipe, self.agents_name))
        except Exception as e:
            print(f"Warning: Could not reset worker VLM: {e}")

        # self.proposer.reset(self.recipe, self.agents_name)
        # if not self.only_propose:
        #     self.ranker.reset(self.recipe, self.agents_name)

        # if not self.only_propose and not self.no_belief:
        #     self.intenttracker.reset(self.recipe, self.agents_name)

        # Reset all async-related state
        self.async_result = None
        self.vlm_done_flag.value = False
        self.vlm_running = False
        self.async_plan.clear()
        self.async_info.clear()
        self.prev_action_at_vlm_submit = None

    def write_log_to_file(self,log_message, file_name=None):
        file_name = self.record_dir
        with open(file_name, 'a') as file:  
            file.write(log_message + '\n')
    
    def cleanup(self):
        """Clean up multiprocessing resources"""
        if hasattr(self, 'vlm_pool') and self.vlm_pool is not None:
            self.vlm_pool.close()
            self.vlm_pool.join()
            self.vlm_pool = None  

    def submit_async_vlm_query(self, img_path, agent_id, save_path, progress, available_plans, steps, action_history, episode):
        """Submit an async VLM query to the process pool"""
        if not hasattr(self, 'vlm_pool') or self.vlm_pool is None:
            raise RuntimeError("VLM pool not initialized")
        
        # Pack arguments for worker
        vlm_args = (img_path, agent_id, save_path, progress, available_plans, steps, action_history, episode)
        
        # Reset shared state
        self.vlm_done_flag.value = False
        self.async_plan.clear()
        self.async_info.clear()
        
        # Submit async task
        self.async_result = self.vlm_pool.apply_async(
            async_vlm_worker,
            (vlm_args, self.vlm_done_flag, self.async_plan, self.async_info)
        )
        self.vlm_running = True
        return True
        
    def check_async_vlm_result(self):
        """Check if async VLM query is complete and return results"""

        # Check if result is ready
        if self.vlm_done_flag.value:
            try:
                # Get the result (this should not block since done_flag is True)
                self.async_result.get(timeout=0.1)
                
                # Extract results from shared memory
                proposes = self.async_plan.get('proposes', None)
                
                # Clear async state
                self.async_result = None
                self.vlm_running = False
                self.vlm_done_flag.value = False  # Reset the flag
                return proposes, {'ready': True}
            except Exception as e:
                print(f"Error retrieving async result: {e}")
                self.vlm_running = False
                self.async_result = None
                self.vlm_done_flag.value = False  # Reset the flag
                return None, {'ready': True, 'success': False, 'error': str(e)}
        
        return None, {'ready': False}

    def act(self, obs, episode):
        self.steps += 1
        output_dir = os.path.join(self.output_dir, f"step_{self.steps}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Log previous step result
        if obs['rejected']:
            print(f"Agent {self.agent_id}: ❌ FAILED ({obs.get('rejected_reason', 'Unknown reason')})")
            self.real_action_history[-1] += " - failed"
            self.action_history[-1] = {"type": None, "prompt": "wait"}
        else:
            print(f"Agent {self.agent_id}: ✅ SUCCESS")
        if self.action_history[-1]["prompt"].startswith("pick up"):
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
        elif self.action_history[-1]["prompt"].startswith("place"):
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
        if self.task == 'cook':
            num = [0, 0]
            for agent_idx in range(2):
                plate_pos = self.dest_plates[agent_idx]
                last_height = -5
                for i in range(len(self.recipe[agent_idx])):
                    name = self.recipe[agent_idx][i]
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
                    num[agent_idx] = len(self.recipe[agent_idx])
            progress_info = num

        proposes, info = self.check_async_vlm_result()
        print("proposes: ", proposes)
        is_vlm_action = False
        if proposes is not None and len(proposes) > 0:
            if proposes[0][0] not in available_plans:
                proposes = None
            else:
                proposes = proposes[0]
                is_vlm_action = True
                # Update cache with VLM recommendation
                if self.prev_action_at_vlm_submit is not None:
                    self.action_cache.update(self.prev_action_at_vlm_submit, proposes[0])
                    self.prev_action_at_vlm_submit = None

        # If VLM proposes not available, use cache
        if proposes is None or len(proposes) == 0:
            if len(self.real_action_history) > 0:
                # Find the most recent non-failed, non-wait action
                base_action = None
                for i in range(len(self.real_action_history) - 1, -1, -1):
                    if "failed" not in self.real_action_history[i].lower() and "wait" not in self.real_action_history[i].lower():
                        base_action = self.real_action_history[i].replace(" - vlm", "")
                        break
                # Fallback to last action if all are failed
                if base_action is None:
                    base_action = "wait"
            else:
                # First step: use empty string so cache maps to "none" bigram
                base_action = ""
            
            cached_action = self.action_cache.access(base_action, available_plans, progress=progress_info, real_action_history=self.real_action_history)
            proposes = [cached_action]

        action = {"type": None, "prompt": proposes[0]}
        action_to_log = action["prompt"] + " - vlm" if is_vlm_action else action["prompt"]
        self.real_action_history.append(action_to_log)
        self.action_history.append(action)
        
        # Submit VLM query only if not currently running and available_plans has meaningful actions
        # Skip if available_plans is empty or only contains "wait", or if my recipe is finished
        # Also submit if we used cache action (is_vlm_action is False)
        has_meaningful_actions = not (len(available_plans) == 1 and "wait" in available_plans[0].lower())

        if (not self.vlm_running) and (has_meaningful_actions) and (not is_vlm_action):
            self.prev_action_at_vlm_submit = action["prompt"]
            self.submit_async_vlm_query([inpainting_top_down], self.agent_id, output_dir, self.progress, available_plans, self.steps, self.action_history, episode)
        
        return action

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


