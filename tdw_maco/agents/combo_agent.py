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
from agents.game_plan_agent import convert_pos

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
            self.relative_position_dict = info["relative_position_dict"].copy()
            # puzzle_ids will be determined in act() based on proximity
        
        # Set trace log path in episode dir (one level up from agent dir)
        self.trace_log_path = os.path.join(output_dir, f"{self.agent_id}_trace.jsonl")

        self.proposer.reset(self.recipe, self.agents_name)
        if not self.only_propose:
            self.ranker.reset(self.recipe, self.agents_name)

        if not self.only_propose and not self.no_belief:
            self.intenttracker.reset(self.recipe, self.agents_name)

    def write_log_to_file(self,log_message, file_name=None):
        file_name = self.record_dir
        with open(file_name, 'a') as file:  
            file.write(log_message + '\n')  

    def act(self, obs, episode):
        self.steps += 1
        output_dir = os.path.join(self.output_dir, f"step_{self.steps}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Log previous step result
        if obs['rejected']:
            self.write_log_to_file(f"Agent {self.agent_id}: ❌ FAILED ({obs.get('rejected_reason', 'Unknown reason')})")
            self.write_log_to_file(f"######Episode {episode}/Step {self.steps-1}######")
            self.action_history[-1] = {"type": None, "prompt": "wait"}
        else:
            self.write_log_to_file(f"Agent {self.agent_id}: ✅ SUCCESS {self.action_history[-1]['prompt']}")
            self.write_log_to_file(f"######Episode {episode}/Step {self.steps-1}######")
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

        # step3: propose actions
        proposes = self.proposer.run([inpainting_top_down], self.agent_id, output_dir, self.progress, available_plans, steps=self.steps, action_history=self.action_history, episode=episode)[0]

        selected = proposes[0]

        # Build task-specific trace metadata
        extra_info = {}
        if self.task == 'game':
            # Compute puzzle progress: how many pieces are correctly placed
            id2pos = {obj['id']: obj['pos'] for obj in obs["objects"]}
            completed = 0
            total = 0
            for puzzle_id in self.puzzle_id2piece_id:
                if puzzle_id not in id2pos:
                    continue
                for piece_id in self.puzzle_id2piece_id[puzzle_id]:
                    if piece_id not in id2pos:
                        continue
                    total += 1
                    target_pos = id2pos[puzzle_id] + convert_pos(int(self.agent_id), self.relative_position_dict[piece_id])
                    if np.linalg.norm(np.array(id2pos[piece_id]) - np.array(target_pos)) <= IN_PUZZLE_THRESHOLD:
                        completed += 1
            extra_info = {"puzzle_completed": completed, "puzzle_total": total}
        elif self.task == 'cook' and self.recipe is not None:
            my_idx = self.agent_id
            opp_idx = 1 - my_idx
            my_remaining = self.recipe[my_idx][self.progress:] if self.progress < len(self.recipe[my_idx]) else []
            opp_remaining = self.recipe[opp_idx]
            # What does my recipe need next?
            my_next_needed = self.recipe[my_idx][self.progress] if self.progress < len(self.recipe[my_idx]) else None
            # Is the selected item relevant to my recipe or opponent's?
            item_in_selected = None
            if selected and selected != 'wait' and 'the ' in selected:
                parts = selected.split('the ', 1)
                if len(parts) > 1:
                    item_in_selected = parts[1].split(' from')[0].split(' onto')[0].strip()
            item_base = item_in_selected.replace('_slice','').replace('_whole','') if item_in_selected else None
            my_next_base = my_next_needed.replace('_slice','').replace('_whole','') if my_next_needed else None
            opp_next_needed = opp_remaining[0] if len(opp_remaining) > 0 else None
            opp_next_base = opp_next_needed.replace('_slice','').replace('_whole','') if opp_next_needed else None
            extra_info = {
                "my_recipe_remaining": my_remaining,
                "opp_recipe_full": list(opp_remaining),
                "my_next_needed": my_next_needed,
                "opp_next_needed": opp_next_needed,
                "selected_item": item_in_selected,
                "selected_is_my_next": item_base == my_next_base if item_base else False,
                "selected_is_opp_next": item_base == opp_next_base if item_base else False,
                "selected_in_my_recipe": item_base in [r.replace('_slice','').replace('_whole','') for r in my_remaining] if item_base else False,
                "selected_in_opp_recipe": item_base in [r.replace('_slice','').replace('_whole','') for r in opp_remaining] if item_base else False,
            }

        # Append trace record to JSONL
        prev_action_prompt = self.action_history[-1]['prompt'] if self.action_history else 'wait'
        trace_record = {
            "episode": episode,
            "step": self.steps,
            "agent_id": self.agent_id,
            "prev_action": prev_action_prompt,
            "prev_action_success": not obs['rejected'],
            "selected_action": selected,
            "available_actions": available_plans,
            "progress": self.progress,
            "object_in_hand": self.object_in_hand,
            **extra_info,
        }
        with open(self.trace_log_path, 'a') as f:
            f.write(json.dumps(trace_record) + '\n')

        if self.only_propose:
            action = {"type": None, "prompt": selected}
            self.action_history.append(action)
            return action
        
        action = {"type": None, "prompt": selected}
        self.action_history.append(action)
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


