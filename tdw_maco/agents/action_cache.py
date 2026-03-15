import os
import re
import json
import math
import random
from collections import Counter
from typing import Dict, List, Tuple, Optional


# Cook action types that get subdivided by item relevance (my_next/opp_next/other)
COOK_SUBDIVIDED_TYPES = {
    "pick_from_private_region", "pick_from_cutting_board",
    "place_on_cutting_board", "place_in_private_region",
}


def extract_item_from_action(action_text: str) -> str:
    """Extract item name from action text (e.g. 'pick up the cheese_slice from ...' -> 'cheese_slice')."""
    action_text = action_text.lower().strip()
    m = re.match(r'pick up the (.+?) from ', action_text)
    if m:
        return m.group(1)
    m = re.match(r'place the (.+?) onto ', action_text)
    if m:
        return m.group(1)
    m = re.match(r'cut the (.+)', action_text)
    if m:
        return m.group(1)
    return ""


def get_base_item(item: str) -> str:
    """Strip whole_/slice suffixes to get base ingredient name."""
    item = item.lower().strip()
    item = item.replace("whole_", "")
    item = item.replace("_whole", "")
    item = item.replace("_slice", "")
    return item


def items_match(item1: str, item2: str) -> bool:
    """Check if two items refer to the same base ingredient."""
    if not item1 or not item2:
        return False
    return get_base_item(item1) == get_base_item(item2)


def parse_action_type(action_text: str) -> Optional[str]:
    """
    Parse complex action text to extract the action type.
    
    Examples:
        "pick up the brown_bread_slice from the private region" -> "pick_from_private_region"
        "pick up the whole_onion from the cutting board" -> "pick_from_cutting_board"
        "cut the whole_onion" -> "cut"
        "wait" -> "wait"
        "place the cheese_slice on the plate" -> "place_on_plate"
    
    Args:
        action_text: The complex action description
        
    Returns:
        The action type string, or None if pattern not recognized
    """
    action_text = action_text.lower().strip()
    
    # Pattern matching for different action types
    if action_text == "wait":
        return "wait"
    
    # Pick actions
    if "pick up" in action_text or "pick" in action_text:
        if "from the private region" in action_text or "from private region" in action_text:
            return "pick_from_private_region"
        elif "from the cutting board" in action_text or "from cutting board" in action_text:
            return "pick_from_cutting_board"
        elif "from the shared region" in action_text or "from shared region" in action_text:
            return "pick_from_shared_region"
    
    # Place actions
    if "place" in action_text or "put" in action_text:
        if "on the plate" in action_text or "on plate" in action_text:
            return "place_on_plate"
        elif "on the cutting board" in action_text or "on cutting board" in action_text:
            return "place_on_cutting_board"
        elif "in the private region" in action_text or "in private region" in action_text:
            return "place_in_private_region"
        elif "on the shared region" in action_text or "on shared region" in action_text:
            return "place_on_shared_region"
        elif "in the puzzle" in action_text or "in puzzle" in action_text:
            return "place_in_puzzle"
    
    # Cut action
    if action_text.startswith("cut ") or action_text == "cut":
        return "cut"
    
    return None



class ActionCache:
    """
    Cache for storing and retrieving available action plans.
    Contains its own get_available_plans method for generating action plans.
    """
    def __init__(self, task: str, agent_id=None, recipe=None):
        """
        Initialize ActionCache.
        
        Args:
            task: Task type ('game' or 'cook')
            agent_id: Agent ID
            recipe: Recipe information for cook task (optional)
        """
        self.task = task.lower()
        self.agent_id = agent_id
        self.action_rates: Dict[str, float] = {}  # action -> selection rate
        self.success_bigrams: Dict[str, List[Tuple[str, int]]] = {}
        self.real_failure_bigrams: Dict[str, List[Tuple[str, int]]] = {}
        
        # Store task-specific information
        if self.task == 'cook':
            self.recipe = recipe
        
        self._load_action_data()
    
    def parse_action_prompt_to_cache_format(self, action_prompt: str,
                                            my_next_needed: str = None,
                                            opp_next_needed: str = None,
                                            my_recipe: list = None,
                                            opp_recipe: list = None) -> str:
        """
        Parse natural language action prompt to cache format action type.
        Supports both Cook and Game tasks with their specific action patterns.
        
        For cook pick/place (except plate), appends relevance suffix:
          _my_next / _opp_next / _my_later / _opp_later / _neither
        Falls back to _other if full recipe unavailable.
        
        Args:
            action_prompt: Natural language action prompt
            my_next_needed: My next needed recipe item (cook only, optional)
            opp_next_needed: Opponent's next needed recipe item (cook only, optional)
            my_recipe: My full/remaining recipe list (cook only, optional)
            opp_recipe: Opponent's full recipe list (cook only, optional)
            
        Returns:
            Action type string in cache format (e.g., "pick_from_private_region_my_next")
            Returns "unknown" if pattern not recognized
        """
        action_prompt = action_prompt.lower().strip()
        
        # No previous action (first step)
        if action_prompt in ("", "none"):
            return "none"
        
        # Wait action
        if action_prompt == "wait":
            return "wait"
        
        # Cut action (Cook only)
        if action_prompt.startswith("cut ") or action_prompt == "cut":
            return "cut"
        
        base_type = None
        
        # Pick actions
        if "pick up" in action_prompt or "pick" in action_prompt:
            if self.task == "cook":
                if "from the private region" in action_prompt:
                    base_type = "pick_from_private_region"
                elif "from the cutting board" in action_prompt:
                    base_type = "pick_from_cutting_board"
            elif self.task == "game":
                if "private region" in action_prompt:
                    return "pick_from_private_region"
                elif "left border" in action_prompt:
                    return "pick_from_left_border"
                elif "right border" in action_prompt:
                    return "pick_from_right_border"
        
        # Place actions
        if base_type is None and ("place" in action_prompt or "put" in action_prompt):
            if self.task == "cook":
                if "onto the plate" in action_prompt:
                    return "place_on_plate"
                elif "onto the cutting board" in action_prompt:
                    base_type = "place_on_cutting_board"
                elif "private region" in action_prompt:
                    base_type = "place_in_private_region"
            elif self.task == "game":
                if "of the puzzle box" in action_prompt:
                    return "place_in_puzzle"
                elif "private region" in action_prompt:
                    return "place_in_private_region"
                elif "left border" in action_prompt:
                    return "place_on_left_border"
                elif "right border" in action_prompt:
                    return "place_on_right_border"
        
        # Subdivide cook pick/place by item relevance
        if base_type is not None:
            if (my_next_needed or opp_next_needed) and base_type in COOK_SUBDIVIDED_TYPES:
                item = extract_item_from_action(action_prompt)
                if item:
                    if my_next_needed and items_match(item, my_next_needed):
                        suffix = "_my_next"
                    elif opp_next_needed and items_match(item, opp_next_needed):
                        suffix = "_opp_next"
                    elif my_recipe and any(items_match(item, r) for r in my_recipe):
                        suffix = "_my_later"
                    elif opp_recipe and any(items_match(item, r) for r in opp_recipe):
                        suffix = "_opp_later"
                    elif my_recipe or opp_recipe:
                        suffix = "_neither"
                    else:
                        suffix = "_other"
                    # Add cut status for cutting board actions
                    if "cutting_board" in base_type:
                        cut_suffix = "_to_cut" if item.startswith("whole_") else "_ready"
                        return f"{base_type}{suffix}{cut_suffix}"
                    return f"{base_type}{suffix}"
            return base_type
        
        return "unknown"
    
    def _load_action_data(self):
        """
        Load action selection rates and bigram data from JSON cache file.
        Falls back to txt files if JSON not found.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        json_file = os.path.join(base_path, f"{self.task}_cache_data.json")
        
        if os.path.exists(json_file):
            self._load_from_json(json_file)
        else:
            # Fallback to txt files
            selection_file = os.path.join(base_path, f"{self.task}_action_selection_report.txt")
            ngram_file = os.path.join(base_path, f"{self.task}_ngram_analysis.txt")
            self._load_selection_rates_txt(selection_file)
            self._load_bigram_data_txt(ngram_file)
    
    def _load_from_json(self, filepath: str):
        """
        Load all cache data from structured JSON file.
        JSON contains: selection_rates, bigrams (overall, success, failure, progress_*).
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load selection rates
        for action_type, stats in data.get("selection_rates", {}).items():
            self.action_rates[action_type] = stats["rate"]
        
        # Helper to convert JSON bigram list to tuple list
        def parse_bigram_dict(bigram_dict):
            result = {}
            for prev_action, next_list in bigram_dict.items():
                result[prev_action] = [(entry[0], entry[1]) for entry in next_list]
            return result
        
        bigrams = data.get("bigrams", {})
        self.success_bigrams = parse_bigram_dict(bigrams.get("success", {}))
        self.real_failure_bigrams = parse_bigram_dict(bigrams.get("real_failure", {}))
    
    def _load_selection_rates_txt(self, filepath: str):
        """
        Parse action selection report to extract selection rates (txt fallback).
        """
        with open(filepath, 'r') as f:
            for line in f:
                match = re.match(r'\d+\.\s+([\w_]+):\s+\d+/\d+\s+\(([\d.]+)%\)', line.strip())
                if match:
                    action_name = match.group(1)
                    rate = float(match.group(2)) / 100.0
                    self.action_rates[action_name] = rate
    
    def _load_bigram_data_txt(self, filepath: str):
        """
        Parse ngram analysis to extract bigram counts (txt fallback).
        """
        with open(filepath, 'r') as f:
            for line in f:
                match = re.match(r'\d+\.\s+\[([\w_]+)\s+->\s+([\w_]+)\]:\s+(\d+)\s+\([\d.]+%\)', line.strip())
                if match:
                    prev_action = match.group(1)
                    next_action = match.group(2)
                    count = int(match.group(3))
                    if prev_action not in self.success_bigrams:
                        self.success_bigrams[prev_action] = []
                    self.success_bigrams[prev_action].append((next_action, count))
    
    @staticmethod
    def _get_repetition_excluded(real_action_history: List[str],
                                  window: int = 6, max_repeat: int = 2) -> set:
        """Return raw action strings that appeared >= max_repeat times in the
        last `window` actions.  'wait' is never excluded."""
        if not real_action_history or len(real_action_history) < max_repeat:
            return set()
        recent = real_action_history[-window:]
        cleaned = []
        for a in recent:
            c = a.replace(" - vlm", "")
            for tag in (" - failed", " - FAILED"):
                if tag in c:
                    c = c.split(tag)[0]
            cleaned.append(c.strip())
        counts = Counter(cleaned)
        return {act for act, cnt in counts.items()
                if cnt >= max_repeat and act.lower() != "wait"}

    @staticmethod
    def _swap_cut_ready(action_type: str) -> Optional[str]:
        """Swap _to_cut <-> _ready suffix for fallback matching."""
        if action_type.endswith('_to_cut'):
            return action_type[:-7] + '_ready'
        elif action_type.endswith('_ready'):
            return action_type[:-6] + '_to_cut'
        return None

    @staticmethod
    def _game_piece_suffix(prev_piece: str, action: str) -> str:
        """Return '_A' if action's piece matches prev_piece or prev has no piece.
        '_B' if different piece. '' if current action has no piece (e.g. wait)."""
        curr_piece = extract_item_from_action(action)
        if not curr_piece:
            return ""
        if not prev_piece:
            return "_A"
        return "_A" if curr_piece == prev_piece else "_B"

    @staticmethod
    def _should_fast_track_action(action_type: str, rate: float) -> bool:
        return rate == 1.0 or action_type.startswith("place_on_plate") or action_type.startswith("place_in_puzzle")

    def _select_bigrams(self, prev_action: str, prev_action_success: bool, progress=None,
                         real_failed_action: str = None):
        """
        Select the best bigram table based on context.
        If action failed and real_failure bigrams exist for it, use those.
        Otherwise use success bigrams.
        
        Returns:
            Tuple of (bigram_dict, use_failed_key: bool)
        """
        # If action failed, try real_failure bigrams first
        if real_failed_action and real_failed_action in self.real_failure_bigrams:
            return self.real_failure_bigrams, True
        
        # Use success bigrams
        return self.success_bigrams, False
    
    def get_next_action(self, prev_action: str) -> Optional[str]:
        """
        Get the next action with the highest score based on previous action.
        
        Score = log(count) * selection_rate
        
        Args:
            prev_action: The previous action taken
            
        Returns:
            The next action with highest score, or None if no data available
        """
        if prev_action not in self.success_bigrams:
            return None
        
        best_action = None
        best_score = -1
        
        # Calculate score for each possible next action
        for next_action, count in self.success_bigrams[prev_action]:
            # Get selection rate for this action (default to 0 if not found)
            rate = self.action_rates.get(next_action, 0.0)
            
            # Calculate score: count * rate
            score = count * rate
            
            if score > best_score:
                best_score = score
                best_action = next_action
        
        return best_action
    
    def get_all_next_actions(self, prev_action: str) -> List[Tuple[str, float]]:
        """
        Get all possible next actions with their scores for a given previous action.
        
        Args:
            prev_action: The previous action taken
            
        Returns:
            List of (next_action, score) tuples sorted by score in descending order
        """
        if prev_action not in self.success_bigrams:
            return []
        
        action_scores = []
        
        for next_action, count in self.success_bigrams[prev_action]:
            rate = self.action_rates.get(next_action, 0.0)
            score = count * rate
            action_scores.append((next_action, score))
        
        # Sort by score in descending order
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        return action_scores  # Returns all possible next actions sorted by score (log(count) * rate)

    def update(self, prev_action: str, vlm_action: str):
        """
        Update bigram counts based on VLM recommendation.
        Called when a VLM result returns, using the prev_action at query submission time.
        
        Args:
            prev_action: The action that was taken when VLM query was submitted
            vlm_action: The action recommended by VLM
        """
        parsed_prev = self.parse_action_prompt_to_cache_format(prev_action)
        parsed_next = self.parse_action_prompt_to_cache_format(vlm_action)
        
        if parsed_prev == "unknown" or parsed_next == "unknown":
            return
        
        # Update bigram: increment count if pair exists, otherwise add new entry
        if parsed_prev in self.success_bigrams:
            found = False
            for i, (action, count) in enumerate(self.success_bigrams[parsed_prev]):
                if action == parsed_next:
                    self.success_bigrams[parsed_prev][i] = (action, count + 1)
                    found = True
                    break
            if not found:
                self.success_bigrams[parsed_prev].append((parsed_next, 1))
        else:
            self.success_bigrams[parsed_prev] = [(parsed_next, 1)]
        
        # Update selection rate: nudge toward 1.0 for VLM-endorsed actions
        if parsed_next not in self.action_rates:
            self.action_rates[parsed_next] = 0.5
        
        print(f"[ActionCache] Updated bigram: {parsed_prev} -> {parsed_next}")

    def access(self, prev_action: str, possible_actions: List[str], progress=None, real_action_history=None) -> Optional[str]:
        """
        Get the best next action from the list of possible actions based on previous action.
        
        Filters the cached next actions by the possible_actions list and returns
        the one with the highest score (log(count) * selection_rate).
        
        Args:
            prev_action: The previous action taken
            possible_actions: List of currently available/possible actions to choose from
            progress: Progress information (num array for cook task, None for game task)
            
        Returns:
            The best action from possible_actions with highest score, or None if no match found
        """
        if self.task == "cook":
            return self.access_cook(prev_action, possible_actions, progress, real_action_history)
        else:
            return self.access_game(prev_action, possible_actions, progress, real_action_history)
    # GAME task actions: place_in_puzzle, place_on_shared_region, pick_from_shared_region, 
    #                    pick_from_private_region, wait, place_in_private_region
    # COOK task actions: place_on_plate, cut, pick_from_private_region, pick_from_cutting_board,
    #                    wait, place_in_private_region, place_on_cutting_board
    def access_cook(self, prev_action: str, possible_actions: List[str], progress=None,real_action_history=None) -> Optional[str]:
        """
        Get the best next action from the list of possible actions based on previous action.
        
        Filters the cached next actions by the possible_actions list and returns
        the one with the highest score (log(count) * selection_rate).
        
        Args:
            prev_action: The previous action taken
            possible_actions: List of currently available/possible actions to choose from
            progress: Progress information (num array for cook task, None for game task)
            real_action_history: List of previous actions taken
            
        Returns:
            The best action from possible_actions with highest score, or None if no match found
        """
        print("possible_actions: ", possible_actions)
        # Compute my_next_needed / opp_next_needed and full recipe lists for item relevance tagging
        my_next_needed = None
        opp_next_needed = None
        my_recipe_list = None
        opp_recipe_list = None
        if self.recipe and progress is not None:
            my_recipe_idx = self.agent_id if self.agent_id is not None else 0
            opp_recipe_idx = 1 - my_recipe_idx
            my_progress = progress[my_recipe_idx]
            if my_progress < len(self.recipe[my_recipe_idx]):
                my_next_needed = self.recipe[my_recipe_idx][my_progress]
            opp_progress = progress[opp_recipe_idx]
            if opp_progress < len(self.recipe[opp_recipe_idx]):
                opp_next_needed = self.recipe[opp_recipe_idx][opp_progress]
            # Remaining recipe items (excluding next) for my_later/opp_later classification
            my_recipe_list = self.recipe[my_recipe_idx][my_progress:]
            opp_recipe_list = self.recipe[opp_recipe_idx]
        
        # Parse action prompts to cache format (with item relevance)
        parsed_prev_action = self.parse_action_prompt_to_cache_format(
            prev_action, my_next_needed, opp_next_needed, my_recipe_list, opp_recipe_list)
        
        # === Score-based selection (n-gram driven) ===
        prev_action_success = not (real_action_history and "failed" in real_action_history[-1].lower())
        
        # Extract real failed action from history (env converts failures to 'wait')
        real_failed_action = None
        if not prev_action_success and real_action_history:
            failed_text = real_action_history[-1].replace(" - vlm", "").split(" - failed")[0].split(" - FAILED")[0]
            real_failed_action = self.parse_action_prompt_to_cache_format(
                failed_text, my_next_needed, opp_next_needed, my_recipe_list, opp_recipe_list)
        
        active_bigrams, use_failed_key = self._select_bigrams(parsed_prev_action, prev_action_success, progress,
                                               real_failed_action=real_failed_action)
        
        # Determine lookup key: use real_failed_action only if real_failure bigrams were selected
        lookup_key = real_failed_action if use_failed_key else parsed_prev_action
        
        if lookup_key not in active_bigrams:
            return "wait"

        # Hard-exclude recently repeated actions to prevent oscillation loops
        excluded = self._get_repetition_excluded(real_action_history)
        if excluded:
            filtered = [a for a in possible_actions if a not in excluded]
            if filtered:
                possible_actions = filtered

        # Create mapping from parsed action to list of original actions
        parsed_to_original = {}
        for action in possible_actions:
            parsed = self.parse_action_prompt_to_cache_format(
                action, my_next_needed, opp_next_needed, my_recipe_list, opp_recipe_list)
            if parsed not in parsed_to_original:
                parsed_to_original[parsed] = []
            parsed_to_original[parsed].append(action)
        
        parsed_possible_actions = list(parsed_to_original.keys())

        # Fast-track: always pick my_next item immediately
        if "pick_from_cutting_board_my_next_ready" in parsed_to_original:
            return random.choice(parsed_to_original["pick_from_cutting_board_my_next_ready"])
        if "pick_from_private_region_my_next" in parsed_to_original:
            return random.choice(parsed_to_original["pick_from_private_region_my_next"])

        score_to_actions = {}
        for next_action, count in active_bigrams[lookup_key]:
            # Try exact match, then _to_cut/_ready fallback
            matched_key = None
            if next_action in parsed_possible_actions:
                matched_key = next_action
            else:
                alt = self._swap_cut_ready(next_action)
                if alt and alt in parsed_possible_actions:
                    matched_key = alt
            if matched_key:
                rate = self.action_rates.get(next_action, 0.0)
                if self._should_fast_track_action(matched_key, rate):
                    return random.choice(parsed_to_original[matched_key])
                score = round(count * rate, 6)
                if score not in score_to_actions:
                    score_to_actions[score] = []
                score_to_actions[score].extend(parsed_to_original[matched_key])
        
        sorted_actions = []
        for score in sorted(score_to_actions.keys(), reverse=True):
            actions_with_same_score = score_to_actions[score].copy()
            random.shuffle(actions_with_same_score)
            sorted_actions.extend(actions_with_same_score)
        print("sorted_actions", sorted_actions)
        if sorted_actions:
            return sorted_actions[0]
        return "wait"

    def access_game(self, prev_action: str, possible_actions: List[str], progress=None, real_action_history=None) -> Optional[str]:
        """
        Get the best next action from the list of possible actions based on previous action.
        
        Filters the cached next actions by the possible_actions list and returns
        the one with the highest score (log(count) * selection_rate).
        
        Args:
            prev_action: The previous action taken
            possible_actions: List of currently available/possible actions to choose from
            progress: Progress information (num array for cook task, None for game task)
            real_action_history: List of previous actions taken
            
        Returns:
            The best action from possible_actions with highest score, or None if no match found
        """
        
        # Parse action prompts to cache format (lookup key: no A/B suffix)
        parsed_prev_action = self.parse_action_prompt_to_cache_format(prev_action)
        prev_piece = extract_item_from_action(prev_action.lower())
        
        # === Score-based selection (n-gram driven, no ad-hoc filters) ===
        prev_action_success = not (real_action_history and "failed" in real_action_history[-1].lower())
        
        # Extract real failed action from history (env converts failures to 'wait')
        real_failed_action = None
        real_failed_piece = None
        if not prev_action_success and real_action_history:
            failed_text = real_action_history[-1].replace(" - vlm", "").split(" - failed")[0].split(" - FAILED")[0]
            real_failed_action = self.parse_action_prompt_to_cache_format(failed_text)
            real_failed_piece = extract_item_from_action(failed_text.lower())
        
        active_bigrams, use_failed_key = self._select_bigrams(parsed_prev_action, prev_action_success, progress,
                                               real_failed_action=real_failed_action)
        
        # Determine lookup key: use real_failed_action only if real_failure bigrams were selected
        lookup_key = real_failed_action if use_failed_key else parsed_prev_action
        ref_piece = real_failed_piece if use_failed_key else prev_piece
        
        if lookup_key not in active_bigrams:
            return "wait"

        # Hard-exclude recently repeated pick actions to prevent oscillation loops
        # excluded = {a for a in self._get_repetition_excluded(real_action_history, max_repeat=1)
        excluded = {a for a in self._get_repetition_excluded(real_action_history)
                    if "pick" in a.lower()}
        if excluded:
            filtered = [a for a in possible_actions if a not in excluded]
            if filtered:
                possible_actions = filtered

        # Create mapping from parsed action to list of original actions
        # Candidates get _A/_B suffix based on piece match with prev action
        parsed_to_original = {}
        for action in possible_actions:
            parsed = self.parse_action_prompt_to_cache_format(action)
            parsed += self._game_piece_suffix(ref_piece, action)
            if parsed not in parsed_to_original:
                parsed_to_original[parsed] = []
            parsed_to_original[parsed].append(action)
        
        parsed_possible_actions = list(parsed_to_original.keys())
        
        score_to_actions = {}
        for next_action, count in active_bigrams[lookup_key]:
            if next_action in parsed_possible_actions:
                rate = self.action_rates.get(next_action, 0.0)
                if self._should_fast_track_action(next_action, rate):
                    return random.choice(parsed_to_original[next_action])
                score = round(count * rate, 6)
                if score not in score_to_actions:
                    score_to_actions[score] = []
                score_to_actions[score].extend(parsed_to_original[next_action])
        
        sorted_actions = []
        for score in sorted(score_to_actions.keys(), reverse=True):
            actions_with_same_score = score_to_actions[score].copy()
            random.shuffle(actions_with_same_score)
            sorted_actions.extend(actions_with_same_score)
        print("sorted_actions", sorted_actions)
        if sorted_actions:
            return sorted_actions[0]
        return "wait"
    
    def print_cache_entries(self):
        """
        Print all cache entries (action rates and bigrams).
        """
        print(f"\n=== Action Cache Entries (Task: {self.task}) ===\n")
        
        # Print action selection rates
        print("Action Selection Rates:")
        for action, rate in sorted(self.action_rates.items(), key=lambda x: x[1], reverse=True):
            print(f"  {action}: {rate:.2%}")
        
        # Print bigram entries
        print(f"\nAction Bigrams (Total: {len(self.success_bigrams)} prev actions):")
        for prev_action, next_actions in sorted(self.success_bigrams.items()):
            print(f"  {prev_action} ->")
            # Calculate scores and sort by score in descending order
            scored_actions = []
            for next_action, count in next_actions:
                rate = self.action_rates.get(next_action, 0.0)
                score = count * rate
                scored_actions.append((next_action, count, rate, score))
            
            # Sort by score (descending)
            for next_action, count, rate, score in sorted(scored_actions, key=lambda x: x[3], reverse=True):
                print(f"    {next_action}: count={count}, rate={rate:.2%}, score={score:.3f}")
        print()