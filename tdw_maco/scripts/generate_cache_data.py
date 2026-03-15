#!/usr/bin/env python3
"""
Generate action cache data from trace.jsonl files.
Produces:
  - Selection rates (action_selection_report.txt)
  - Bigram analysis (ngram_analysis.txt)  
  - Structured JSON with conditioned bigrams (cache_data.json)

Usage:
  python generate_cache_data.py
"""

import re
import json
from collections import Counter, defaultdict
from pathlib import Path


# Cook action types that get subdivided by item relevance
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


def get_recipe_context(step: dict) -> dict:
    """Extract recipe context from a trace step for parse_action_type."""
    return {
        "my_next_needed": step.get("my_next_needed") or None,
        "opp_next_needed": step.get("opp_next_needed") or None,
        "my_recipe": step.get("my_recipe_remaining") or None,
        "opp_recipe": step.get("opp_recipe_full") or None,
    }


def parse_action_type(action_text: str, task: str,
                      my_next_needed: str = None, opp_next_needed: str = None,
                      my_recipe: list = None, opp_recipe: list = None) -> str:
    """
    Parse action text to simplified action type.
    
    For cook pick/place (except plate), appends relevance suffix:
      _my_next / _opp_next / _my_later / _opp_later / _neither
    when recipe context is provided. Falls back to _other if full recipe unavailable.
    
    Cook types: pick_from_private_region[_my_next|_opp_next|_my_later|_opp_later|_neither], ...
                place_on_plate, cut, wait
    Game types: unchanged (no recipe concept)
    """
    action_text = action_text.lower().strip()
    
    if action_text == "wait":
        return "wait"
    
    base_type = None
    
    if task == "cook":
        if action_text.startswith("cut "):
            return "cut"
        # Check place before pick (avoid "pickle" matching "pick")
        if "place" in action_text:
            if "plate" in action_text:
                return "place_on_plate"
            elif "cutting board" in action_text:
                base_type = "place_on_cutting_board"
            elif "private region" in action_text:
                base_type = "place_in_private_region"
        if base_type is None and ("pick up" in action_text or "pick" in action_text):
            if "private region" in action_text:
                base_type = "pick_from_private_region"
            elif "cutting board" in action_text:
                base_type = "pick_from_cutting_board"
        
        if base_type is not None:
            # Subdivide by item relevance if context provided
            if (my_next_needed or opp_next_needed) and base_type in COOK_SUBDIVIDED_TYPES:
                item = extract_item_from_action(action_text)
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
    
    elif task == "game":
        # Check place before pick
        if "place" in action_text:
            # Check private region BEFORE puzzle box (some actions mention both)
            if "private region" in action_text:
                return "place_in_private_region"
            elif "left border" in action_text:
                return "place_on_left_border"
            elif "right border" in action_text:
                return "place_on_right_border"
            elif "puzzle box" in action_text:
                return "place_in_puzzle"
        if "pick up" in action_text or "pick" in action_text:
            if "private region" in action_text:
                return "pick_from_private_region"
            elif "left border" in action_text:
                return "pick_from_left_border"
            elif "right border" in action_text:
                return "pick_from_right_border"
    
    return "unknown"


def get_game_piece_suffix(prev_action: str, current_action: str) -> str:
    """Return '_A' if current action's piece matches prev action's piece or
    if either has no piece (e.g. wait). '_B' if different piece."""
    prev_piece = extract_item_from_action(prev_action)
    curr_piece = extract_item_from_action(current_action)
    if not curr_piece:
        return ""
    if not prev_piece:
        return "_A"
    return "_A" if prev_piece == curr_piece else "_B"


def get_progress_bucket(step_data: dict, task: str) -> str:
    """
    Get progress bucket: early/mid/late.
    
    Cook: based on progress integer (items placed on plate)
      early: 0-1, mid: 2-3, late: 4+
    Game: based on puzzle_completed / puzzle_total ratio
      early: <33%, mid: 33-66%, late: >66%
    """
    if task == "cook":
        progress = step_data.get("progress", 0) or 0
        if progress <= 1:
            return "early"
        elif progress <= 3:
            return "mid"
        else:
            return "late"
    elif task == "game":
        completed = step_data.get("puzzle_completed", 0) or 0
        total = step_data.get("puzzle_total", 15) or 15
        ratio = completed / total if total > 0 else 0
        if ratio < 0.33:
            return "early"
        elif ratio < 0.66:
            return "mid"
        else:
            return "late"
    return "early"


def cross_reference_opp_next(trace_pairs: list) -> None:
    """Fix opp_next_needed by cross-referencing paired agent traces.
    
    Each pair is (agent0_steps, agent1_steps) from the same episode.
    For each agent's step, replace opp_next_needed with the partner's
    my_next_needed at the same step number (since partner knows their own progress).
    """
    for agent0_steps, agent1_steps in trace_pairs:
        # Build step -> my_next_needed lookup for each agent
        a0_by_step = {s["step"]: s.get("my_next_needed") for s in agent0_steps}
        a1_by_step = {s["step"]: s.get("my_next_needed") for s in agent1_steps}
        
        # Fix agent0's opp_next from agent1's my_next
        for step in agent0_steps:
            partner_next = a1_by_step.get(step["step"])
            if partner_next is not None:
                step["opp_next_needed"] = partner_next
        
        # Fix agent1's opp_next from agent0's my_next
        for step in agent1_steps:
            partner_next = a0_by_step.get(step["step"])
            if partner_next is not None:
                step["opp_next_needed"] = partner_next


def load_trace_files(trace_paths: list) -> list:
    """Load all steps from multiple trace.jsonl files."""
    all_steps = []
    for path in trace_paths:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_steps.append(json.loads(line))
    return all_steps


def load_trace_files_grouped(trace_paths: list) -> list:
    """Load trace files preserving per-file step sequences.
    Returns list of lists: [[steps_file1], [steps_file2], ...]"""
    grouped = []
    for path in trace_paths:
        file_steps = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    file_steps.append(json.loads(line))
        grouped.append(file_steps)
    return grouped


def compute_real_failure_bigrams(grouped_steps: list, task: str) -> Counter:
    """
    Compute bigrams keyed by the REAL failed action (not 'wait').
    When prev_action_success=False, the environment records prev_action='wait',
    but the actual failed action is the previous step's selected_action.
    Bigram: real_failed_action_type -> next_selected_action_type
    """
    bigrams = Counter()
    
    for file_steps in grouped_steps:
        for i, step in enumerate(file_steps):
            if not step.get("prev_action_success", True) and i > 0:
                prev_step = file_steps[i - 1]
                real_failed = prev_step.get("selected_action", "")
                next_selected = step.get("selected_action", "")
                ctx = get_recipe_context(step)
                
                failed_type = parse_action_type(real_failed, task, **ctx)
                next_type = parse_action_type(next_selected, task, **ctx)
                
                # Game: add _A/_B to next_type based on piece match with failed action
                if task == "game":
                    next_type += get_game_piece_suffix(real_failed, next_selected)
                
                if failed_type != "unknown" and next_type != "unknown":
                    bigrams[(failed_type, next_type)] += 1
    
    return bigrams


def compute_selection_rates(steps: list, task: str) -> dict:
    """
    Compute action selection rates.
    Denominator: # of steps where that action TYPE appeared in available_actions (deduplicated by type)
    Numerator: # of times that action TYPE was selected
    For game: appends _A/_B suffix based on whether action's piece matches prev action's piece.
    """
    stats = defaultdict(lambda: {"chosen": 0, "total": 0})
    
    for step in steps:
        available = step.get("available_actions", [])
        selected = step.get("selected_action", "")
        ctx = get_recipe_context(step)
        prev_action = step.get("prev_action", "")
        
        # Get unique action types from available actions (deduplicate by type)
        available_types = set()
        for action in available:
            action_type = parse_action_type(action, task, **ctx)
            if action_type != "unknown":
                if task == "game":
                    action_type += get_game_piece_suffix(prev_action, action)
                available_types.add(action_type)
        
        # Count each type as available once per step
        for action_type in available_types:
            stats[action_type]["total"] += 1
        
        # Count selected
        selected_type = parse_action_type(selected, task, **ctx)
        if task == "game":
            selected_type += get_game_piece_suffix(prev_action, selected)
        if selected_type in available_types:
            stats[selected_type]["chosen"] += 1
    
    # Calculate rates
    for action_type in stats:
        total = stats[action_type]["total"]
        chosen = stats[action_type]["chosen"]
        stats[action_type]["rate"] = chosen / total if total > 0 else 0.0
    
    return dict(stats)


def compute_bigrams(steps: list, task: str, condition_fn=None) -> Counter:
    """
    Compute bigram frequencies: prev_action_type -> selected_action_type.
    condition_fn: optional function(step) -> bool to filter steps.
    For game: selected_type gets _A/_B suffix (prev_type does NOT).
    """
    bigrams = Counter()
    
    for step in steps:
        if condition_fn and not condition_fn(step):
            continue
        
        prev_action = step.get("prev_action", "")
        selected_action = step.get("selected_action", "")
        ctx = get_recipe_context(step)
        
        # First step has no real prev_action (env initializes as "wait")
        if step.get("step", 0) == 1:
            prev_type = "none"
        else:
            prev_type = parse_action_type(prev_action, task, **ctx)
        selected_type = parse_action_type(selected_action, task, **ctx)
        
        # Game: add _A/_B to selected_type based on piece match with prev_action
        if task == "game":
            selected_type += get_game_piece_suffix(prev_action, selected_action)
        
        if prev_type != "unknown" and selected_type != "unknown":
            bigrams[(prev_type, selected_type)] += 1
    
    return bigrams


def analyze_wait_patterns(steps: list, task: str) -> dict:
    """
    Analyze wait behavior:
    - Consecutive wait streaks (max, avg, all streak lengths)
    - Wait with alternatives: how often wait was chosen despite other options
    """
    # Group steps by (trace file source, episode) to find consecutive streaks
    episodes = defaultdict(list)  # (episode, agent_id) -> [steps in order]
    for step in steps:
        key = (step.get("episode", 0), step.get("agent_id", 0))
        episodes[key].append(step)
    
    # Sort each episode's steps by step number
    for key in episodes:
        episodes[key].sort(key=lambda s: s.get("step", 0))
    
    # Find consecutive wait streaks
    all_streaks = []
    for key, ep_steps in episodes.items():
        streak = 0
        for step in ep_steps:
            ctx = get_recipe_context(step)
            selected_type = parse_action_type(step.get("selected_action", ""), task, **ctx)
            if selected_type == "wait":
                streak += 1
            else:
                if streak > 0:
                    all_streaks.append(streak)
                streak = 0
        if streak > 0:
            all_streaks.append(streak)
    
    # Wait with alternatives analysis
    wait_total = 0
    wait_with_alternatives = 0
    wait_only_option = 0
    wait_alternative_types = Counter()  # what other options were available when agent chose wait
    
    for step in steps:
        ctx = get_recipe_context(step)
        selected_type = parse_action_type(step.get("selected_action", ""), task, **ctx)
        
        if selected_type != "wait":
            continue
        
        wait_total += 1
        available = step.get("available_actions", [])
        non_wait = [a for a in available if parse_action_type(a, task, **ctx) != "wait"]
        
        if non_wait:
            wait_with_alternatives += 1
            for a in non_wait:
                atype = parse_action_type(a, task, **ctx)
                if atype != "unknown":
                    wait_alternative_types[atype] += 1
        else:
            wait_only_option += 1
    
    return {
        "streaks": sorted(all_streaks, reverse=True),
        "max_streak": max(all_streaks) if all_streaks else 0,
        "avg_streak": sum(all_streaks) / len(all_streaks) if all_streaks else 0,
        "wait_total": wait_total,
        "wait_with_alternatives": wait_with_alternatives,
        "wait_only_option": wait_only_option,
        "wait_alternative_types": wait_alternative_types,
    }


# ── Output writers ──────────────────────────────────────────────────────────

def write_selection_report(stats: dict, output_path: str, task_name: str):
    """Write selection rate report in txt format (same format as before)."""
    sorted_actions = sorted(
        stats.items(),
        key=lambda x: x[1]["rate"],
        reverse=True
    )
    
    with open(output_path, 'w') as f:
        f.write(f"Task: {task_name.upper()}\n")
        f.write(f"Analysis: Action Selection Rates\n")
        f.write("=" * 80 + "\n\n")
        
        for i, (action_type, s) in enumerate(sorted_actions, 1):
            rate_pct = s['rate'] * 100
            f.write(f"{i}. {action_type}: {s['chosen']}/{s['total']} ({rate_pct:.2f}%)\n")
        f.write("\n")


def write_ngram_report(bigrams: Counter, output_path: str, task_name: str,
                       wait_analysis: dict = None, real_failure_bigrams: Counter = None):
    """Write ngram report in txt format, with optional wait pattern analysis."""
    grouped = defaultdict(list)
    for (prev, next_), count in bigrams.items():
        grouped[prev].append((next_, count))
    
    for prev in grouped:
        grouped[prev].sort(key=lambda x: x[1], reverse=True)
    
    action_totals = {prev: sum(c for _, c in nexts) for prev, nexts in grouped.items()}
    sorted_actions = sorted(action_totals.items(), key=lambda x: x[1], reverse=True)
    
    total_bigrams = sum(bigrams.values())
    
    with open(output_path, 'w') as f:
        f.write(f"Task: {task_name.upper()}\n")
        f.write(f"Analysis: Action Bigrams (2-grams)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total bigrams: {total_bigrams:,}\n")
        f.write(f"Unique bigrams: {len(bigrams)}\n")
        f.write(f"Unique action types: {len(grouped)}\n\n")
        
        # Wait pattern analysis section
        if wait_analysis:
            f.write("=" * 80 + "\n")
            f.write("WAIT PATTERN ANALYSIS\n")
            f.write("-" * 80 + "\n")
            wa = wait_analysis
            f.write(f"Total wait actions: {wa['wait_total']}\n")
            f.write(f"  - Wait was only option: {wa['wait_only_option']}\n")
            f.write(f"  - Wait despite alternatives: {wa['wait_with_alternatives']}\n")
            f.write(f"Consecutive wait streaks: {wa['streaks']}\n")
            f.write(f"  - Max consecutive: {wa['max_streak']}\n")
            f.write(f"  - Avg streak length: {wa['avg_streak']:.1f}\n")
            if wa['wait_alternative_types']:
                f.write(f"Alternative action types available when wait was chosen:\n")
                for atype, cnt in wa['wait_alternative_types'].most_common():
                    f.write(f"  - {atype}: {cnt} times\n")
            f.write("\n")
        
        for prev, total_count in sorted_actions:
            f.write("=" * 80 + "\n")
            f.write(f"{prev}\n")
            f.write("-" * 80 + "\n")
            
            for i, (next_, count) in enumerate(grouped[prev], 1):
                within_pct = 100.0 * count / total_count
                f.write(f"{i}. [{prev} -> {next_}]: {count} ({within_pct:.1f}%)\n")
            f.write("\n")
        
        # Real failure bigrams section
        if real_failure_bigrams:
            rf_grouped = defaultdict(list)
            for (prev, next_), count in real_failure_bigrams.items():
                rf_grouped[prev].append((next_, count))
            for prev in rf_grouped:
                rf_grouped[prev].sort(key=lambda x: x[1], reverse=True)
            
            f.write("=" * 80 + "\n")
            f.write("REAL FAILURE BIGRAMS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total: {sum(real_failure_bigrams.values())} bigrams\n\n")
            
            for prev in sorted(rf_grouped.keys()):
                total_count = sum(c for _, c in rf_grouped[prev])
                f.write(f"  {prev} FAILED\n")
                for i, (next_, count) in enumerate(rf_grouped[prev], 1):
                    within_pct = 100.0 * count / total_count
                    f.write(f"    {i}. [{prev} FAILED -> {next_}]: {count} ({within_pct:.1f}%)\n")
                f.write("\n")


def bigrams_to_serializable(bigrams: Counter) -> dict:
    """Convert bigram Counter to JSON-serializable grouped dict."""
    grouped = defaultdict(list)
    for (prev, next_), count in bigrams.items():
        grouped[prev].append([next_, count])
    for prev in grouped:
        grouped[prev].sort(key=lambda x: x[1], reverse=True)
    return dict(grouped)


def write_cache_json(selection_rates: dict,
                     success_bigrams: Counter, real_failure_bigrams: Counter,
                     output_path: str):
    """Write structured JSON for action_cache.py to load."""
    data = {
        "selection_rates": {},
        "bigrams": {
            "success": bigrams_to_serializable(success_bigrams),
            "real_failure": bigrams_to_serializable(real_failure_bigrams),
        }
    }
    
    for action_type, stats in selection_rates.items():
        data["selection_rates"][action_type] = {
            "chosen": stats["chosen"],
            "total": stats["total"],
            "rate": round(stats["rate"], 6)
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# ── Main ────────────────────────────────────────────────────────────────────

def process_task(task: str, trace_files: list, output_dir: Path, trace_pairs=None):
    """Process a single task (cook or game).
    
    Args:
        trace_pairs: Optional list of (agent0_path, agent1_path) tuples for
                     cross-referencing opp_next_needed from partner traces (cook only).
    """
    # Load with cross-referencing if pairs provided
    grouped_by_file = None  # For real_failure bigrams (preserves per-file sequences)
    if trace_pairs:
        all_pair_steps = []
        for a0_path, a1_path in trace_pairs:
            a0_steps = []
            with open(a0_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        a0_steps.append(json.loads(line))
            a1_steps = []
            with open(a1_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        a1_steps.append(json.loads(line))
            all_pair_steps.append((a0_steps, a1_steps))
        cross_reference_opp_next(all_pair_steps)
        steps = [s for a0, a1 in all_pair_steps for s in a0 + a1]
        grouped_by_file = [file_steps for a0, a1 in all_pair_steps for file_steps in (a0, a1)]
    else:
        steps = load_trace_files(trace_files)
    print(f"\n{'='*60}")
    print(f"{task.upper()}: loaded {len(steps)} steps from {len(trace_files)} files")
    
    # Selection rates
    rates = compute_selection_rates(steps, task)
    write_selection_report(rates, output_dir / f"{task}_action_selection_report.txt", task)
    
    print(f"  Selection rates:")
    for action_type, s in sorted(rates.items(), key=lambda x: x[1]["rate"], reverse=True):
        print(f"    {action_type}: {s['chosen']}/{s['total']} ({s['rate']*100:.1f}%)")
    
    # Wait pattern analysis
    wait_analysis = analyze_wait_patterns(steps, task)
    print(f"  Wait patterns: {wait_analysis['wait_total']} total, "
          f"max streak={wait_analysis['max_streak']}, "
          f"with alternatives={wait_analysis['wait_with_alternatives']}, "
          f"only option={wait_analysis['wait_only_option']}")
    
    # Success bigrams
    success = compute_bigrams(steps, task, lambda s: s.get("prev_action_success", True))
    print(f"  Success bigrams: {sum(success.values())} total")
    
    # Real failure bigrams: real_failed_action -> next_selected
    if grouped_by_file is None:
        grouped_by_file = load_trace_files_grouped(trace_files)
    real_failure = compute_real_failure_bigrams(grouped_by_file, task)
    print(f"  Real failure bigrams: {sum(real_failure.values())} total")
    for (ft, nt), cnt in real_failure.most_common():
        print(f"    {ft} FAILED -> {nt}: {cnt}")
    
    # Write ngram report (success + real failure bigrams)
    write_ngram_report(success, output_dir / f"{task}_ngram_analysis.txt", task,
                       wait_analysis=wait_analysis, real_failure_bigrams=real_failure)
    
    # Write JSON cache
    write_cache_json(rates, success, real_failure,
                     output_dir / f"{task}_cache_data.json")
    
    print(f"  Written: {task}_action_selection_report.txt, {task}_ngram_analysis.txt, {task}_cache_data.json")


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "agents"
    base = script_dir.parent / "results" / "baseline"
    
    # Cook: episodes 0 and 1 (paired for cross-referencing opp_next)
    cook_pairs = [
        (base / "cook/gpt5/0/0_trace.jsonl", base / "cook/gpt5/0/1_trace.jsonl"),
        (base / "cook/gpt5/1/0_trace.jsonl", base / "cook/gpt5/1/1_trace.jsonl"),
    ]
    cook_files = [str(f) for pair in cook_pairs for f in pair]
    process_task("cook", cook_files, output_dir, trace_pairs=cook_pairs)
    
    # Game: episode 0
    game_files = [
        base / "game/gpt5/0/0_trace.jsonl",
        base / "game/gpt5/0/1_trace.jsonl",
        base / "game/gpt5/0/2_trace.jsonl",
        base / "game/gpt5/0/3_trace.jsonl",
    ]
    process_task("game", [str(f) for f in game_files], output_dir)
    
    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()
