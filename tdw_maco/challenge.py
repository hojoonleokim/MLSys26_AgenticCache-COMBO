import argparse
import os
import json
from pathlib import Path
import shutil

import gym
import time
import pickle
import logging
import sys
import numpy as np
# Get the absolute path of the project root directory
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

gym.envs.registration.register(
	id='tdw_maco-v0',
	entry_point='envs.tdw_gym:TDW'
)

MAX_DATAPOINT = 10000

from agents import CookPlanAgent, GamePlanAgent, COMBOAgent
from utils.utils import convert_np_for_print


BIN_CORNER_NAMES = ["top left corner", "top right corner", "bottom left corner", "bottom right corner"]


def find_object_bin_corner(obj_name, objects, reachable_bins, threshold=0.1):
	"""Find which bin corner an object is at based on its position. Returns corner name or None."""
	obj_pos = None
	for obj in objects:
		if obj.get("name") == obj_name:
			obj_pos = obj["pos"]
			break
	if obj_pos is None:
		return None
	for i, bin_pos in enumerate(reachable_bins):
		if isinstance(bin_pos, dict):
			bp = np.array([bin_pos["x"], bin_pos["y"], bin_pos["z"]])
		else:
			bp = np.array(bin_pos)
		dist = np.linalg.norm(np.array(obj_pos) - bp)
		if dist < threshold:
			return BIN_CORNER_NAMES[i]
	return None


def get_reverse_prompt(prompt, source_bin_corner=None):
	"""Compute the reverse action prompt for rollback. Returns None if irreversible or no-op.
	
	Handles both cook-style ("pick up the X") and game-style ("pick up X") prompts.
	"""
	if prompt == "wait":
		return None

	# --- pick up → place ---
	if prompt.startswith("pick up"):
		# Determine prefix: cook uses "the ", game uses ""
		if prompt.startswith("pick up the "):
			rest = prompt[len("pick up the "):]
			prefix = "the "
		elif prompt.startswith("pick up "):
			rest = prompt[len("pick up "):]
			prefix = ""
		else:
			return None

		if " from the " not in rest:
			return None
		obj_name, location = rest.split(" from the ", 1)

		# Cook: "private region" (no corner) needs source_bin_corner
		if location == "private region":
			if source_bin_corner:
				return f"place {prefix}{obj_name} onto the {source_bin_corner} of the private region"
			return None

		# All other locations are self-contained (cutting board, game locations, etc.)
		return f"place {prefix}{obj_name} onto the {location}"

	# --- place → pick up ---
	if prompt.startswith("place"):
		if prompt.startswith("place the "):
			rest = prompt[len("place the "):]
			prefix = "the "
		elif prompt.startswith("place "):
			rest = prompt[len("place "):]
			prefix = ""
		else:
			return None

		if " onto the " not in rest:
			return None
		obj_name, location = rest.split(" onto the ", 1)

		# Irreversible targets
		if "plate" in location or "puzzle box" in location:
			return None

		return f"pick up {prefix}{obj_name} from the {location}"

	# --- cut → irreversible ---
	if prompt.startswith("cut"):
		return None

	return None


def get_draft_action_str(agent, step):
	"""Extract the action string from agent's draft_plan at a given step."""
	dp = agent.draft_plan.get(step)
	if dp and len(dp) > 0 and len(dp[0]) > 0:
		return dp[0][0]
	return "wait"


def get_target_action_str(agent, step):
	"""Extract the action string from agent's target_plan at a given step."""
	tp = agent.target_plan.get(step)
	if tp and len(tp) > 0 and len(tp[0]) > 0:
		return tp[0][0]
	return "wait"

class Challenge:
	def __init__(self, logger, task, port, data_path, output_dir, number_of_agents=2, max_steps=30,
				 launch_build=True, screen_size=512, data_prefix='dataset/', save_img=True, save_per_step=8, 
				 skip_success=False, not_skip=False, is_test=False):
		self.env = gym.make("tdw_maco", task=task, port=port, number_of_agents=number_of_agents, save_dir=output_dir,
							max_steps=max_steps, launch_build=launch_build, screen_size=screen_size,
							data_prefix=data_prefix, save_img=save_img, save_per_step=save_per_step, is_test=is_test)
		self.task = task
		self.logger = logger
		self.logger.debug(port)
		self.logger.info("Environment Created")
		self.output_dir = output_dir
		self.max_steps = max_steps
		self.save_img = save_img
		self.skip_success = skip_success
		self.not_skip = not_skip
		self.data = json.load(open(os.path.join(data_prefix, data_path), "r"))
		self.logger.info("done")

	def submit(self, agents, logger, eval_episodes, start_id, num_runs):
		if eval_episodes[0] == -1:
			if start_id is not None and num_runs is not None:
				assert start_id + num_runs <= MAX_DATAPOINT, f"datapoint id exceeds {MAX_DATAPOINT}"
				eval_episodes = range(start_id, start_id + num_runs)
			else:
				eval_episodes = range(len(self.data))

		num_eval_episodes = len(eval_episodes)

		start = time.time()
		results = {}
		for i, episode in enumerate(eval_episodes):
			start_time = time.time()
			if not self.skip_success and not self.not_skip:
				if os.path.exists(os.path.join(self.output_dir, str(episode), 'result_episode.json')):
					# The episode has been evaluated before
					with open(os.path.join(self.output_dir, str(episode), 'result_episode.json'), 'r') as f:
						result = json.load(f)
					results[episode] = result
					continue
			elif self.skip_success:
				# only skips successful trials
				path = os.path.join(self.output_dir, str(episode))
				if os.path.exists(path):
					result_path = os.path.join(path, 'result_episode.json')
					if os.path.exists(result_path):
						with open(result_path, 'r') as f:
							result = json.load(f)
						if result['success']:
							results[episode] = result
							continue

				if os.path.exists(path):
					shutil.rmtree(path)

			elif self.not_skip:
				# don't skip
				path = os.path.join(self.output_dir, str(episode))
				if os.path.exists(path):
					shutil.rmtree(path)

			if not os.path.exists(os.path.join(self.output_dir, str(episode))):
				os.makedirs(os.path.join(self.output_dir, str(episode)))
			self.logger.info('Episode {} ({}/{})'.format(episode, i + 1, num_eval_episodes))
			self.logger.info(f"Resetting Environment ... data is {self.data[episode]}")
			state, info = self.env.reset(seed=self.data[episode]['seed'], options={
				"output_dir": os.path.join(self.output_dir, str(episode)),
				"save_img": self.save_img,
			})
			self.rng = np.random.RandomState(self.data[episode]['seed'])
			recipe = info['recipe'] if self.task == 'cook' else None
			if self.task == 'cook':
				json.dump(recipe, open(os.path.join(self.output_dir, str(episode), 'recipe.json'), 'w'))

			if self.task == 'game':
				direction = {"direction": "clockwise" if self.env.controller.clockwise else "counter_clockwise"}
				json.dump(direction, open(os.path.join(self.output_dir, str(episode), 'direction.json'), 'w'))

			for agent_id, agent in enumerate(agents):
				if agent.agent_type == 'genco_agent':
					obs = self.filter_obs(state[str(agent_id)])
				else:
					obs = state[str(agent_id)]
				agent.reset(obs, info, output_dir=os.path.join(self.output_dir, str(episode)))
			self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")
			done = False
			step_num = 0
			local_reward = 0.0
			metadata = [] # {"step": 0, "actions": "", "frame_start": 0, "frame_end": 13, "prompt": ""}
			camera_matrix_metadata = dict() # dump to pickle
			self.next_agent_id = None
			executed_history = {}  # {step_num: {step_num, agent_actions, rollback?, ...}}
			rollback_steps = 0  # Count of reverse steps (excluded from step budget)
			total_reverse_steps = 0  # Total reverse steps used (for logging)
			last_verified_step = -1  # Last step where target == draft was confirmed
			force_target_next = False  # When rollback impossible, force next step to use target
			loop_start_time = time.time()
			while not done:
				actions_to_print = {}
				# if self.save_img: self.env.save_images(os.path.join(self.output_dir, str(episode), 'Images'))
				print(f"######Step {step_num}######")
				
				# Synchronize draft and target plan speeds for all agents
				# Draft plan should not be more than 3 steps ahead of target plan
				# Only count plans for steps beyond last_verified_step (stale plans are ignored)
				for agent in agents:
					draft_count = sum(1 for k in agent.draft_plan if k > last_verified_step)
					target_count = sum(1 for k in agent.target_plan if k > last_verified_step)
					depth_diff = draft_count - target_count
					
					# If speculative depth is maxed out (3), wait for at least one new target result
					if depth_diff >= 3:
						max_wait = 120  # seconds
						wait_elapsed = 0
						while wait_elapsed < max_wait:
							agent.check_async_vlm_result(pool_type='target')
							new_target_count = sum(1 for k in agent.target_plan if k > last_verified_step)
							if new_target_count > target_count:
								break
							time.sleep(1)
							wait_elapsed += 1
						else:
							print(f"[WARNING] Agent {agent.agent_id}: depth sync timed out after {max_wait}s, proceeding anyway")

				# --- Check for target vs draft disagreement and rollback if needed ---
				# Collect newly arrived target results
				for agent in agents:
					agent.check_async_vlm_result(pool_type='target')
				
				# Find the earliest disagreement step across all agents
				rollback_step = None
				for check_step in range(last_verified_step + 1, step_num):
					# Check if all agents have target results for this step
					all_have_target = all(check_step in agent.target_plan for agent in agents)
					if not all_have_target:
						break  # Can't verify beyond this point
					
					# Compare draft vs target for all agents at this step
					disagree = False
					for agent in agents:
						draft_action = get_draft_action_str(agent, check_step)
						target_action = get_target_action_str(agent, check_step)
						if draft_action != target_action:
							print(f"[ROLLBACK] Disagreement at step {check_step}, Agent {agent.agent_id}: "
								  f"draft='{draft_action}' vs target='{target_action}'")
							disagree = True
							break
					
					if disagree:
						rollback_step = check_step
						break
					else:
						last_verified_step = check_step
				
				if rollback_step is not None:
					print(f"[ROLLBACK] Disagreement at step {rollback_step}, current step {step_num}")
					
					# 0. Pre-check: abort if ANY agent has an irreversible action in rollback range
					rollback_possible = True
					for check_rev in range(rollback_step, step_num):
						entry = executed_history[check_rev]
						for agent in agents:
							aid = str(agent.agent_id)
							agent_entry = entry["agent_actions"].get(aid, {})
							prompt = agent_entry.get("prompt", "wait")
							rejected = agent_entry.get("rejected", False)
							if rejected or prompt == "wait":
								continue
							rev_prompt = get_reverse_prompt(prompt, source_bin_corner=agent_entry.get("source_bin_corner"))
							if rev_prompt is None:
								print(f"[ROLLBACK] ABORT: irreversible action at step {check_rev} Agent {aid}: '{prompt}'")
								rollback_possible = False
								break
						if not rollback_possible:
							break
					
					if not rollback_possible:
						print(f"[ROLLBACK] Skipping rollback (irreversible actions), will use target for next step")
						force_target_next = True
					else:
						# 1. Reverse ALL agent actions from (step_num - 1) down to rollback_step
						#    Each reverse is a real env step with rollback marker in history
						#    If any reverse is rejected, abort rollback and fall through to force_target_next
						rollback_aborted = False
						for rev_step in range(step_num - 1, rollback_step - 1, -1):
							reverse_actions = {}
							entry = executed_history[rev_step]
							has_any_reverse = False
							
							for agent in agents:
								aid = str(agent.agent_id)
								agent_entry = entry["agent_actions"].get(aid, {})
								prompt = agent_entry.get("prompt", "wait")
								rejected = agent_entry.get("rejected", False)
								
								if rejected or prompt == "wait":
									reverse_actions[aid] = {"type": None, "prompt": "wait"}
								else:
									rev_prompt = get_reverse_prompt(prompt, source_bin_corner=agent_entry.get("source_bin_corner"))
									reverse_actions[aid] = {"type": None, "prompt": rev_prompt}
									has_any_reverse = True
									print(f"  [ROLLBACK] Step {rev_step} Agent {aid}: '{prompt}' -> '{rev_prompt}'")
							
							if has_any_reverse:
								print(f"  [ROLLBACK] Executing reverse for step {rev_step} (env step {step_num})")
								state, rev_reward, rev_done, rev_info = self.env.step(reverse_actions)
								# Record in history with rollback marker
								executed_history[step_num] = {
									"step_num": step_num,
									"rollback": True,
									"rollback_type": "reverse",
									"reversing_step": rev_step,
									"agent_actions": {
										str(a.agent_id): {
											"prompt": reverse_actions[str(a.agent_id)]["prompt"],
											"rejected": state[str(a.agent_id)]["rejected"],
										}
										for a in agents
									}
								}
								for agent in agents:
									aid = str(agent.agent_id)
									agent.action_history.append(reverse_actions[aid])
								step_num += 1
								rollback_steps += 1
								total_reverse_steps += 1
								self.env.unwrapped.max_steps += 1  # Extend env limit for rollback step
								
								# Check if any reverse was rejected → abort rollback
								any_rejected = False
								for agent in agents:
									aid = str(agent.agent_id)
									if reverse_actions[aid]["prompt"] != "wait" and state[aid]["rejected"]:
										print(f"  [ROLLBACK] ABORT: Reverse for Agent {aid} rejected at step {rev_step}: {state[aid].get('rejected_reason', 'unknown')}")
										any_rejected = True
								
								if any_rejected:
									print(f"  [ROLLBACK] Aborting rollback, switching to force_target_next")
									rollback_aborted = True
									break
								else:
									for agent in agents:
										aid = str(agent.agent_id)
										rev_prompt = reverse_actions[aid]["prompt"]
										rev_rejected = state[aid]["rejected"]
										agent.update_state_from_action(rev_prompt, rejected=rev_rejected)
										if rev_prompt != "wait" and not rev_rejected:
											print(f"  [ROLLBACK] Reverse for Agent {aid} succeeded")
						
						if rollback_aborted:
							force_target_next = True
							for agent in agents:
								agent.drain_stale_slots(invalidate_from_step=rollback_step)
								agent.steps = step_num
							last_verified_step = step_num - 1
						else:
							# 2. Execute ONLY the target plan at the disagreement step
							target_actions = {}
							for agent in agents:
								aid = str(agent.agent_id)
								target_actions[aid] = {"type": None, "prompt": get_target_action_str(agent, rollback_step)}
							
							print(f"  [ROLLBACK] Executing target plan (env step {step_num}): { {a: target_actions[a]['prompt'] for a in target_actions} }")
							state, target_reward, target_done, target_info = self.env.step(target_actions)
							
							# Record target execution with rollback marker
							executed_history[step_num] = {
								"step_num": step_num,
								"rollback": True,
								"rollback_type": "target_correction",
								"correcting_step": rollback_step,
								"agent_actions": {
									str(a.agent_id): {
										"prompt": target_actions[str(a.agent_id)]["prompt"],
										"rejected": state[str(a.agent_id)]["rejected"],
									}
									for a in agents
								}
							}
							for agent in agents:
								agent.action_history.append(target_actions[str(agent.agent_id)])
							step_num += 1
							self.env.unwrapped.max_steps += 1  # Extend env limit (env.step was called)
							# NOTE: no rollback_steps increment — target correction is a real forward action

							for agent in agents:
								aid = str(agent.agent_id)
								tgt_prompt = target_actions[aid]["prompt"]
								tgt_rejected = state[aid]["rejected"]
								agent.update_state_from_action(tgt_prompt, rejected=tgt_rejected)
								if tgt_rejected:
									print(f"  [ROLLBACK] Target Agent {aid}: REJECTED ({state[aid].get('rejected_reason', '')})")
								else:
									print(f"  [ROLLBACK] Target Agent {aid}: OK")
							
							# 3. Drain stale slots and invalidate stale plans
							for agent in agents:
								agent.drain_stale_slots(invalidate_from_step=rollback_step)
								agent.steps = step_num  # Sync agent step counter
							
							last_verified_step = step_num - 1  # Skip all rollback steps
							
							print(f"[ROLLBACK] Complete. Continuing from step {step_num}")
				# --- End rollback check ---

				# Call act() for all agents to submit draft/target queries
				plan_success, _ = self.plan_agent_actions(agents, state, episode)
				if not plan_success:
					done = True
					break
				# Wait for plans to be ready for current step
				use_target = force_target_next
				if use_target:
					print(f"[FORCE_TARGET] Waiting for target plans at step {step_num}")
				while True:
					pool_type = 'target' if use_target else 'draft'
					for agent in agents:
						agent.check_async_vlm_result(pool_type=pool_type)
					
					plan_dict_attr = 'target_plan' if use_target else 'draft_plan'
					all_ready = all(step_num in getattr(agent, plan_dict_attr) for agent in agents)
					
					if all_ready:
						break
					
					time.sleep(1)  # Short sleep to avoid busy waiting
				
				# Build actions from plans (target if forced, draft otherwise)
				actions = {}
				for agent in agents:
					if use_target:
						plan = agent.target_plan[step_num]
						print(f"Agent {agent.agent_id} TARGET plan for step {step_num}: {plan}")
					else:
						plan = agent.draft_plan[step_num]
						print(f"Agent {agent.agent_id} draft plan for step {step_num}: {plan}")
					
					if plan and len(plan) > 0 and len(plan[0]) > 0:
						action = {"type": None, "prompt": plan[0][0]}
					else:
						action = {"type": None, "prompt": "wait"}
					actions[str(agent.agent_id)] = action
					
					# Add action to agent's action_history
					agent.action_history.append(action)
				
				if use_target:
					last_verified_step = step_num  # Target-executed step is verified
					force_target_next = False
					print(f"[FORCE_TARGET] Executed target plan at step {step_num}")

				# Compute source bin info BEFORE env.step() (object positions change after step)
				# Only needed for cook task's "pick up the X from the private region"
				# where the corner isn't specified. Game locations are self-contained.
				source_infos = {}
				for agent in agents:
					aid = str(agent.agent_id)
					prompt = actions[aid]["prompt"]
					if prompt.startswith("pick up the ") and prompt.endswith("from the private region"):
						# Cook task: "pick up the X from the private region" (no corner specified)
						obj_name = prompt[len("pick up the "):].split(" from ")[0]
						source_infos[aid] = find_object_bin_corner(
							obj_name, state[aid].get("objects", []), agent.reachable_bins
						)
					else:
						source_infos[aid] = None

				frame_start = self.env.num_frames
				last_obs = convert_np_for_print(state["0"]["objects"])
				print(f"Agent actions: {actions}")
				state, reward, done, info = self.env.step(actions)
				
				# Print action results for each agent
				step_results = []
				for agent_id in range(len(agents)):
					if state[str(agent_id)]["rejected"]:
						step_results.append(f"Agent {agent_id}: ❌ FAILED ({state[str(agent_id)]['rejected_reason']})")
					else:
						step_results.append(f"Agent {agent_id}: ✅ SUCCESS")
				print(f"Results: {' | '.join(step_results)}")
				
				# Record executed actions for rollback tracking
				step_entry = {"step_num": step_num, "agent_actions": {}}
				for agent in agents:
					aid = str(agent.agent_id)
					step_entry["agent_actions"][aid] = {
						"prompt": actions[aid]["prompt"],
						"rejected": state[aid]["rejected"],
						"source_bin_corner": source_infos.get(aid),
					}
				executed_history[step_num] = step_entry
				
				metadata.append({"step": step_num, "obs": last_obs, "actions": actions_to_print, "frame_start": frame_start, "frame_end": self.env.num_frames, "prompt": "", "prompt_value": info["prompt_value"]})
				
				# for agent_id, agent in enumerate(agents):
				# 	print("end_pos: ", agent_id, self.env.controller.agents[agent_id].dynamic.transform.position)
				for agent in agents:
					agent_id = agent.agent_id
					if state[str(agent_id)]["rejected"]:
						final_action = {"type": "wait", "prompt": "wait"}
						if "prompt_proposer" in actions[str(agent_id)]:
							prompt_proposer = actions[str(agent_id)]["prompt_proposer"].split("I choose to")[0]
							final_action["prompt_proposer"] = prompt_proposer + "I choose to wait."

					else:
						final_action = actions[str(agent_id)]

					actions_to_print[str(agent_id)] = convert_np_for_print(final_action)

				metadata[-1]["prompt"] = state["0"]["last_joint_actions"]
				# print(f"metadata: {metadata[-1]['prompt']}\n{metadata[-1]['prompt_value']}")
				# metadata[-1]["frame_end"] += info["num_frames_for_step"]

				if self.save_img:
					for k, v in info["camera_matrices"].items():
						camera_matrix_metadata[k] = v

				local_reward += reward
				self.logger.info(
					f"Executing step {step_num} for episode: {episode}, actions: {actions}, frame: {self.env.num_frames}")
				step_num += 1
				if done or (step_num - rollback_steps) > self.max_steps:
					break

			loop_elapsed_time = time.time() - loop_start_time
			real_steps = step_num - rollback_steps
			print(f"[EPISODE {episode}] Total env steps: {step_num}, real steps: {real_steps}, reverse steps: {total_reverse_steps}, rollback overhead: {rollback_steps}")
			
			# Cleanup VLM pools for all agents
			for agent in agents:
				if hasattr(agent, 'cleanup'):
					agent.cleanup()
			
			if 'success' in info:
				result = {
					"success": info['success'],
					"steps": step_num,
					"rollback_steps": rollback_steps,
					"effective_steps": step_num - rollback_steps,
					"time": loop_elapsed_time,
				}
			else:
				result = {
					"success": False,
					"steps": step_num,
					"rollback_steps": rollback_steps,
					"effective_steps": step_num - rollback_steps,
					"time": loop_elapsed_time,
				}

			with open(os.path.join(self.output_dir, str(episode), 'result_episode.json'), 'w') as f:
				json.dump(result, f)
			# print(f"metadata: {metadata}")
			with open(os.path.join(self.output_dir, str(episode), 'metadata.json'), 'w') as f:
				json.dump(metadata, f, indent=4)

			# Save execution history with rollback info (append to single file)
			with open(os.path.join(self.output_dir, 'action_history_all.txt'), 'a') as f:
				f.write(f'\n{"#" * 50}\n')
				f.write(f'Episode {episode} | Steps: {step_num} | Success: {info.get("success", False)}\n')
				f.write(f'{"#" * 50}\n\n')
				
				for s in sorted(executed_history.keys()):
					entry = executed_history[s]
					is_rollback = entry.get("rollback", False)
					rollback_type = entry.get("rollback_type", "")
					
					# Step header
					if is_rollback:
						if rollback_type == "reverse":
							f.write(f'Step {s} [ROLLBACK: reversing step {entry.get("reversing_step")}]\n')
						elif rollback_type == "target_correction":
							f.write(f'Step {s} [ROLLBACK: target correction for step {entry.get("correcting_step")}]\n')
						else:
							f.write(f'Step {s} [ROLLBACK]\n')
					else:
						f.write(f'Step {s}\n')
					
					# Per-agent actions
					for aid in sorted(entry["agent_actions"].keys()):
						a_entry = entry["agent_actions"][aid]
						prompt = a_entry.get("prompt", "wait")
						rejected = a_entry.get("rejected", False)
						status = "FAIL" if rejected else "OK"
						f.write(f'  Agent {aid}: {prompt} [{status}]\n')
					f.write('\n')

			# print("camera_matrix_metadata: ", camera_matrix_metadata)
			if self.save_img:
				with open(os.path.join(self.output_dir, str(episode), 'camera_matrix_metadata.pickle'), 'wb') as f:
					pickle.dump(camera_matrix_metadata, f)

			results[episode] = result
		avg_succ = np.mean([results[episode]['success'] for episode in results])
		avg_succ_steps = np.mean([results[episode]['steps'] for episode in results if results[episode]['success']])
		results = {
			"avg_succ": avg_succ,
			"avg_succ_steps": avg_succ_steps,
			"episode_results": results,
		}
		if num_eval_episodes > 1:
			with open(os.path.join(self.output_dir, 'eval_result.json'), 'w') as f:
				json.dump(results, f, indent=4)
		self.logger.info(f'eval done, avg success rate {avg_succ}, avg success steps {avg_succ_steps}')
		self.logger.info('time: {}'.format(time.time() - start))
		return avg_succ, avg_succ_steps
	
	def plan_agent_actions(self, agents, state, episode):
		actions = {}
		for agent in agents:
			agent_id = agent.agent_id
			# if agent.agent_type == 'combo_agent':
			# 	obs = self.filter_obs(state[str(agent_id)])
			# else:
			obs = state[str(agent_id)] # How to form state
			action = agent.act(obs,episode)
			# print(agent_id, action)
			actions[str(agent_id)] = action
		return True, actions

	def close(self):
		self.env.close()

	def filter_obs(self, obs:dict):
		# filter out oracle observations
		filtered_obs = obs.copy()
		filtered_obs.pop('objects')
		return filtered_obs


def init_logs(output_dir, port, name='simple_example'):
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)
	fh = logging.FileHandler(os.path.join(output_dir, f"output_{port}.log"))
	fh.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	logger.addHandler(fh)
	logger.addHandler(ch)
	return logger


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", type=str, choices=("cook", "game", "game_3", "game_2"), default="cook")
	parser.add_argument("--output_dir", type=str, default="results")
	parser.add_argument("--experiment_name", type=str, default="train")
	parser.add_argument("--run_id", type=str, default='run_0')
	parser.add_argument("--data_path", type=str, default="train.json")
	parser.add_argument("--data_prefix", type=str, default="dataset/")
	parser.add_argument("--port", default=1071, type=int)
	parser.add_argument("--start_id", type=int, default=None)
	parser.add_argument("--num_runs", type=int, default=None)
	parser.add_argument("--agents_type", nargs='+', type=str, default=("replicant",))
	parser.add_argument("--agents_algo", nargs='+', type=str, default=("cook_plan_agent", "cook_plan_agent"))
	parser.add_argument("--eval_episodes", nargs='+', default=(-1,), type=int, help="which episodes to evaluate on")
	parser.add_argument("--max_steps", default=30, type=int, help="max steps per episode")
	parser.add_argument("--no_launch_build", action='store_true')
	parser.add_argument("--debug", action='store_true')
	parser.add_argument("--screen_size", default=512, type=int)
	parser.add_argument("--no_save_img", action='store_true', help="do not save images", default=False)
	parser.add_argument("--save_per_step", default=8, type=int, help="save images every n frames")
	parser.add_argument("--skip_success", action='store_true', help="only skip successful trials", default=False)
	parser.add_argument("--not_skip", action='store_true', help="don't skip episodes", default=False)
	parser.add_argument("--metadata_file", type=str, default="results/game/0/metadata.json" )

	# combo parameters
	parser.add_argument("--only_propose", action="store_true")
	parser.add_argument("--no_belief", action="store_true")
	parser.add_argument("--num_propose", type=int, default=1)
	parser.add_argument("--plan_horizon", type=int, default=1)
	parser.add_argument("--plan_beam", type=int, default=1)

	# LLM parameters
	parser.add_argument("--proposer_lm_id", "-pllm", type=str, default="gpt-4-vision-preview")
	parser.add_argument("--belief_lm_id", "-bllm", type=str, default="gpt-4-vision-preview")
	parser.add_argument("--ranker_lm_id", "-rllm", type=str, default="gpt-4-vision-preview")
	parser.add_argument("--lm_source", type=str, choices=["openai", "llava", "azure", "huggingface", "llava_server"], default="openai")
	parser.add_argument("--temperature", "-t", type=float, default=0.7)
	parser.add_argument("--top_p", default=1.0, type=float)
	parser.add_argument("--max_tokens", default=512, type=int)
	parser.add_argument("--n", default=1, type=int)
	parser.add_argument("--logprobs", default=1, type=int)
	parser.add_argument("--echo", action='store_true', help="to include prompt in the outputs")
	parser.add_argument("--cot", action="store_true")

	parser.add_argument("--guidance_weight", "-gw", type=int, default=5)

	args = parser.parse_args()
	if args.lm_source == 'llava' and 'llava' not in args.proposer_lm_id:
		args.proposer_lm_id = "liuhaotian/llava-v1.5-7b"
		args.belief_lm_id = "liuhaotian/llava-v1.5-7b"
		args.ranker_lm_id = "liuhaotian/llava-v1.5-7b"
	args.number_of_agents = len(args.agents_algo)
	os.makedirs(args.output_dir, exist_ok=True)
	args.output_dir = os.path.join(args.output_dir, args.experiment_name)
	os.makedirs(args.output_dir, exist_ok=True)
	args.output_dir = os.path.join(args.output_dir, args.task)
	os.makedirs(args.output_dir, exist_ok=True)
	args.output_dir = os.path.join(args.output_dir, args.run_id)
	os.makedirs(args.output_dir, exist_ok=True)
	logger = init_logs(args.output_dir, args.port)

	challenge = Challenge(logger, args.task, args.port, args.data_path, args.output_dir, args.number_of_agents,
						  args.max_steps, not args.no_launch_build, screen_size=args.screen_size,
						  data_prefix=args.data_prefix, save_img=not args.no_save_img, save_per_step=args.save_per_step, 
						  skip_success = args.skip_success, not_skip = args.not_skip,
						  is_test=(args.data_path == "test.json"))
	agents = []
	print(args)
	for i, agent in enumerate(args.agents_algo):
		if agent == 'cook_plan_agent':
			agents.append(CookPlanAgent(i, logger, args.output_dir, is_altruism=False))
		elif agent == 'cook_plan_agent_altruism':
			agents.append(CookPlanAgent(i, logger, args.output_dir, is_altruism=True))
		elif agent == 'cook_plan_agent_selfish':
			agents.append(CookPlanAgent(i, logger, args.output_dir, is_altruism=False))
		elif agent == 'game_plan_agent_clockwise': # agent with fixed clockwise passing direction
			agents.append(GamePlanAgent(i, logger, args.output_dir, False, fix_clockwise=True))
		elif agent == 'game_plan_agent_counter_clockwise': # agent with fixed counter-clockwise passing direction
			agents.append(GamePlanAgent(i, logger, args.output_dir, False, fix_clockwise=False))
		elif agent in ['combo_agent', 'genco_agent']:
			agents.append(COMBOAgent(
				task=args.task,
				agent_id=i,
				logger=logger,
				output_dir=args.output_dir,
				max_tokens=args.max_tokens,
				debug_mode=args.debug,
				num_propose=args.num_propose,
				temperature=args.temperature,
				proposer_lm_id=args.proposer_lm_id,
				belief_lm_id=args.belief_lm_id,
				ranker_lm_id=args.ranker_lm_id,
				lm_source=args.lm_source,
				only_propose=args.only_propose,
				no_belief=args.no_belief,
				plan_horizon=args.plan_horizon,
				plan_beam=args.plan_beam,
				cot=args.cot,
				guidance_weight=args.guidance_weight,
				run_id=args.run_id,
				experiment_name=args.experiment_name,
			))
		else:
			pass
	try:
		challenge.submit(agents, logger, args.eval_episodes, args.start_id, args.num_runs)
	finally:
		# Clean up agents before closing challenge
		for agent in agents:
			if hasattr(agent, 'cleanup'):
				try:
					agent.cleanup()
				except Exception as e:
					print(f"Warning: Error cleaning up agent: {e}")
		challenge.close()


if __name__ == "__main__":
	main()