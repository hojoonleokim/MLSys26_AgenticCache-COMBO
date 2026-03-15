import json
import os

import numpy as np
import requests
import re
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI, OpenAI
import base64


from envs.game_controller import get_proposal_prompt as GameProposerPrompt, get_future_proposal_prompt as GameFutureProposerPrompt
from envs.cook_controller import get_proposal_prompt as CookProposerPrompt, get_future_proposal_prompt as CookFutureProposerPrompt
# Function to encode the image
IMAGE_PLACEHOLDER = "<image>"
api_key = os.environ['OPENAI_API_KEY']

def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class Proposer:
    """
    a Vision Language Model that takes in the current observation and propose an action for a given agent
    """
    SERVER_ADDR = "http://localhost:8081"
    def __init__(
        self,
        task: str,
        max_tokens: int = 512,
        debug_mode: bool = False,
        num_propose: int = 2,
        temperature: float = 1,
        top_p: float = 0.9,
        lm_id: str = "gpt-4",
        lm_source: str = "openai",
        cot: bool = False,
        agent_id: int = 0,
        run_id: str = "baseline",
        experiment_name: str = "train",
        output_dir: str = None
    ):
        self.agents_name = None
        self.task = task
        if task == "cook":
            self.get_proposer_prompt = CookProposerPrompt
            self.get_future_proposer_prompt = CookFutureProposerPrompt
        else:
            self.get_proposer_prompt = GameProposerPrompt
            self.get_future_proposer_prompt = GameFutureProposerPrompt
        self.recipe = None
        self.debug_mode = debug_mode
        self.num_propose = num_propose

        self.lm_id = lm_id
        self.lm_source = lm_source
        self.lm_base = None
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = 3
        self.cot = cot

        if output_dir is not None:
            chat_base = os.path.join(output_dir, task, self.lm_id)
        else:
            chat_base = f"chat_log/{experiment_name}/{run_id}/{task}/{self.lm_id}"
        self.record_dir = f"{chat_base}/{agent_id}_chat.jsonl"
        os.makedirs(chat_base, exist_ok=True)
        self.token_usage_dir = f"{chat_base}/{agent_id}_token_usage.jsonl"
        if self.lm_source == "openai":
            self.client = OpenAI(
                api_key=api_key,  # this is also the default, it can be omitted
                max_retries=self.max_retries,
                )
        elif self.lm_source == "azure":
            pass
        elif self.lm_source == "huggingface":
            # self.client = AutoModelForCausalLM.from_pretrained(self.lm_id)
            pass

        else:
            raise NotImplementedError(f"{self.lm_source} is not supported!")


        def lm_engine(source, lm_id):
            def openai_generate(prompt, img):
                if type(img) is list:
                    img = img[0]
                if type(prompt) is list:
                    prompt = prompt[0]
                
                try:
                    response = self.client.chat.completions.create(
                        model=lm_id,
                        messages=[
                            # {"role": "user", "content": ""},
                            {"role": "user",
                             "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"},
                                    # "detail": "low"
                                },
                             ],
                             },
                        ],
                        # max_tokens=self.max_tokens,
                        # temperature=self.temperature,
                        )
                    # with open(f"combo/chat_raw.jsonl", 'a') as f:
                    #     f.write(response.model_dump_json(indent=4))
                    #     f.write('\n')
                    usage = response.usage
                    response = response.choices[0].message.content
                    self.write_token_usage_to_file(f"Input tokens: {usage.prompt_tokens}")
                    self.write_token_usage_to_file(f"Output tokens: {usage.completion_tokens}")
                    self.write_token_usage_to_file(f"Total tokens: {usage.total_tokens}")
                    # print('======= response ======= \n ', response)
                    # print('======= usage ======= \n ', usage)
                    return response
                    
                except Exception as e:
                    self.write_log_to_file(f"\n{'='*80}\nERROR at OpenAI API call:\n{type(e).__name__}: {str(e)}\n{'='*80}\n")
                    
                    # 에러 발생 시 wait 반환
                    print("⚠️  Returning 'wait' action due to error\n")
                    return "wait"


            def default_generate(prompt, img):
                return ["wait"]

            def _generate(prompt, img):
                if source == 'openai':
                    return openai_generate(prompt, img)
                elif source == 'azure':
                    raise ValueError("azure is not supported!")
                elif source == 'huggingface':
                    return default_generate(prompt, img)
                else:
                    raise ValueError("invalid source")

            return _generate

        self.generator = lm_engine(self.lm_source, self.lm_id)


    def reset(self, recipe, agents_name):
        self.recipe = recipe
        self.agents_name = agents_name
    
    def parse_action_from_response(self, response, available_plans):
        """Parse action from LLM response
        
        Args:
            response: Raw response string from LLM
            available_plans: List of available action plans
            
        Returns:
            Parsed action string
        """
        next_action = None
        try:
            # Find the line that contains "next action"
            lines = response.split('\n')
            action_keywords = ['pick up', 'place', 'cut', 'wait']
            for i, line in enumerate(lines):
                if 'next action' in line.lower():
                    # Get the action from the same line or next line
                    if ':' in line:
                        # Check if action is on the same line after ':'
                        action_text = line.split(':', 1)[1].strip()
                        
                        # Check if it's a letter choice (A, B, C, etc.)
                        if action_text and len(action_text) == 1 and action_text.upper().isalpha():
                            choice_idx = ord(action_text.upper()) - ord('A')
                            if 0 <= choice_idx < len(available_plans):
                                next_action = available_plans[choice_idx]
                                break
                        
                        # Check if it starts with letter choice format "A. action"
                        if action_text and len(action_text) >= 3 and action_text[0].upper().isalpha() and action_text[1] == '.':
                            choice_idx = ord(action_text[0].upper()) - ord('A')
                            if 0 <= choice_idx < len(available_plans):
                                next_action = available_plans[choice_idx]
                                break
                        
                        if action_text and action_text != '-' and any(kw in action_text.lower() for kw in action_keywords):
                            # Format: "Next action: pick up the burger_bottom"
                            next_action = action_text
                            break
                    
                    # If not found on same line, search subsequent lines
                    if not next_action:
                        # Search up to 10 lines after "next action" header
                        for j in range(i + 1, min(i + 11, len(lines))):
                            next_line = lines[j].strip()
                            if not next_line:  # Skip empty lines
                                continue
                            
                            # Check if it's a letter choice
                            if len(next_line) == 1 and next_line.upper().isalpha():
                                choice_idx = ord(next_line.upper()) - ord('A')
                                if 0 <= choice_idx < len(available_plans):
                                    next_action = available_plans[choice_idx]
                                    break
                            
                            # Check if it starts with letter choice format "A. action"
                            if len(next_line) >= 3 and next_line[0].upper().isalpha() and next_line[1] == '.':
                                choice_idx = ord(next_line[0].upper()) - ord('A')
                                if 0 <= choice_idx < len(available_plans):
                                    next_action = available_plans[choice_idx]
                                    break
                            
                            # Remove leading dashes
                            if next_line.startswith('--'):
                                next_line = next_line[2:].strip()
                            elif next_line.startswith('-'):
                                next_line = next_line[1:].strip()
                            
                            # Check if this line contains a valid action keyword
                            if any(next_line.lower().startswith(kw) for kw in action_keywords):
                                next_action = next_line
                                break
                    break
            
            # Clean up the action
            if next_action:
                if next_action.endswith('.'):
                    next_action = next_action[:-1]
                next_action = next_action.lower().strip()
            
        except Exception as e:
            print(f"###### Error parsing next action: {e} ######")
        
        # Default to wait if no action found
        if not next_action:
            next_action = "wait"
            print(f"No next action found, defaulting to wait")
        
        return next_action

    def run(self, img_path, agent_id, save_path, progress, available_plans, steps, action_history, episode, use_future_prompt=False):
        # Format action_history - take last 10 non-wait actions
        non_wait_actions = []
        for i, action in enumerate(action_history):
            if 'wait' not in action['prompt'].lower():
                non_wait_actions.append((i, action))
        
        last_n_actions = non_wait_actions[-10:] if len(non_wait_actions) > 10 else non_wait_actions
        history_parts = []
        for step_num, action in last_n_actions:
            history_parts.append(f"{action['prompt']} at step {step_num}")
        history_text = ", ".join(history_parts)
        
        # Get base prompt based on use_future_prompt flag
        if use_future_prompt:
            # For future prediction, don't need available_plans list
            base_prompt = self.get_future_proposer_prompt(agent_id, self.recipe, self.agents_name, steps)
            base_prompt = base_prompt.replace("#ACTION_HISTORY#", history_text.strip())
        else:
            # Format available_plans as A, B, C list
            plans = ""
            for i, plan in enumerate(available_plans):
                plans += f"{chr(ord('A') + i)}. {plan}\n"
            
            base_prompt = self.get_proposer_prompt(agent_id, self.recipe, self.agents_name, steps)
            base_prompt = base_prompt.replace("#POSSIBLE_ACTIONS#", plans.strip())
            base_prompt = base_prompt.replace("#ACTION_HISTORY#", history_text.strip())
        self.write_log_to_file(f"Base Prompt Episode {episode} #{steps}: \n{base_prompt}")
        prompt = f"{IMAGE_PLACEHOLDER}\n{base_prompt}"
        prompt = [prompt for _ in range(len(img_path) if type(img_path) is list else 1)]
        if self.debug_mode:
            prompt = input(f"Enter prompt for {img_path}:\n")
            self.generator(prompt, img_path)
        
        all_responses = self.generator(prompt, img_path) # b,
        self.write_log_to_file(f"Responses Episode {episode} #{steps}: \n{all_responses}")
        all_proposes = []
        # Wrap single response in a list for iteration
        responses_list = [all_responses] if isinstance(all_responses, str) else all_responses
        for responses in responses_list:
            # Use the extracted parsing method
            next_action = self.parse_action_from_response(responses, available_plans)
            all_proposes.append({'action': [next_action], 'raw_response': responses})
        return all_proposes

    def write_log_to_file(self,log_message, file_name=None):
        file_name = self.record_dir
        with open(file_name, 'a') as file:  
            file.write(log_message + '\n')  

    def write_token_usage_to_file(self,log_message, file_name=None):
        file_name = self.token_usage_dir
        with open(file_name, 'a') as file:  
            file.write(log_message + '\n')  