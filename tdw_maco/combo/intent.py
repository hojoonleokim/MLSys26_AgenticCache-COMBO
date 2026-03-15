import json
import os

import numpy as np
import requests
import re
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI, OpenAI
import base64



from envs.game_controller import get_belief_prompt as GameBeliefPrompt
from envs.cook_controller import get_belief_prompt as CookBeliefPrompt

# Function to encode the image
def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class IntentTracker:
    """
    a Vision Language Model that takes in the observation history and infer other agents' intents
    """
    SERVER_ADDR = "http://localhost:8083"
    def __init__(
        self,
        task: str,
        max_tokens: int = 512,
        debug_mode: bool = False,
        temperature: float = 0,
        top_p: float = 0.9,
        lm_id: str = "gpt-4",
        lm_source: str = "openai",
        cot: bool = False
    ):
        self.recipe = None
        self.task = task
        if task == "cook":
            self.get_belief_prompt = CookBeliefPrompt
        else:
            self.get_belief_prompt = GameBeliefPrompt
        self.agents_name = None
        self.name2id = None
        self.debug_mode = debug_mode

        self.lm_id = lm_id
        self.lm_source = lm_source
        self.lm_base = None
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = 3
        self.cot = cot

        if self.lm_source == "openai":
            self.client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
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
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    )
                with open(f"combo/chat_raw.jsonl", 'a') as f:
                    f.write(response.model_dump_json(indent=4))
                    f.write('\n')
                usage = dict(response.usage)
                response = response.choices[0].message.content
                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                return response

            def default_generate(prompt, img):
                return ["wait"]

            def _generate(prompt, img):
                if source == 'openai':
                    return openai_generate(prompt, img)
                elif source == 'azure':
                    raise ValueError("azure is not supported!")
                elif source == 'huggingface':
                    return default_generate(prompt, img)
                    # raise ValueError("huggingface is not supported!")
                else:
                    raise ValueError("invalid source")

            return _generate

        self.generator = lm_engine(self.lm_source, self.lm_id)


    def reset(self, recipe, agents_name):
        self.recipe = recipe
        self.agents_name = agents_name
        self.name2id = {name: i for i, name in enumerate(self.agents_name)}
        
    def run(self, obs_history, agent_id, save_path):
        """
        obs_history: [obs_{i-2}, obs_{i-1}, obs_i], img_path
        """
        # print("obs_history:", obs_history)
        if obs_history[0] is None:
            responses = {"0": {"prompt": "wait"}, "1": {"prompt": "wait"}, "2": {"prompt": "wait"}, "3": {"prompt": "wait"}}
            responses.pop(str(agent_id))
            return responses

        prompt = f"{IMAGE_PLACEHOLDER}\n{IMAGE_PLACEHOLDER}\n{IMAGE_PLACEHOLDER}\n{self.get_belief_prompt(agent_id, self.recipe, self.agents_name)}"
        print("belief prompt:", prompt)
        # img = ','.join(obs_history) if type(obs_history[0]) == str else obs_history
        img = obs_history.copy()
        generated_text = self.generator(prompt, img)
        separated_generated_text = generated_text[0].split('\n')
        actions = {self.name2id[name]: "" for name in self.agents_name}
        for raw_actions in separated_generated_text:
            raw_actions_tokens = raw_actions.split()
            agent_name = raw_actions_tokens[0]
            actions[self.name2id[agent_name]] = ' '.join(raw_actions_tokens[1:])
        responses = {str(i): {"prompt": actions[i]} for i in range(len(self.agents_name))}
        print("generation finished - responses:", responses)
        with open(os.path.join(save_path, f'chat_raw.jsonl'), 'a') as f:
            f.write(json.dumps({"prompt": prompt, "imgs": img,"response": responses}, indent=4))
        # responses = {"0": {"prompt": "wait"}, "1": {"prompt": "wait"}, "2": {"prompt": "wait"}, "3": {"prompt": "wait"}}
        responses.pop(str(agent_id))
        return responses