import json
import os

import numpy as np
import requests
import re
from PIL import Image
from io import BytesIO
from openai import AzureOpenAI, OpenAI
import base64
import cv2


from envs.game_controller import get_value_prompt as GameValuePrompt
from envs.cook_controller import get_value_prompt as CookValuePrompt

# Function to encode the image
def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class Ranker:
    """
    a Vision Language Model that evaluate an image outcome with a numerical score
    """
    SERVER_ADDR = "http://localhost:8082"
    def __init__(
        self,
        task: str,
        max_tokens: int = 512,
        debug_mode: bool = False,
        temperature: float = 0,
        top_p: float = 0.9,
        lm_id: str = "gpt-4",
        lm_source: str = "openai",
        cot: bool = False,
    ):
        self.progress = None
        self.agents_name = None
        self.recipe = None
        self.task = task
        if task == "cook":
            self.get_value_prompt = CookValuePrompt
        else:
            self.get_value_prompt = GameValuePrompt
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
                else:
                    raise ValueError("invalid source")

            return _generate

        self.generator = lm_engine(self.lm_source, self.lm_id)

    def reset(self, recipe, agents_name):
        self.recipe = recipe
        self.agents_name = agents_name
        self.progress = [0 for _ in range(len(agents_name))]

    def run(self, img_path, save_path, previous_scores_dicts):
        # input: img_path
        # output: score [number of the objects to operate, e.g. 5]
        prompt = f"{IMAGE_PLACEHOLDER}\n" + self.get_value_prompt(0, self.recipe, self.agents_name)
        prompt = [prompt for _ in range(len(img_path) if type(img_path) is list else 1)]
        responses = self.generator(prompt, img_path)
        with open(os.path.join(save_path, f'chat_raw.jsonl'), 'a') as f:
            f.write(json.dumps({"prompt": prompt, "img": img_path, "response": responses}, indent=4))
        scores = []
        scores_dict = []
        for i, response in enumerate(responses):
            score_dict = {}
            items = [[],[]]
            score_total = 0
            for line in response.split('\n'):
                item = None
                try:
                    item = line.split(':')[0]
                    score = int(re.search(r'needs (\d+) more step', line).group(1))
                    if previous_scores_dicts[i] is not None and abs(score - previous_scores_dicts[i][item]) > 1:
                        print(f"Ranker score {score} from {line} {img_path[i]} is abnormal compared to previous {previous_scores_dicts[i][item]}, use default")
                        score = previous_scores_dicts[i][item]
                    score_dict[item] = score
                    score_total += score
                    if self.task == "cook":
                        if score == 0:
                            if item in self.recipe[0]:
                                self.progress[0] = max(self.progress[0], self.recipe[0].index(item) + 1)
                            elif item in self.recipe[1]:
                                self.progress[1] = max(self.progress[1], self.recipe[1].index(item) + 1)
                        elif "on the cutting_board" in line:
                            items[0].append(item)
                            items[1].append(item)
                        elif "in the correct agent's" in line:
                            if item in self.recipe[0]:
                                items[0].append(item)
                            elif item in self.recipe[1]:
                                items[1].append(item)
                        elif "in the other agent's" in line:
                            if item in self.recipe[0]:
                                items[1].append(item)
                            elif item in self.recipe[1]:
                                items[0].append(item)
                except Exception as e:
                    print(f"Ranker score not found from {line}, use default, {e}")
                    if previous_scores_dicts[i] is not None and item and item in previous_scores_dicts[i]:
                        score = previous_scores_dicts[i][item]
                    else:
                        print(f"even item is not found, use 10 as default")
                        score = 10
                    score_total += score
                    # score = previous_scores_dicts[i][item]
                if self.task == "cook":
                    to_put = [None, None]
                    if self.progress[0] < len(self.recipe[0]):
                        to_put[0] = self.recipe[0][self.progress[0]]
                    if self.progress[1] < len(self.recipe[1]):
                        to_put[1] = self.recipe[1][self.progress[1]]
                    if len(items[0]) == 5 and to_put[0] not in items[0]:
                        with open(os.path.join(save_path, f'chat_raw.jsonl'), 'a') as f:
                            f.write(json.dumps({"img": img_path, "stuck cutting_board": True}, indent=4))
                        score_total += 4
                    if len(items[1]) == 5 and to_put[1] not in items[1]:
                        with open(os.path.join(save_path, f'chat_raw.jsonl'), 'a') as f:
                            f.write(json.dumps({"img": img_path, "stuck cutting_board": True}, indent=4))
                        score_total += 4
            scores_dict.append(score_dict)
            scores.append(score_total)
        return scores, scores_dict