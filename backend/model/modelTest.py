import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import urllib.request
import random
from datetime import datetime
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import LTokenizer
from transformers import pipeline, AutoModel, AutoModelForCausalLM
import transformers
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import json
import time
import xlsxwriter
import deepspeed
from tqdm import tqdm
import torch

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class TextGenerator():
    def __init__(self, model, use_deepspeed = False, prompt = None):
        
        self.use_deepspeed = use_deepspeed
        self.prompt = prompt
        self.PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
                "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{prompt}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task.\n"
                "아래는 작업을 설명하는 명령어입니다.\n\n"
                "Write a response that appropriately completes the req  uest.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
            ),
        }

    async def generate(self,model, prompt):
        '''
        model = AutoModelForCausalLM.from_pretrained(self.model, return_dict_in_generate=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        generated_outputs = model.generate(input_ids, max_new_tokens = 256, do_sample=True, num_return_sequences=1, output_scores=True)
        gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]

        probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)
        gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
        '''

        # 밑에 코드 옮겨야함 (모델 로드하는데 시간이 오래걸림)

        if model == 'kogpt2':
            self.model = 'model/files/opencoding/kogpt2_epoch_25_lr_1e-05'
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                'skt/kogpt2-base-v2',
                padding_side="right",
                model_max_length=256,
            )
            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )    
            tokenizer.pad_token = tokenizer.eos_token
            self.tokenizer = tokenizer
        elif model == 'polyglot':
            self.model = 'modelPath' #TODO
            self.tokenizer = 'tokenizer' #TODO
        elif model == 'trinity':
            self.model = 'modelPath' #TODO
            self.tokenizer = 'tokenizer' #TODO

        # Using deepspeed

        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        world_size = int(os.getenv('WORLD_SIZE', '1'))
            
        generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=local_rank)

        generator.model = deepspeed.init_inference(generator.model,
                                                mp_size=world_size,
                                                dtype=torch.float,
                                                replace_with_kernel_inject=True
                                                )      
        
        list_result = []

        if type(prompt) is str:
            result = generator(prompt, do_sample=True, max_length=256)
        else:
            for content in tqdm(list_prompt):
                list_result.append(generator(content, do_sample=True, max_length=256))

        return result