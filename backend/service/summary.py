import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import urllib.request
from datetime import datetime
from transformers import pipeline, AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
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

TRINITY_PATH = 'model/summary/test_trinity_dataver4_12_4_1e-4/checkpoint-7500'

class SummaryGenerator():
    def __init__(self):

        # trinity (summary) on GPU
        print('---setting trinity(summary)---')
        config = PeftConfig.from_pretrained(TRINITY_PATH)
        print('setting model')
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map = 'auto')
        print('loading model')
        self.trinity = PeftModel.from_pretrained(model, TRINITY_PATH)
        print('loading tokenizer')
        self.trinity_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        #TODO: sentiment

        #TODO: keyword

        

    async def generateText(self, model, task, prompt):

        if task == 'summary':
            model = self.trinity
            tokenizer = self.trinity_tokenizer
        elif task == 'sentiment':
            model = self.polyglot #TODO
            tokenizer = self.polyglot_tokenizer #TODO
        elif task == 'keyword':
            self.model = 'modelPath' #TODO
            self.tokenizer = 'tokenizer' #TODO

        # Using deepspeed
        '''
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        world_size = int(os.getenv('WORLD_SIZE', '1'))
            
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=local_rank)
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

        '''

        model.to('cuda')
        gened = model.generate(
            **tokenizer(
                prompt,
                return_tensors='pt',
                return_token_type_ids=False
            ).to('cuda'),
            max_new_tokens=128,
            early_stopping=True,
            do_sample=True,
            eos_token_id=2,
        )
        print([tokenizer.decode(gened[0])])

        return tokenizer.decode(gened[0])
    
    
    async def formatter(self, result:str):
        sentence = ''
        # response와 sentence 추출
        for paragraph in result.split('\n\n'):
            if paragraph.startswith('### Response(응답):'):
                result = result.replace('\n', '')
                result = paragraph.split(':')[1]
                break
            elif paragraph.startswith('### Input(입력):\n'):
                sentence = paragraph.split('\n')[1]
                sentence = sentence[1:]
        
        return result