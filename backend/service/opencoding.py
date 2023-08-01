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

KOGPT2_PATH = 'model/opencoding/kogpt2_epoch_25_lr_1e-05'
POLYGLOT_PATH = 'model/opencoding/polyglot_epoch_50/checkpoint-264000'
TRINITY_PATH = 'PLEASE ADD PATH !!!!!!' #TODO

class OpencodingGenerator():
    def __init__(self):
        # kogpt2
        print('---setting kogpt2---')
        self.kogpt2 = AutoModelForCausalLM.from_pretrained(KOGPT2_PATH)
        kogpt2_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'skt/kogpt2-base-v2',
            padding_side="right",
            model_max_length=256,
        )
        kogpt2_tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )    
        kogpt2_tokenizer.pad_token = kogpt2_tokenizer.eos_token
        self.kogpt2_tokenizer = kogpt2_tokenizer

        # polyglot
        print('---setting polyglot---')
        config = PeftConfig.from_pretrained(POLYGLOT_PATH)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        self.polyglot = PeftModel.from_pretrained(model, POLYGLOT_PATH)
        self.polyglot_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        #TODO: trinity
        '''
        print('---setting trinity---')
        config = PeftConfig.from_pretrained(TRINITY_PATH)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        self.trinity = PeftModel.from_pretrained(model, TRINITY_PATH)
        self.trinity_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        '''

    async def generateText(self, model, prompt):

        if model == 'kogpt2':
            model = self.kogpt2
            tokenizer = self.kogpt2_tokenizer
        elif model == 'polyglot':
            model = self.polyglot
            tokenizer = self.polyglot_tokenizer
        elif model == 'trinity':
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
            max_new_tokens=48,
            early_stopping=True,
            do_sample=True,
            eos_token_id=2,
        )
        print([tokenizer.decode(gened[0])])

        return [tokenizer.decode(gened[0])]
    
    
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

        result = result.replace(' ', '').replace('><', '>#<').split('#')

        # prediction 제거
        preds = []
        for vocab in result:
            if vocab == '</s>' or '</s>' in vocab:
                break
            if '>' not in vocab or ',' not in vocab:
                break
            if vocab in preds:
                break
            # words = vocab.split(',')
            # first_word = words[0].replace(',','').replace('<', '')
            # second_word = words[1].replace(',', '').replace('>', '')
            preds.append(vocab)
        
        return preds