import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import urllib.request
import random
from datetime import datetime
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
    def __init__(self, use_deepspeed = False, prompt = None):
        
        self.use_deepspeed = use_deepspeed
        self.prompt = prompt
        self.kogpt2 = AutoModelForCausalLM.from_pretrained('models/opencoding/kogpt2_epoch_25_lr_1e-05')
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

    @torch.inference_mode
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
            model = self.kogpt2
            tokenizer = self.kogpt2_tokenizer
        elif model == 'polyglot':
            self.model = 'modelPath' #TODO
            self.tokenizer = 'tokenizer' #TODO
        elif model == 'trinity':
            self.model = 'modelPath' #TODO
            self.tokenizer = 'tokenizer' #TODO

        # Using deepspeed

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

        return result
    
    
    async def formatter(result):
        sentence = ''

        # response와 sentence 추출
        for paragraph in result.split('\n\n'):
            if paragraph.startswith('### Response(응답):\n'):
                result = paragraph.split('\n')[1]
                result = result[1:]
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