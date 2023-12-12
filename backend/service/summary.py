import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import urllib.request
from datetime import datetime
from transformers import pipeline, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftConfig, PeftModel
import transformers
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import json
import time
import xlsxwriter
import deepspeed
import re
from tqdm import tqdm
import torch

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

SUM_KOGPT_PATH = 'model/summary/test_trinity_dataver4_12_4_1e-4/checkpoint-7500'
SA_KOGPT_PATH = 'model/summary/kogpt2_sentiment'
KEYWORD_KOGPT_PATH = 'model/Keyphrase'
TOTAL_KOGPT_PATH = 'model/TotalKogpt'

class SummaryGenerator():
    def __init__(self):
        # trinity (summary) on GPU
        print('---setting summary (trinity)---')
        config = PeftConfig.from_pretrained(SUM_TRINITY_PATH)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map = 'auto')
        self.sum_trinity = PeftModel.from_pretrained(model, SUM_TRINITY_PATH)
        self.sum_trinity_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        #TODO: sentiment
        print('---setting sentiment (kogpt2)---')

        id2label = {0: "부정", 1: "긍정"}
        label2id = {"부정": 0, "긍정": 1}
        self.sa_kogpt2 = AutoModelForSequenceClassification.from_pretrained(
            SA_KOGPT2_PATH, num_labels=2, id2label=id2label, label2id=label2id
        )
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
        self.sa_kogpt2_tokenizer = kogpt2_tokenizer
        self.classifier = pipeline('sentiment-analysis', model=self.sa_kogpt2, tokenizer=self.sa_kogpt2_tokenizer)

        #TODO: keyword
        print('---setting keyword (trinity)---')
        config = PeftConfig.from_pretrained(KEYWORD_TRINITY_PATH)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map = 'auto')
        self.key_trinity = PeftModel.from_pretrained(model, KEYWORD_TRINITY_PATH)
        self.key_trinity_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        #TODO: Total
        print('---setting Total (kogpt)---')
        config = PeftConfig.from_pretrained(TOTAL_KOGPT_PATH)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map = 'auto')
        self.total_kogpt = PeftModel.from_pretrained(model, TOTAL_KOGPT_PATH)
        self.total_kogpt_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    async def generateText(self, model, task, prompt):

        print(task)

        if task == 'summary':
            model = self.sum_trinity
            tokenizer = self.sum_trinity_tokenizer
        elif task == 'sa':
            model = self.sa_kogpt2
            tokenizer = self.sa_kogpt2_tokenizer
        elif task == 'keyword':
            model = self.key_trinity
            tokenizer = self.key_trinity_tokenizer
        elif task == 'total':
            model = self.total_kogpt
            tokenizer = self.total_kogpt_tokenizer

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
        if task == 'sa':
            result = self.classifier(prompt)
        else:
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
            result = tokenizer.decode(gened[0])

        return result
    
    async def summaryFormatter(self, result:str):
        def remove_pattern(text):
            pattern = re.compile("[\u4e00-\u9fff]+")
            result = re.sub(pattern, "", str(text))
            return result

        # 'Response(응답)' 다음의 문자열 찾기
        response_index = result.find('Response(응답):')
        if response_index == -1:
            # 'Response(응답)'가 없으면 원본 문자열 그대로 반환
            return result

        modified_string = remove_pattern(result[response_index + 13:])

        match = re.search(r'\[(.*?)\]', modified_string)
        if match:
            extracted_part = match.group(1)
        else:
            extracted_part = ""

        return extracted_part
    
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

        result = result.replace("<unk>", "\n•")
        idx = result.find("<|endoftext|>")
        result = result[:idx]
        
        return result
