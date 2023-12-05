# 모델 학습 및 성능 테스트하는 파일입니다.

import torch
from torch.utils.data import Dataset
import os
import json
import datasets
import pandas as pd
from datetime import datetime
import time
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from tqdm import tqdm
import xlsxwriter
import re
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Sequence
import logging
from dataclasses import dataclass
import copy

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
        "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{prompt}\n\n### Question(질문):\n{question}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n"
        "아래는 작업을 설명하는 명령어입니다.\n\n"
        "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
    ),
}

# data config
IGNORE_INDEX = -100

# Special token들 정의
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "<\s>"
DEFAULT_BOS_TOKEN = "<\s>"
DEFAULT_UNK_TOKEN = "<\s>"

class SFT_dataset(Dataset):
    '''SFT dataset by wygo'''
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, start, end, verbose=False):
        super(SFT_dataset, self).__init__()
        logging.warning("Loading data...")
        
        # 프롬프트 템블릿 정의
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]  # 템플릿 가져오기
        
        # 학습 데이터 불러오기
        with open(data_path, "r", encoding='utf-8-sig') as json_file:
            list_data_dict = json.load(json_file)

        if DEFAULT_EOS_TOKEN not in list_data_dict[0]['completion']:
            for i in range(len(list_data_dict)):
                list_data_dict[i]['completion'] = list_data_dict[i]['completion'] + DEFAULT_EOS_TOKEN
            
        list_data_dict = list_data_dict[int(len(list_data_dict) * start):int(len(list_data_dict) * end)] # 데이터셋의 60%만 training에 사용
        
        for i in range(len(list_data_dict)):
            for key in list_data_dict[i].keys():
                list_data_dict[i][key] = self.remove_abn_type(list_data_dict[i][key]) # 이상 데이터 제거
            
        # 입력 프롬프트 (sources)
        sources = []
        for example in list_data_dict:
            sources.append(prompt_input.format_map(example)) # 템플릿에 맞춰 프롬프트 작성

        # 정답값 (targets)
        targets = []
        for example in list_data_dict:
            targets.append(f"{example['completion']}{tokenizer.eos_token}") # Completion + eos 토큰으로 target 정의

        if verbose:
            idx = 0
            print((sources[idx]))
            print((targets[idx]))
            print("Tokenizing inputs... This may take some time...")
            
        examples = [s + t for s, t in zip(sources, targets)] # 입력 프롬프트와 정답값을 합친 값

        # Tokenizer를 이용해 만든 sources & examples tokenize
        # _tokenize_fn() 함수는 아래에 정의
        sources_tokenized = self._tokenize_fn(sources, tokenizer)  # source만 tokenize
        examples_tokenized = self._tokenize_fn(examples, tokenizer)  # source + target tokenize
        
        input_ids = examples_tokenized["input_ids"] 
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX # targets(를 tokenize 한 것에서 앞에 sources까지는 전부 ignore index로 설정)
            
        # e.g. 
        # sources = [1, 2, 3, 4, 5, 6], examples = [1, 2, 3, 4, 5, 6, 7, 8] 일 때 (=> 7, 8이 정닶값의 token으로 볼 수 있음)
        # label은 [-100, -100, -100, -100, -100, -100, 7, 8] 으로 정의됨

        # 여기서 data_dict은 examples(source + target)을 tokenize한 값(input_ids)과, 
        # examples에서 source 부분만 ignore index로 바꾼 값(labels)을 가지게 됨
        data_dict = dict(input_ids=input_ids, labels=labels)   
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        logging.warning("Loading data done!!: %d"%(len(self.labels)))    
    
    def remove_abn_type(self, text):
        return text.replace("_x000D_", "").replace("\n", " ").replace("‘", "").replace("’", "").replace("<unk>", "")
        
    # 파라미터로 받은 'strings'를 'tokenizer'를 이용해 tokenize
    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings # strings 안에 있는 데이터를 하나씩 tokenizer하여 list로 저장
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        
        # dictionary 형태로 반환
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )        
        
    def __len__(self): #샘플 개수
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]: #샘플 가져오기
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        print(self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

class OpenCodingTrain():
    # ------------------------------------------------------------------------------------------
    # func name: remove_abn_type                                                                
    # 목적/용도: 유저 응답값에 있는 이상 데이터를 전처리하여 제거                                   
    # Input: 유저 응답값 (String)                                                            
    # Output: 이상값을 데이터를 제거한 유저 응답값 (String)
    # ------------------------------------------------------------------------------------------
    def remove_abn_type(self, text):
        return text.replace("_x000D_", "").replace("\n", " ").replace("‘", "").replace("’", "").replace("<unk>", "")

    # ------------------------------------------------------------------------------------------
    # func name: print_trainable_parameters                                                                
    # 목적/용도: Base 모델에서 학습 가능한 파라미터 비율 출력                                  
    # Input: Base model (AutoModelForCausalLM, PeftModel 등)                                                         
    # Output: 학습 가능 파라미터 비율 출력
    # ------------------------------------------------------------------------------------------
    def print_trainable_parameters(self,model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    # ------------------------------------------------------------------------------------------
    # func name: train_model                                                                
    # 목적/용도: 모델 학습                              
    # Input:
    # - model: 학습할 모델 종류 ({"kogpt2", "polyglot", "trinity", "kogpt"} 중에 택 1)
    # - epochs: 학습 에포크 (int > 0)
    # - batch_size: 학습 배치 사이즈 (int > 0)
    # - data_path: 학습 데이터 ("~~~.json" 형태의 String)
    # - save_dir: 모델 저장 주소 (String)
    # - save_step: n step마다 모델 파라미터 저장 (int > 0)
    # Output: 
    # - Input으로 주어진 save_dir에 학습된 모델 저장
    # - loss_log.txt / loss_log(epoch).png : 학습 loss 로그
    # - answer_log.txt / answer_log.xlsx : 전체 테스트 로그
    # - error_log.txt / error_log.xlsx : 틀린 생성 값들에 대한 테스트 로그
    # ------------------------------------------------------------------------------------------
    def train_model(self, model, epochs, batch_size, data_path, save_dir, save_step):
        if model == 'polyglot' or model == 'trinity' or model == 'kogpt':
            if model == 'polyglot': model_id = "EleutherAI/polyglot-ko-1.3b"  
            elif model == 'trinity': model_id = "skt/ko-gpt-trinity-1.2B-v0.5"
            else: model_id = "rycont/kakaobrain__kogpt-6b-8bit"
            
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0})
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model) 
            
            config = LoraConfig(
                r=8,
                lora_alpha=32,
                # target_modules=["query_key_value"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, config)
            self.print_trainable_parameters(self.model)
            
        
        elif model == "kogpt2":
            model_id = 'skt/kogpt2-base-v2'
            self.model = AutoModelForCausalLM.from_pretrained(model_id)

        else:
            print("Invalid model")
            os._exit(0)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="right",
            model_max_length=256,
        )

        # 앞서 정의한 special tokens 추가
        self.tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )    
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        train_dataset = SFT_dataset(data_path=data_path, tokenizer=self.tokenizer, start=0.0, end=0.6)
        eval_dataset  = SFT_dataset(data_path=data_path, tokenizer=self.tokenizer, start=0.6, end=0.8)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

        if not os.path.exists(save_dir + "/logs"): os.makedirs(save_dir + "/logs") 

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            args=transformers.TrainingArguments(
                output_dir=save_dir, # 모델 저장 폴더
                overwrite_output_dir=True, # 모델 덮어쓰기 유무
                num_train_epochs=int(epochs), # 학습 epoch 수
                per_device_train_batch_size=int(batch_size), # 학습 batch 사이즈
                per_device_eval_batch_size=int(batch_size), # Eval batch 사이즈
                save_steps=int(save_step), # 모델이 저장될 steps 수 (n step마다 모델 파라미터 저장)
                logging_strategy="epoch",
                evaluation_strategy="epoch", # evaluation 기준을 epoch로
                warmup_steps=5,# lr scheduler가 warmup할 step 수
                prediction_loss_only=True,
            )
        )

        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()

        loss_df = pd.DataFrame(trainer.state.log_history)
        
        log = ""
        
        epoch_log = [i+1 for i in range(epoch)]
        loss_log = [loss for loss in loss_df['loss'].dropna(axis=0)]
        eval_loss_log = [eval_loss for eval_loss in loss_df['eval_loss'].dropna(axis=0)]

        with open(dir + "/logs/loss_log.txt", 'w') as f:
            for i in range(len(epoch_log)):
                log += "Epoch : " + str(epoch_log[i]) + "\t\tLoss : " + str(loss_log[i]) + "\t\tEval_Loss : " + str(eval_loss_log[i]) + "\n"
            print()
            f.write(log)

        plt.plot(epoch_log, loss_log, label="loss")
        plt.plot(epoch_log, eval_loss_log, label="validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(dir + "/logs/loss_log(epoch).png")

    # ------------------------------------------------------------------------------------------
    # func name: score_cal                                                                
    # 목적/용도: 모델의 예측값 평가                                  
    # Input:
    # - ans_list: 정답 리스트 (2D-array 형태 e.g. [[<모자, 예쁘다>, <상품, 많다>], [<음식, 맛있다>], ...])      
    # - pred_list: 정답 리스트 (2D-array 형태 e.g. [[<모자, 예쁘다>, <상품, 많다>, <가게, 많다>], [<음식, 맛있다>], ...])                                           
    # Output: Accuracy, Precision, Recall, F1-score
    # ------------------------------------------------------------------------------------------
    def score_cal(self, ans_list, pred_list):
        TP = 0 # 실제로 answer에 있고 맞춘 것
        TN = 0 # answer에 없고 생산 안한 것
        FP = 0 # answer에 없는데 정답이라고 한 것
        FN = 0 # answer에 있는데 맞추지 못한 것

        for i in range(len(ans_list)):
            for j in range(len(pred_list[i])):
                if pred_list[i][j] in ans_list[i]:
                    TP += 1
                else:
                    FP += 1
            for j in range(len(ans_list[i])):
                if ans_list[i][j] not in pred_list[i]:
                    FN += 1

        accuracy = (TP + FN) / (TP + FN + FP + TN) if TP + FN + FP + TN != 0 else 0
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1_score = 2 * (recall * precision) / (recall + precision) if recall + precision != 0 else 0

        return accuracy, precision, recall, f1_score

    # ------------------------------------------------------------------------------------------
    # func name: test_model                                                            
    # 목적/용도: 학습한 모델 테스트                                
    # Input:
    # - data_path: 학습 데이터 ("~~~.json" 형태의 String)
    # - save_dir: 테스트 결과값을 저장할 주소 (String)
    # Output: result.xlsx (테스트 결과 파일)
    # ------------------------------------------------------------------------------------------
    def test_model(self, data_path, save_dir):
        a_json = open(data_path, encoding = 'utf-8-sig')
        a_load = json.load(a_json)

        a_load = a_load[int(len(a_load) * 0.8):]

        list_prompt = [PROMPT_DICT['prompt_input'].format_map(tmp) for tmp in a_load]
        
        list_result = []

        pattern = re.compile(r'<[ㄱ-ㅣ가-힣|a-zA-Z|\s]+[,][ㄱ-ㅣ가-힣|a-zA-Z|\s]+>')

        self.model.eval()
        start = time.time()
        
        for content in tqdm(list_prompt):
            input_ids = self.tokenizer(content, return_tensors='pt', return_token_type_ids=False).to('cuda')
            gened = self.model.generate(
                    **input_ids,
                    max_new_tokens=32,
                    do_sample=True
            )
            output = self.tokenizer.decode(gened[0][input_ids['input_ids'].shape[-1]:])
            list_result.append(output)
        
        end = time.time()

        save_path = save_dir + "/logs"
        if not os.path.exists(save_path): os.makedirs(save_path) 
        with open(save_path + "/answer_log.txt", 'w', encoding='utf8', errors="ignore") as f:
            workbook = xlsxwriter.Workbook(save_path + "/answer_log.xlsx")
            worksheet = workbook.add_worksheet()

            count = 0

            answer_list = []
            pred_list = []

            worksheet.write(1, 1, "Question")
            worksheet.write(1, 2, "User input")
            worksheet.write(1, 3, "Answer")
            worksheet.write(1, 4, "Comletion1")
            worksheet.write(1, 5, "Comletion2")
            worksheet.write(1, 6, "Comletion3")

            for prompt, result in zip(list_prompt, list_result):
                f.write("-----------------------------------------------------------------------------------------------\n\n")
                question = a_load[count]['question'] # Dataset으로부터 질문 가져옴
                worksheet.write(count+2, 1, question) # excel 파일에 작성
                f.write("Question: ")
                f.write(question) # txt 파일에 작성
                f.write("\n")

                user_input = a_load[count]['input']
                worksheet.write(count+2, 2, user_input) # excel 파일에 작성
                f.write("Input: ")
                f.write(user_input) # txt 파일에 작성
                f.write("\n")

                answers = pattern.findall(a_load[count]['completion']) # 모델 생성 값
                answer_list.append(answers)

                preds = pattern.findall(result)
                pred_list.append(preds)
                
                for pred in preds:
                    worksheet.write(count+2, 4, pred)

                f.write("Prediction: ")
                f.write(result)
                f.write("\n\n")

                answers_str = ""
                for ans in answers: answers_str += (ans + " ") # 실제 정답 값 String 형태로 변환
                worksheet.write(count+2, 3, answers_str) # excel에 작성
                f.write("Answer : ")
                f.write(answers_str) # txt 파일에 작성
                f.write("\n\n")

                count += 1

        workbook.close()

        wrong_count = 0

        workbook = xlsxwriter.Workbook(save_path + "/error_log.xlsx")
        worksheet = workbook.add_worksheet()
        worksheet.write(1, 1, "Question")
        worksheet.write(1, 2, "User input")
        worksheet.write(1, 3, "Answer")
        worksheet.write(1, 4, "Comletion1")
        worksheet.write(1, 5, "Comletion2")
        worksheet.write(1, 6, "Comletion3")

        with open(save_path + "/error_log.txt", 'w', encoding="utf-8") as f:
            for i in range(len(answer_list)):
                ans_correct = True
                for ans in answer_list[i]:
                    if ans not in pred_list[i]:
                        ans_correct = False
                for pred in pred_list[i]:
                    if pred not in answer_list[i]:
                        ans_correct = False

                if not ans_correct:
                    wrong_count += 1
                    wrong_answer = ""
                    wrong_pred = ""
                    for ans in answer_list[i]: wrong_answer += (ans + " ")
                    for pred in pred_list[i]: wrong_pred += (pred + " ")
                    error_msg = "\n\nWrong answer #" + str(wrong_count) + "\nQuestion: " + a_load[i]['question'] + "\nInput: " + a_load[i]['input'] + "\nAnswer: " + wrong_answer + "\nPrediction: " + wrong_pred
                    f.write(error_msg)
                    worksheet.write(wrong_count+1, 1, a_load[i]['question'])
                    worksheet.write(wrong_count+1, 2, a_load[i]['input'])
                    worksheet.write(wrong_count+1, 3, wrong_answer)
                    for j in range(len(pred_list[i])): worksheet.write(wrong_count+1, 4+j, pred_list[i][j])

            accuracy, precision, recall, f1_score = self.score_cal(answer_list, pred_list)

            result_msg = "\n\n-------- Test results --------"
            result_msg += "\naccuracy_score : " + str(accuracy)
            result_msg += "\nrecall_score : " + str(recall)
            result_msg += "\nprecision_score : " + str(precision)
            result_msg += "\nf1_score : " + str(f1_score)
            result_msg += "\n\nTime taken : " + str(end - start)

            f.write(result_msg)
        
        workbook.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='모델 학습 설정 방법입니다.')

    parser.add_argument('--model', required=True, choices=['kogpt2', 'polyglot', 'trinity', 'kogpt'], help='학습 모델(kogpt2/polyglot/trinity/kogpt 중 택 1)')
    parser.add_argument('--epochs', default=20, help='학습 에포크 (default: 20)')
    parser.add_argument('--batch_size', default=8, help='학습 배치 사이즈 (default: 8)')
    parser.add_argument('--save_step', default=500, help='모델 저장 스탭 수 (default: 500)')
    parser.add_argument('--save_dir', default="./models", help='모델 저장 주소 (default: "./model")')
    parser.add_argument('--dataset', required=True, help='학습할 데이터셋 (json 파일)')
    args = parser.parse_args()

    if args.model == 'polyglot' or args.model == 'trinty':
        DEFAULT_EOS_TOKEN = "<|endoftext|>"
        DEFAULT_BOS_TOKEN = "<|endoftext|>"
        DEFAULT_UNK_TOKEN = "<|endoftext|>"

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    models = OpenCodingTrain()
    models.load_model(model=args.model, epochs=args.epochs, batch_size=args.batch_size, data_path=args.dataset, save_dir=args.save_dir, save_step=args.save_step)
    models.test_model(data_path=args.dataset, save_dir=args.save_dir)
