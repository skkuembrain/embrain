import torch
import json
import datasets
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from tqdm import tqdm
from utils import convert_xlsx_to_json


# 모델을 학습시키는 Class
class Model_train:

    # ------------------------------------------------------------------------------------------
    # func name: __init__                                                                
    # 목적/용도: 클래스를 선언할 때 model_id와 json_file을 넣고 선언 -> 받은 model_id, json_file로 객체의 변수를 초기화                              
    # Input: model_id, json_file 경로                                                        
    # Output: X
    # KoGPT와 Trinity 학습 가능
    # model_id(KoGPT): "rycont/kakaobrain__kogpt-6b-8bit"
    # model_id(Trinity): "skt/ko-gpt-trinity-1.2B-v0.5"
    # ------------------------------------------------------------------------------------------
    def __init__(self, model_id, json_file):
        if model_id == "kogpt":
            self.model_id = "rycont/kakaobrain__kogpt-6b-8bit"
        elif model_id == "trinity":
            self.model_id = "skt/ko-gpt-trinity-1.2B-v0.5"
        else:
            print("Invalid model")
            os._exit(0)
        self.data = self.load_dataset(json_file)
        self.model = self.load_model()


    # ------------------------------------------------------------------------------------------
    # func name: load_dataset                                                              
    # 목적/용도: json_file을 받고 모델에 학습시킬 데이터셋 구축                           
    # Input: json_file 경로(json_file은 prompt, input, completion으로 구성되어 있음)                                                        
    # Output: data
    # ------------------------------------------------------------------------------------------
    def load_dataset(self, json_file):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id) # class에 저장된 model_id로 tokenizer를 불러옴
        tokenizer.pad_token = tokenizer.eos_token # tokenizer의 pad토큰과 eos토큰을 같게 만들어줌
        a_json = open(json_file, encoding='utf-8-sig') # json 파일을 utf-8-sig 방식으로 인코딩하여 열고 a_json에 저장
        a_load = json.load(a_json) # a_json(json 형식) -> a_load(Dict) json 형식을 Python의 Dictionary 형태로 변환(List 안에 Dictionary 형태의 객체가 존재)
        a_load = datasets.Dataset.from_list(a_load) # List를 Dataset Class의 객체로 변환
        data = datasets.DatasetDict({"train": a_load}) # Dataset -> DatasetDict
        data = data.map(
            lambda x: {'text': f"### Instruction(명령어):\n{x['prompt']}\n\n### Input(입력):\n{x['input']}\n\n### Response(응답):\n{x['completion']}"}
        ) # json_file에서 하나의 객체 당 prompt, input, completion이 존재, x = 하나의 객체를 의미하며 해당 객체의 prompt, input, completion을 위의 형식대로 대입
        data = data.map(lambda samples: tokenizer(samples["text"]), batched=True) # 위에 만들었던 데이터를 토큰화하여 data 생성

        return data # 만든 data를 반환


    # ------------------------------------------------------------------------------------------
    # func name: load_dataset                                                              
    # 목적/용도: 클래스를 선언할 때 받았던 model_id로 모델을 로드                          
    # Input: X                                                       
    # Output: model
    # Lora = Low rank adaptation -> 거대 언어 모델의 학습을 적은 자원으로 수행하기 위해 경량화
    # r: rank의 개수 - 많으면 계산량 증가 & 성능 증가 & 시간 증가, rank가 높아질수록 계산량과 시간은 증가하지만 성능은 증가하지 않을 수 있음
    # lora_alpha: 가중치 행렬의 크기 조절
    # r, lora_alpha의 값을 변형시키며 실험하여 성능 증가를 노려볼 수 있음
    # dropout: 계산 시 5%는 버리고 계산 - 과적합 방지
    # ------------------------------------------------------------------------------------------
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map={"": 0}) # 라이브러리에 있는 pretrain 된 모델을 로드
        model.gradient_checkpointing_enable() # 메모리 사용량 감소 및 모델의 학습 속도 상승을 위한 기술을 활성화하는 함수
        model = prepare_model_for_kbit_training(model) # kbit_training(모델의 학습 속도 개선)을 위한 모델 준비

        config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.05, 
            bias="none",
            task_type="CAUSAL_LM" 
        )

        model = get_peft_model(model, config) 
        model.print_trainable_parameters()
        return model # model을 반환


    # ------------------------------------------------------------------------------------------
    # func name: train_model                                                            
    # 목적/용도: model을 학습시키는 함수                          
    # Input: epoch, batch, learning_rate, dir                                            
    # Output: X
    # batch - 한 번에 몇 개씩 학습할지
    # gradient_accumulation_steps - cuda-memory error를 해결하기 위한 기술 이 step * batch size만큼의 batch size 성능이 나타남
    # epoch - 학습을 몇 번 반복할지
    # learning_rate - 학습의 세기, 높으면 반영을 많이 함
    # float_point - 32bit로 표현하던 실수를 16bit로 표현하여 계산 속도를 올림 (성능은 낮아질 수 있음)
    # logging_steps - logging_steps마다 학습 관련 log를 출력
    # output_dir - 모델을 저장하는 폴더 경로
    #
    # model의 성능에 영향을 주는 parameter들
    # TrainingArgument의 epoch, batch, learning_rate, fp16, optim, gradient_accumulation_steps
    # data_collator의 mlm
    # ------------------------------------------------------------------------------------------
    def train_model(self, epoch, batch, l_rate, dir):
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_id) # class에 저장된 model_id로 tokenizer를 불러옴
        tokenizer.pad_token = tokenizer.eos_token # tokenizer의 pad토큰과 eos토큰을 같게 만들어줌

        trainer = Trainer(
            model=self.model, 
            train_dataset=self.data["train"],
            args=TrainingArguments(
                per_device_train_batch_size=batch, 
                gradient_accumulation_steps=1, 
                num_train_epochs=epoch, 
                learning_rate=l_rate, 
                fp16=True, 
                logging_steps=200, 
                output_dir=dir, 
                optim="paged_adamw_8bit" 
            ),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=True), # data의 전처리, mlm = mask language modeling으로 일부를 마스킹하여 학습 진행
        )
        self.model.config.use_cache = False # cache를 사용하여 같은 계산을 스킵하기 위함(시간은 줄어드나, 성능 저하 우려 존재) -> False
        trainer.train() 
        print('train completed')
        

# ------------------------------------------------------------------------------------------
# func name: load_dataset                                                              
# 목적/용도: learning_rate에 따라 여러 model을 학습시킴                          
# Input: model_train class, epoch, batch, l_rate_min, l_rate_max, dir(학습한 모델의 저장 경로)                                                      
# Output: X
# epoch를 변화시키지 않은 이유는 큰 epoch로 학습시키고 해당 모델의 checkpoint를 불러오면 다른 epoch로 학습한 모델을 불러오는 것과 같은 효과
# learning_rate는 1e-05 ~ 5e-05를 추천
# ------------------------------------------------------------------------------------------
def train_model_multiple(model_train, epoch, batch, l_rate_min, l_rate_max, dir, model_id):
    while l_rate_min <= l_rate_max: # l_rate_min에 1e-05(10^-5)씩 더하며 l_rate_max가 될 때까지 돌림
        dir + "/" + model_id + "/" + str(epoch) + "_" + str(l_rate_min) # 모델을 저장할 폴더 경로 설정
        model_train.train_model(epoch=epoch, batch=batch, l_rate=l_rate_min, dir=dir) # 모델 학습
        l_rate_min += 1e-05


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='모델 학습 설정 방법입니다.')

    parser.add_argument('--model', required=True, choices=['trinity', 'kogpt'], help='학습 모델(trinity/kogpt 중 택 1)')
    parser.add_argument('--epochs', default=32, help='학습 에포크 (default: 32)')
    parser.add_argument('--batch_size', default=4, help='학습 배치 사이즈 (default: 4)')
    parser.add_argument('--l_rate', default=3e-05, help='러닝 레이트 (default: 3e-05)')
    parser.add_argument('--save_dir', default="./model", help='모델 저장 주소 (default: "./model")')
    parser.add_argument('--dataset', required=True, help='학습할 데이터셋 (xlsx 파일)')
    parser.add_argument('--mode', default=3, help='학습 모드 (default: 3)')
    args = parser.parse_args()
    
    file_name = args.dataset
    json_file = file_name.replace("xlsx", "json")
    convert_xlsx_to_json(file_name, json_file, int(args.mode))
    model_train = Model_train(args.model, json_file)
    save_dir = args.save_dir + "/" + args.model + "/" + str(args.l_rate)
    model_train.train_model(args.epochs, args.batch_size, args.l_rate, save_dir)
    
    # mode = 3 # 학습 모드 선택 (0: 요약, 1: 핵심구문추출, 2: 감성분석, 3: 전체)
    
    # file_name = "./dataset/train_data.xlsx" # 학습 데이터로 사용될 데이터 파일의 이름(경로)
    # json_file = "./dataset/train_data.json" # 엑셀 파일을 json 파일로 변경할 때의 이름

    # convert_xlsx_to_json(file_name, json_file, mode) # xlsx -> json으로 변경

    # model_train = Model_train("kogpt", json_file) # model_id와 json_file을 넣고 Class 생성

    # train_model_multiple(model_train, 50, 4, 3e-05, 3e-05, "./model", "kogpt") # 다른 파라미터로 여러 모델 학습
