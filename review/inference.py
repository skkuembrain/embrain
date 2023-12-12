import torch
import os
import json
import datasets
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from functools import partial
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer, util
import sklearn
from tqdm import tqdm
import re
import openpyxl
import argparse

# 모델을 테스트하는 Class
# 정답이 있는 엑셀 -> 생성한 output + score
# 정답이 없는 엑셀 -> 생성한 output
class Model_test:
    # ------------------------------------------------------------------------------------------
    # func name: __init__(self, data_path, model_path, mode)
    # 목적/용도: 인자를 바탕으로 테스트 class를 생성하는 역할
    # input: data_path(테스트 데이터 경로), model_path(테스트 모델 경로), mode(테스트 모드)
    # output: test class 초기화
    # 주의사항:
    # mode(테스트 모드)의 경우 0, 1, 2, 3의 숫자로 지정하고, 각 숫자는 어떤 task에 대한 테스트를 진행할 것인지를 지정함
    # 0 - 요약 테스트, 1 - 핵심 구문 테스트, 2 - 감성분석 테스트, 3 - 모든 task 테스트
    # ------------------------------------------------------------------------------------------
    def __init__(self, data_path, model_path, mode):
        self.original_data, self.test_data, self.answer_data = self.load_dataset(data_path, mode)
        self.model, self.tokenizer = self.load_model(model_path)
        self.mode = mode


    # ------------------------------------------------------------------------------------------
    # func name: __call__(self, excel_path)
    # 목적/용도: 테스트 데이터로부터 모델의 결과를 생성하여 엑셀 파일로 excel_path 경로에 저장
    # input: excel_path(모델 출력 결과 엑셀 파일이 저장될 경로)
    # output: 모델 출력 결과가 저장된 엑셀 파일
    # ------------------------------------------------------------------------------------------
    def __call__(self, excel_path):
        list_result = []
        
        # data의 모든 context에 대해 모델의 출력을 생성하고, 이를 리스트에 추가
        # tqdm 라이브러리를 사용하여 유저에게 모델 출력 생성 과정이 얼마나 진행되었는지 시각적으로 확인하게 함
        for context in tqdm(self.test_data):
            formatted_result = self.gen_output(context)
            list_result.append(formatted_result)
        
        # 결과 리스트를 엑셀 파일로 변환하여 저장
        self.result_to_excel(list_result, excel_path)
        print("test complete")


    # ------------------------------------------------------------------------------------------
    # func name: load_model(self, path)
    # 목적/용도: 클래스를 선언할 때 지정한 model_path에 저장된 모델을 로드
    # input: path(클래스를 선언할 때 지정한 model_path)
    # output: 학습된 모델과 토크나이저를 불러옴
    # ------------------------------------------------------------------------------------------
    def load_model(self, path):
        config = PeftConfig.from_pretrained(path) # 주어진 경로에서 모델의 설정을 로드
        
        # 모델 양자화를 위해 Bits and Bytes 설정을 구성하여 효율성을 개선
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # 지정된 양자화 설정을 사용하여 사전 훈련된 모델, 토크나이저를 로드
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
        model = PeftModel.from_pretrained(model, path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        # 모델과 토크나이저를 반환함
        return model, tokenizer


    # ------------------------------------------------------------------------------------------
    # func name: gen_output(self, input_text)
    # 목적/용도: input_text로부터 모델이 생성한 결과에 후처리를 적용한 텍스트를 생성함
    # input: input_text(test data 엑셀에 들어 있던 각 text)
    # output: input_text로부터 모델이 생성한 텍스트(후처리 적용)가 반환됨
    # ------------------------------------------------------------------------------------------
    def gen_output(self, input_text):
        with torch.no_grad():
            gened = self.model.generate(
                **self.tokenizer(       
                    input_text,
                    return_tensors='pt',
                    return_token_type_ids=False
                ).to('cuda'),
                max_new_tokens=128, # 생성할 최대 토큰 수를 설정, 현재 128, 더 긴 문장을 원할 경우 이 값을 조정
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.extract_str(self.tokenizer.decode(gened[0]))


    # ------------------------------------------------------------------------------------------
    # func name: remove_pattern(self, text)
    # 목적/용도: 텍스트로부터 특정 패턴(중국어 한자)을 제거하는 용도
    # input: text(모델이 생성한 각 text)
    # output: text로부터 특정 패턴(중국어 한자)가 제거된 텍스트
    # 주의사항:
    # 만약 중국어 패턴이 아니라 다른 패턴(처리가 필요한 다른 패턴)도 제거하고 싶은 경우, pattern의 정규 표현식을 바꾸면 됨
    # ------------------------------------------------------------------------------------------
    def remove_pattern(self, text):
        pattern = re.compile("[\u4e00-\u9fff]+") # 중국어 한자에 해당하는 정규 표현식
        result = re.sub(pattern, "", str(text))
        return result


    # ------------------------------------------------------------------------------------------
    # func name: extract_str(self, result)
    # 목적/용도: result(후처리가 완료된 모델 생성 결과)에서 "Response(응답):" 다음에 오는, 대괄호 안의 문자열을 추출 
    # input: result(후처리가 완료된 모델 생성 결과)
    # output: 완전히 후처리가 완료된 모델 생성 결과 텍스트를 반환
    # 주의사항
    # 해당 모델은 모델 학습 시 후처리(필요한 텍스트만을 뽑고 노이즈를 제거하는 과정)를 원할하게 하기 위해 대괄호 안에 학습 데이터를 넣어 학습하는 방법을 사용
    # 따라서 모델 출력 결과에서 대괄호 내의 부분만을 노이즈가 없는 데이터로 판단하여 나머지 부분을 모두 제거하는 방식으로 후처리를 적용함
    # ------------------------------------------------------------------------------------------
    def extract_str(self, result):
        # 'Response(응답)' 다음의 문자열 찾기
        response_index = result.find('Response(응답):')
        if response_index == -1:
        # 'Response(응답)'가 없으면 원본 문자열 그대로 반환
            return result

        modified_string = self.remove_pattern(result[response_index + 13:])

        match = re.search(r'\[(.*?)\]', modified_string)
        if match:
            extracted_part = match.group(1)
        else:
            extracted_part = ""

        return extracted_part


    # ------------------------------------------------------------------------------------------
    # func name: load_dataset(self, xlsx_path, mode)
    # 목적/용도: xlsx_path에 있는 테스트 데이터를 로드하고, mode에 따라 테스트 데이터 파이썬 객체 형태로 생성하는 역할
    # input: xlsx_path(클래스 생성 시 지정한 테스트 데이터의 경로), mode(클래스 생성 시 지정한 테스트 모드, 세부 사항은 __init__에 기술)
    # output: 테스트 데이터 내의 엑셀 파일에 테스트 mode에 따른 프롬포트를 붙여 모델 input으로 들어갈 텍스트 셋을 파이썬 객체 형태로 만듦
    # 엑셀 파일이 input만 들어온 경우 -> test_data만 생성
    # 엑셀 파일이 input및 정답이 들어온 경우 -> test_data 및 answer_data도 생성하여 스코어 계산이 가능
    # ------------------------------------------------------------------------------------------
    def load_dataset(self, xlsx_path, mode):
        test_data = []
        answer_data = []
        original_data = []

        # 요약 프롬포트
        prompt_summary =  "아래의 텍스트는 요약이 필요한 긴 텍스트입니다.\n"\
                      "이 텍스트에서 긍정적이거나 부정적인 이유를 중심으로 핵심적인 내용을 요약하세요.\n"\
                      "요약문은 텍스트의 핵심적인 내용을 잘 설명해야 합니다.\n"\
                      "요약문의 시작은 [ 로, 끝은 ]로 제시하세요."

        # 핵심구문 프롬프트
        prompt_keyphrase = "아래의 텍스트에서 가장 핵심이 되는 keyphrase들을 3개 이하로 제시하세요.\n"\
                        "각 key phrase는 less than fifteen characters의 짧은 noun phrase로 제시되어야 하고, • 로 시작하며, 개행기호로 구분되어야 합니다.\n"\
                        "noun phrase는 '재미있는 강의내용' 과 같이 명사로 사용될 수 있는 구문입니다. '강의가 재밌다' 와 같이 '다' 로끝나는 문장을 제시하면 안됩니다..\n"\
                        "keyphrase 전체 생성 결과의 시작은 [ 로, 끝은 ]로 제시하세요."
    
        # 감성분석 프롬프트
        prompt_senti = "아래의 텍스트의 감성이 긍정인지 부정인지 긍정, 또는 부정으로 제시하세요.\n"\
                        "감성분석 결과의 시작은 [ 로, 끝은 ]로 제시하세요."

            
        df = pd.read_excel(xlsx_path, sheet_name = None)

        for key in df.keys():
            df[key].dropna(inplace=True)
            df[key].reset_index(drop = True, inplace=True)

        for i in df[key].index:

            input_text = df[key].iloc[i][0]
            original_data.append(input_text)
            if len(df[key].iloc[i]) == 2:
                answer_data.append(df[key].iloc[i][1])
            elif len(df[key].iloc[i]) == 4:
                answer_data.append(df[key].iloc[i][1])
                answer_data.append(df[key].iloc[i][2])
                answer_data.append(df[key].iloc[i][3])

            # 각 모드에 따라 프롬포트를 앞쪽에 붙여 모델에 어떤 task를 수행한 결과를 반환할지 알려줌
            if mode == 0:
                input_text = f"### Instruction(명령어):\n{prompt_summary}\n\n### Input(입력):\n{input_text}\n\n### Response(응답):\n"
                test_data.append(input_text)
            elif mode == 1:
                input_text = f"### Instruction(명령어):\n{prompt_keyphrase}\n\n### Input(입력):\n{input_text}\n\n### Response(응답):\n"
                test_data.append(input_text)
            elif mode == 2:
                input_text = f"### Instruction(명령어):\n{prompt_senti}\n\n### Input(입력):\n{input_text}\n\n### Response(응답):\n"
                test_data.append(input_text)
            elif mode == 3:
                input_text_sum = f"### Instruction(명령어):\n{prompt_summary}\n\n### Input(입력):\n{input_text}\n\n### Response(응답):\n"
                input_text_key = f"### Instruction(명령어):\n{prompt_keyphrase}\n\n### Input(입력):\n{input_text}\n\n### Response(응답):\n"
                input_text_sa = f"### Instruction(명령어):\n{prompt_senti}\n\n### Input(입력):\n{input_text}\n\n### Response(응답):\n"
                test_data.append(input_text_sum)
                test_data.append(input_text_key)
                test_data.append(input_text_sa)
            else:
                raise ValueError("Invalid mode specified")
                
        return original_data, test_data, answer_data


    # ------------------------------------------------------------------------------------------
    # func name: result_to_excel(self, list_result, excel_path)
    # 목적/용도: 모델이 생성한 결과를 엑셀 파일로 만들어 저장
    # input: list_result(모델이 생성한 결과가 저장된 리스트), excel_path(테스트 결과 엑셀이 저장될 경로)
    # output: excel_path 경로에 list_result를 엑셀로 바꾸어 저장함
    # ------------------------------------------------------------------------------------------
    def result_to_excel(self, list_result, excel_path):
        list_result_sum = []
        list_result_key = []
        list_result_sa = []

        # 엑셀파일에 정답이 존재하면 스코어 계산
        if len(self.answer_data) != 0:
            result_df = self.calc_score(list_result)
        # 정답이 존재하지 않으면 생성 결과만 출력
        elif self.mode == 3:
            for i in range(len(list_result)):
                if i % 3 == 0:
                    list_result_sum.append(list_result[i])
                elif i % 3 == 1:
                    list_result[i] = list_result[i].replace('\'', '')
                    list_result[i] = list_result[i].replace(', ', '\n')
                    list_result_key.append(list_result[i])
                else:
                    list_result_sa.append(list_result[i])
            
            result_df = pd.DataFrame(list(zip(self.original_data, list_result_sum, list_result_key, list_result_sa)), columns=['Input', '요약', '핵심구문', '긍부정'])
        else:
            if self.mode == 1:
                for i in range(len(list_result)):
                    list_result[i] = list_result[i].replace('\'', '')
                    list_result[i] = list_result[i].replace(', ', '\n')
            result_df = pd.DataFrame(list(zip(self.original_data, list_result)))
            
        result_df.to_excel(excel_path, index=False)


    # ------------------------------------------------------------------------------------------
    # func name: calc_score(self, list_result)
    # 목적/용도: 모델이 생성한 결과(후처리 적용)로부터 score를 계산해 dataframe 형태로 반환함
    # input: list_result(모델이 생성한 결과가 저장된 리스트)
    # output: 모델이 생성한 결과로부터 각각의 score를 계산하여 pandas dataframe으로 반환
    # 주의 사항:
    # 현재 감성분석의 경우 f1_score와 accuracy를 바탕으로 score를 계산하고, 요약과 핵심구문 추출의 경우 "snunlp/KR-SBERT-V40K-klueNLI-augSTS" 모델을 사용하여 score를 계산합니다
    # flag(0: f1_score 계산, 1: f1_score 미계산) - f1_score는 1대1 비교이기 때문에 긍정/부정이 아닌 다른 문자에 대해 계산을 수행할 수 없으며 오류 발생
    # 그 오류를 막기 위해 긍정/부정 이외의 문자가 나올 경우 f1_score를 계산하지 않음
    # ------------------------------------------------------------------------------------------
    def calc_score(self, list_result):
        list_result_sum = []
        list_result_key = []
        list_result_sa = []
        answer_sum = []
        answer_key = []
        answer_sa = []
        score_sum = []
        score_key = []
        score_sa = []
        
        # 모든 task에 대한 점수를 계산
        if self.mode == 3:
            flag = 0
            # 요약, 핵심구문 task의 score 계산에 사용할 kr-sbert 로드
            model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
            
            for i in range(len(list_result)):
                if i % 3 == 0:
                    list_result_sum.append(list_result[i])
                    answer_sum.append(self.answer_data[i])
                    score_sum.append(util.cos_sim(model.encode(list_result[i]), model.encode(self.answer_data[i])).item())
                    
                elif i % 3 == 1:
                    list_result[i] = list_result[i].replace('\'', '')
                    list_result[i] = list_result[i].replace(', ', '\n')
                    list_result_key.append(list_result[i])
                    answer_key.append(self.answer_data[i])
                    score_key.append(util.cos_sim(model.encode(list_result[i]), model.encode(self.answer_data[i])).item())

                else:
                    # 스코어 계산을 위한 후처리
                    if list_result[i] == self.answer_data[i]:
                        score_sa.append('O')
                    else:
                        score_sa.append('X')
                        if list_result[i] != '긍정' and list_result[i] != '부정':
                            flag = 1
                            print("Can't calculate f1_score")
                    list_result_sa.append(list_result[i])
                    answer_sa.append(self.answer_data[i])

            if flag == 0:
                f1_score = sklearn.metrics.f1_score(list_result_sa, answer_sa, pos_label='긍정', labels=['긍정', '부정', '오류'])
                score_sa.append(f1_score)
                #list의 길이를 맞추기 위함.
                list_result_sum.append("")
                list_result_key.append("")
                self.original_data.append("")
                score_sum.append("")
                score_key.append("")

            accuracy_score = sklearn.metrics.accuracy_score(list_result_sa, answer_sa)
            score_sa.append(accuracy_score)
            #list의 길이를 맞추기 위함.
            if flag == 0:
                list_result_sa.append("")
            self.original_data.append("")
            list_result_sum.append("")
            list_result_key.append("")
            list_result_sa.append("")
            score_sum.append("")
            score_key.append("")

            data = {'Input': self.original_data, '요약': list_result_sum, '핵심구문': list_result_key, '긍부정': list_result_sa, '요약 점수': score_sum, '핵심구문 점수': score_key, '긍부정 점수': score_sa}
            result_df = pd.DataFrame(data)
            
        # 감성분석 task 대한 점수를 계산(f1_score/accuracy)
        elif self.mode == 2:
            flag = 0
            # 스코어 계산을 위한 후처리
            for i in range(len(list_result)):
                if list_result[i] == self.answer_data[i]:
                    score_sa.append('O')
                else:
                    score_sa.append('X')
                    if list_result[i] != '긍정' and list_result[i] != '부정':
                        flag = 1
                        print("Can't calculate f1_score")

            if flag == 0:
                f1_score = sklearn.metrics.f1_score(list_result, self.answer_data, pos_label='긍정', labels=['긍정', '부정'])
                score_sa.append(f1_score)
                #list의 길이를 맞추기 위함.
                self.original_data.append("")
                

            accuracy_score = sklearn.metrics.accuracy_score(list_result, self.answer_data)
            score_sa.append(accuracy_score)
            #list의 길이를 맞추기 위함.
            self.original_data.append("")
            if flag == 0:
                list_result.append("")
            list_result.append("")
        
            data = {'Input': self.original_data, '긍부정': list_result, '긍부정 점수': score_sa}
            result_df = pd.DataFrame(data)
            
        # 핵심 구문 task에 대한 점수를 계산(kr-sbert)
        elif self.mode == 1:
            model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
            for i in range(len(list_result)):
                # 스코어 계산을 위한 후처리
                list_result[i] = list_result[i].replace('\'', '')
                list_result[i] = list_result[i].replace(', ', '\n')
                score_key.append(util.cos_sim(model.encode(list_result[i]), model.encode(self.answer_data[i])).item())
                data = {'Input': self.original_data, '핵심구문': list_result, '핵심구문 점수': score_key}
                result_df = pd.DataFrame(data)
        
        # 요약 task에 대한 점수를 계산(kr-sbert)
        elif self.mode == 0:
            model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
            for i in range(len(list_result)):
                score_sum.append(util.cos_sim(model.encode(list_result[i]), model.encode(self.answer_data[i])).item())
                data = {'Input': self.original_data, '요약': list_result, '요약 점수': score_sum}
                result_df = pd.DataFrame(data)

        return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='모델 테스트 방법입니다.')

    parser.add_argument('--model', required=True, choices=['trinity', 'kogpt'], help='학습 모델(trinity/kogpt 중 택 1)')
    parser.add_argument('--model_dir', required=True, help='테스트 할 모델 주소, 체크포인트까지 입력해야 함')
    parser.add_argument('--save_dir', default="./result", help='테스트 결과를 저장할 주소 (default: "./result")')
    parser.add_argument('--test_file', required=True, help='테스트 할 엑셀 파일')
    parser.add_argument('--mode', default=3, help='테스트할 task mode')

    args = parser.parse_args()
    
    data_path = args.test_file # test할 데이터의 경로
    model_lr = args.model_dir.replace("./model/","")
    excel_path = args.save_dir + "/" + model_lr + ".xlsx" # test한 생성값을 저장할 경로
    test = Model_test(data_path, args.model_dir, int(args.mode)) # class 객체 생성
    test(excel_path) # test


    # mode = 0
    # data_path = "./dataset/score_test.xlsx" # test할 데이터의 경로
    # lr = 3 * 1e-05
    # epoch = 32
    # model_path = "./model/" + str(epoch) + "_" + str(lr) + "/checkpoint-14000" # 모델의 경로
    # excel_path = "./result/" + str(epoch) + "_" + str(lr) + "_checkpoint-14000_mode1_no_score" + ".xlsx" # test한 생성값을 저장할 경로
    # test = Model_test(data_path, model_path, mode) # class 객체 생성
    # test(excel_path) # test
