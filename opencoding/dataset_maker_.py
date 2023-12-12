# 오픈코딩 데이터셋을 제작하는 파일입니다.

import os
import pandas as pd
import urllib.request
import random
from datetime import datetime
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import LTokenizer
import json
import time
import xlsxwriter
from tqdm import tqdm
import torch

class DatasetConverter():
  # 초기화 메서드
        # path: 데이터셋이 저장된 경로
        # train_percentage: 훈련 데이터셋의 비율
    def __init__(self, path, verbose=False):
        self.verbose = verbose
        self.path = path
        # 문장을 나누는 데 사용되는 조사와 예외 단어 리스트(잘 걸러내지 못하는 경우가 있어 직접 추가해준 경우)
        self.split = ['은', '는', '이', '가', '에', '에게', '을', '를']
        self.exception = ['많이', '많은', '적은', '깨끗이', '깊은', '찾는', '젊은', '원하는', '손잡이', '입을', '특이', '쏘는', '추가', '없이', '없는', '없을', '아이', '유아', '먹는', '먹기에', '싶은', '마시는', '마시기에', '첨가', '무첨가', '접을', '접는', '용이', '먹을', '먹는', '먹기에', '먹기를', '원하는', '원치 않는', '스프레이', '찾는', '지우는', '지우게', '지우기에', '취하는', '취하기에', '부가']
        self.l_tokenizer : LTokenizer


    #------------------------------------------------------------------------------------------
    # func name: train                                                            
    # 목적/용도: 아래 github link의 text파일(2016-10-20.txt)의 223,357문장과,
    #            학습 데이터셋으로 사용할 excel data을 학습시켜,
    #            단어 추출기와 명사 추출기를 훈련시키는 작업을 수행.
    #------------------------------------------------------------------------------------------
    def train(self):

        # 주어진 url에서 텍스트 파일 다운로드
        urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")


        # soynlp 라이브러리의 DoublespaceLineCorpus 클래스를 활용하여 다운로드한 텍스트 파일 읽어들임
        corpus = DoublespaceLineCorpus("2016-10-20.txt", iter_sent=True)


        # WordExtractor 클래스와 LRNounExtractor_v2 클래스를 초기화하고, 
        # 주어진 데이터셋을 활용하여 훈련
        word_extractor = WordExtractor(min_frequency=100,
            min_cohesion_forward=0.05,
            min_right_branching_entropy=0.0)
        word_extractor.train(corpus)

        noun_extractor = LRNounExtractor_v2(verbose=False)
        noun_extractor.train(corpus)


        sentence = []

        # 주어진 경로 내의 모든 엑셀 파일을 읽어들인 후, 

        # 추출할 데이터가 있는 열(column)을 usecols 파라미터를 활용하여 지정
        # 입력으로 들어오는 엑셀 데이터 구성은 다음과 같음
        # col 0 : input (user의 raw한 입력값)
        # col 1 : 관점-의견쌍 문장1 (해당 input에 대한 <속성, 의견> 문장)
        # col 2 : 관점-의견쌍 문장2 (해당 input에 대한 <속성, 의견> 문장)
        # col 3 : 관점-의견쌍 문장3 (해당 input에 대한 <속성, 의견> 문장)
        # 현재 데이터의 경우 최대 3개의 answer를 읽어올 수 있도록 되어있음.
        for file in os.listdir(self.path):
            if not file.endswith('.xlsx'): continue
            file = self.path + "/" + file
            df = pd.read_excel(file, header=0, usecols={1,2,3,4})
            df = df.iloc[1:]
            df.dropna(axis=0, how="all")
            df = df.fillna("")
            for i in df.index:
                for col in df.columns:
                    if df[col][i] != "":
                        sentence.append(df[col][i])


        # WordExtractor 클래스와 LRNounExtractor_v2 클래스를 활용하여 추출된 데이터에 대해 단어와 명사를 추출.
        # 추출된 단어와 명사에 대해 각각의 점수를 계산하여, LTokenizer 클래스를 초기화.

        words = word_extractor.train(sentence)
        words = word_extractor.extract()
        nouns = noun_extractor.train_extract(sentence)

        cohesion_score = {word:score.cohesion_forward for word, score in words.items()}

        noun_scores = {noun:score.score for noun, score in nouns.items()}
        combined_scores = {noun:score + cohesion_score.get(noun, 0)
            for noun, score in noun_scores.items()}

        combined_scores.update(
            {subword:cohesion for subword, cohesion in cohesion_score.items()
            if not (subword in combined_scores)})

        self.l_tokenizer = LTokenizer(scores=combined_scores)


    #------------------------------------------------------------------------------------------
    # func name: extract                                                  
    # 목적/용도: 주어진 문장에서 특정 단어를 추출하여 "<aspect, opinion>" 형식으로 반환하는 기능을 수행
    # - word: 토큰화된 단어를 나타내는 변수
    # - found: 추출된 단어를 찾았는지 여부를 나타내는 불리언 변수. 초기값은 False로 설정
    # - aspect: 추출된 단어를 저장할 변수. 초기값은 "NULL"로 설정.
    # - opinion: 추출된 단어의 주변 문맥을 저장할 변수. 초기값은 빈 문자열("")
    # - temp: 임시적으로 추출된 단어를 저장할 변수. 초기값은 빈 문자열("")
    # Input: 관점-의견쌍 문장, Sentence (String)
    # Output: <관점, 의견>, <aspect, opinion> (String)
    #------------------------------------------------------------------------------------------
    def extract(self, sent):
        found = False
        aspect = "NULL"
        opinion = ""
        temp = ""

        # self.l_tokenizer.tokenize(sent, flatten=False)를 통해 주어진 문장을 토큰화
        for word in self.l_tokenizer.tokenize(sent, flatten=False):
            if not found:

                # 현재 단어가 self.split에 포함된 단어인 경우 found를 True로 설정하고 aspect에 현재 단어를 저장.
                if word[1] in self.split:
                    found = True
                    aspect = word[0]


                # 현재 단어의 마지막 문자가 self.split에 포함되지 않고, 
                # 예외 단어 목록(self.exception)에도 포함되지 않는 경우 
                # found를 True로 설정하고 aspect에 현재 단어의 마지막 문자를 제외한 부분을 저장.
                elif word[0][-1] in self.split and word[0] not in self.exception:
                    found = True
                    aspect = word[0][:-1]

                # 위의 두 조건에 모두 해당하지 않는 경우, 현재 단어의 각 문자를 temp에 추가.    
                else:
                    for string in word:
                        temp += string
                    temp += " "


            # 현재 단어의 각 문자를 opinion에 추가.
            else:
                for string in word:
                    opinion += string
                opinion += " "

            
        # found가 True인 경우 : temp와 aspect를 결합하여 aspect에 저장
        # found가 False인 경우 : temp를 opinion에 저장
        # 추출된 aspect와 opinion을 "<"와 ">"로 묶은 문자열을 반환
        if found:
            aspect = temp + aspect
        else:
            opinion = temp

        return "<" + aspect + ", " + opinion[:-1] + ">"


    #------------------------------------------------------------------------------------------
    # func name: convert                                                            
    # 목적/용도: 지정된 경로 내의 모든 엑셀 파일을 읽어들인 후, 각 엑셀 파일 내에서 정해진 열(column)에 해당하는 데이터를 추출하는 작업을 수행.
    # input: excel data
    # Output: total_dataset (json 파일)
    #         prompt, question, input, completion (1개~3개)로 구성됨.
    #------------------------------------------------------------------------------------------
    def convert(self):
        # 생성된 데이터셋을 저장할 리스트
        dataset = []
        
        # 난수 생성을 위한 시드값을 현재 시간으로 설정
        random.seed(datetime.now().timestamp())

        for file in os.listdir(self.path):
            #선택한 폴더에 있는 파일을 나타냅니다.
            file = self.path + "/" + file
            if file.endswith('.xlsx'):
                if self.verbose: print(file)
                df = pd.read_excel(file)
                first_row = df.columns.tolist()
                
                # 추출할 데이터가 있는 열(column)을 usecols 파라미터를 활용하여 지정
                # 입력으로 들어오는 엑셀 데이터 구성은 다음과 같음
                # col 0 : input (user의 raw한 입력값)
                # col 1 : 관점-의견쌍1 (해당 input에 대한 <속성, 의견>값)
                # col 2 : 관점-의견쌍2 (해당 input에 대한 <속성, 의견>값)
                # col 3 : 관점-의견쌍3 (해당 input에 대한 <속성, 의견>값)
                # 현재 데이터의 경우 최대 3개의 completion을 읽어올 수 있도록 되어있음.
                df = pd.read_excel(file, header=0, usecols={1,2,3,4})

                df = df.iloc[1:]
                first_row = df.columns[df.columns.str.contains('Unnamed') == False].tolist()
                df.columns = ['input', 'ans1', 'ans2', 'ans3']
                df = df.dropna(axis=0, how="all")
                df = df.fillna("")

                # prompt, question, input, answer를 dataset에 넣음
                for i in df.index:
                    data = {}
                    data['prompt'] = "다음 텍스트에 대해서 <속성, 의견> 형태로 의견을 추출해줘."
                    data['question'] = first_row
                    data['input'] = df['input'][i]
                    completion = ""
                    for ans in ["ans1", "ans2", "ans3"]:
                        if(df[ans][i] != ""):
                            completion += self.extract(df[ans][i]) + " "
                    data['completion'] = completion[:-1]
                    dataset.append(data)


        # 데이터셋 무작위로 섞음
        random.shuffle(dataset)

        if self.verbose:
            print(dataset)
        
        total_path = self.path + "/total_dataset.json"

        with open(total_path, 'w', encoding='UTF-8-sig') as f:
            f.write(json.dumps(dataset, indent=2, ensure_ascii=False))



    #------------------------------------------------------------------------------------------
    # func name: json_to_excel                                                            
    # 목적/용도: JSON 파일을 읽어와서 데이터프레임으로 변환한 뒤, 
    #            변환된 데이터프레임을 엑셀 파일로 저장
    # input: json file (total_dataset.json)
    # Output: excel 파일
    #------------------------------------------------------------------------------------------
    def json_to_excel(self):
        # total dataset 불러와서 표현 한번에 확인하기 위한 코드
        file_name = extract_path + '/total_dataset.json'
        with open(file_name, 'r', encoding='utf-8-sig') as file:
            data = json.load(file)

        # 데이터프레임을 만들기 위한 빈 리스트 초기화
        df_data = []


        for entry in data:
            question = entry["question"]
            input_data = entry["input"]  # "input" 필드 추가
            completions = entry["completion"].replace("> <", ">#<")
            completions = completions.strip("<>").split(">#<")

            # "completion"에 여러 개의 "<속성, 의견>"이 있는 경우, 각각을 처리
            for c in completions:
                attributes_and_opinions = c.split(", ")
                attribute = attributes_and_opinions[0]  # 첫 번째 값은 attribute에 할당
                opinion = ", ".join(attributes_and_opinions[1:])  # 나머지 값은 opinion에 할당
                df_data.append([question, input_data, attribute, opinion])

        # 데이터프레임 생성
        df = pd.DataFrame(df_data, columns=["question", "input", "attribute", "opinion"])


        # 질문, 속성, 의견을 그룹화 (질문, 인풋, 속성, 의견이 전부 일치하는 경우 count +1)
        grouped = df.groupby(["question", "input", "attribute", "opinion"]).size().reset_index(name='count')

        # 엑셀 파일로 저장
        with pd.ExcelWriter('output.xlsx') as writer:
            grouped.to_excel(writer, index=False, sheet_name='Data')




if __name__ == "__main__":
   
   # excel 데이터가 있는 폴더명
    extract_path = os.path.dirname(os.path.abspath(__file__))

    # 데이터 변환기 생성 및 훈련
    converter = DatasetConverter(extract_path, verbose=False)
    converter.train()

    # json file 생성
    converter.convert()

    # 데이터셋 추가 수작업을 위해 엑셀파일로 변환
    converter.json_to_excel()
