import torch
import os
import json
import datasets
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, PeftConfig, PeftModel, LoraConfig, get_peft_model
from tqdm import tqdm
import xlsxwriter
import re
import argparse

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

def remove_abn_type(text):
    return text.replace("_x000D_", "").replace("\n", " ").replace("‘", "").replace("’", "").replace("<unk>", "")

def score_cal(ans_list, pred_list):
    TP = 0 # 실제로 answer에 있고 맞춘거
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

def model_test(model_type, model_dir, save_dir, test_file):
    if model_type == 'polyglot' or model_type == 'trinity' or model_type == 'kogpt':
        config = PeftConfig.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_dir)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    elif model_type == 'kogpt2':
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            'skt/kogpt2-base-v2',
            padding_side="right",
            model_max_length=512,
        )
    else:
        print("Wrong model type")
        os._exit(0)

    if model_type == 'polyglot' or model_type == 'trinity':
        tokenizer.add_special_tokens(
            {
                "eos_token": "<|endoftext|>",
                "bos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
            }
        ) 
    else:
        tokenizer.add_special_tokens(
            {
                "eos_token": "<\s>",
                "bos_token": "<\s>",
                "unk_token": "<\s>",
            }
        )    
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    model.to('cuda') # GPU에서 연산을 진행
    model.eval() # 모델 파라미터 업데이트 없이 evaluation으로만 사용

    df = pd.read_excel(test_file)

    test_data = []

    for i in range(len(df['질문 내용'])):
        test_data.append({'prompt': "다음 텍스트에 대해서 <속성, 의견> 형태로 의견을 추출해줘.", 'question': df['질문 내용'][i], 'input': df['응답값'][i]})

    list_prompt = [PROMPT_DICT['prompt_input'].format_map(tmp) for tmp in test_data]

    list_result = []
    pattern = re.compile(r'<[ㄱ-ㅣ가-힣|a-zA-Z|\s]+[,][ㄱ-ㅣ가-힣|a-zA-Z|\s]+>')

    for content in tqdm(list_prompt):
        input_ids = tokenizer(content, return_tensors='pt', return_token_type_ids=False).to('cuda')
        gened = model.generate(
                **input_ids,
                max_new_tokens=32,
                # early_stopping=True,
                do_sample=True,
                eos_token_id=2,
            )
        output = tokenizer.decode(gened[0][input_ids['input_ids'].shape[-1]:])
        list_result.append(output)

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    workbook = xlsxwriter.Workbook(save_dir + "/result.xlsx")
    worksheet = workbook.add_worksheet()

    count = 0

    answer_list = []
    pred_list = []

    worksheet.write(1, 1, "Question")
    worksheet.write(1, 2, "User input")
    worksheet.write(1, 3, "Comletion1")
    worksheet.write(1, 4, "Comletion2")
    worksheet.write(1, 5, "Comletion3")

    for prompt, result in zip(list_prompt, list_result):
        question = test_data[count]['question'] # Dataset으로부터 질문 가져옴
        worksheet.write(count+2, 1, question) # excel 파일에 작성

        user_input = test_data[count]['input']
        worksheet.write(count+2, 2, user_input) # excel 파일에 

        preds = pattern.findall(result)
        preds = list(set(preds))
        
        for i in range(len(preds)):
            worksheet.write(count+2, 3+i, preds[i])

        count += 1

    workbook.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='모델 테스트 방법입니다.')

    parser.add_argument('--model', required=True, choices=['kogpt2', 'polyglot', 'trinity', 'kogpt'], help='학습 모델(kogpt2/polyglot/trinity/kogpt 중 택 1)')
    parser.add_argument('--model_dir', required=True, help='테스트 할 모델 주소(checkpoint까지 주어야 함)')
    parser.add_argument('--save_dir', default="./", help='테스트 결과를 저장할 주소')
    parser.add_argument('--test_file', required=True, help='테스트 할 엑셀 파일')

    args = parser.parse_args()

    model_test(model_type=args.model, model_dir=args.model_dir, save_dir=args.save_dir, test_file=args.test_file)