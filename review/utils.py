import json
import openpyxl
import re
import random

# ------------------------------------------------------------------------------------------
# func name: convert_xlsx_to_json                                                           
# 목적/용도: xlsx 파일을 json 파일로 변환                          
# Input: xlsx 파일의 경로 및 생성할 json 파일을 저장할 경로, 학습 mode                                               
# Output: X
# mode : 0 - 요약, 1 - 핵심구문추출, 2 - 감성분석, 3 - 전체
# xlsx file에는 해당 모드에 맞는 형식으로 데이터가 구성되어 있어야 함.
# 형식
# 1. A행에는 반드시 input
# 2. 사용하는 행의 1열은 제목(사용자가 알아보기 쉬운 제목)
# 3. mode 0,1,2는 A,B 행 사용
# 4. mode 3은 A,B,C,D 행 사용
# ------------------------------------------------------------------------------------------
def convert_xlsx_to_json(file_name, json_file, mode): 

    workbook = openpyxl.load_workbook(file_name)

    sheet = workbook.active

    data = []

    # 요약 프롬포트
    prompt_summary = "아래의 텍스트는 요약이 필요한 긴 텍스트입니다.\n"\
                        "아래의 텍스트에서 내용을 간결하게 요약하세요.\n"\
                        "요약문의 시작은 [ 로, 끝은 ]로 제시하세요."
    
    # 핵심 구문 프롬포트
    prompt_keyphrase = "아래의 텍스트에 대해 가장 핵심이 되는 key noun phrase를 3개 까지만 추출하세요.\n"\
                        "keyphrase의 개수는 세 개 이내로 제시해야 합니다\n"\
                        "각 key phrase는 less than fifteen characters의 짧은 noun phrase로 제시되어야 하고, • 로 시작해야 합니다.\n"\
                        "noun phrase는 '재미있는 강의내용' 과 같이 명사로 사용될 수 있는 구문입니다.\n"\
                        "keyphrase 결과의 시작은 [ 로, 끝은 ]로 제시하세요."
    
    # 감성분석 프롬포트
    prompt_senti = "아래의 텍스트를 분석하여 긍정, 또는 부정으로 제시하세요.\n"\
                    "감성분석 결과의 시작은 [ 로, 끝은 ]로 제시하세요."

    for row in sheet.iter_rows(min_row=2, values_only=True):

        if any(val is None for val in row):
            continue

        if mode == 0: # 요약
            user_input = row[0]

            completion_summary = '[' + row[1] + ']'

            item_summary = {
                "prompt": prompt_summary,
                "input": user_input,
                "completion": completion_summary
            }
            data.append(item_summary)

        elif mode == 1: # 핵심 구문 추출
            user_input = row[0]

            keywords = row[1].split('• ')
            keywords = [keyword.strip().replace('\n', '') for keyword in keywords if keyword.strip()]
            keywords = ["• " + keyword.strip() for keyword in keywords]
            keywords = [re.sub(r'\s{2,}', ' ', keyword) for keyword in keywords]
            completion_keyphrase = '[' + str(keywords)[1:-1] + ']'

            item_keyphrase = {
                "prompt": prompt_keyphrase,
                "input": user_input,
                "completion": completion_keyphrase
            }
            data.append(item_keyphrase)

        elif mode == 2: # 감성 분석
            user_input = row[0]

            completion_senti = '[' + row[1] + ']'

            item_senti = {
                "prompt": prompt_senti,
                "input": user_input,
                "completion": completion_senti
            }
            data.append(item_senti)

        elif mode == 3: # 전체
            user_input = row[0]
            completion_summary = '[' + row[1] + ']'

            keywords = row[2].split('• ')
            keywords = [keyword.strip().replace('\n', '') for keyword in keywords if keyword.strip()]
            keywords = ["• " + keyword.strip() for keyword in keywords]
            keywords = [re.sub(r'\s{2,}', ' ', keyword) for keyword in keywords]
            completion_keyphrase = '[' + str(keywords)[1:-1] + ']'

            completion_senti = '[' + row[3] + ']'

            item_summary = {
                "prompt": prompt_summary,
                "input": user_input,
                "completion": completion_summary
            }
            data.append(item_summary)

            item_keyphrase = {
                "prompt": prompt_keyphrase,
                "input": user_input,
                "completion": completion_keyphrase
            }
            data.append(item_keyphrase)

            item_senti = {
                "prompt": prompt_senti,
                "input": user_input,
                "completion": completion_senti
            }
            data.append(item_senti)

        else:
            raise ValueError("Invalid mode specified")

    workbook.close()

    # random.shuffle(data)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
