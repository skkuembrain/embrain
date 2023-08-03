from fastapi import APIRouter, File, UploadFile, Depends
from service import SummaryGenerator
from fastapi_restful.inferring_router import InferringRouter
from fastapi_utils.cbv import cbv
import pandas as pd
import xlsxwriter
from pydantic import BaseModel
from fastapi.responses import FileResponse


class TextInferenceInput(BaseModel):
    text: str
    task: str # summary, sentiment, keyword
    model: str

router = InferringRouter()

@cbv(router)
class Opencoding:
    #svc: TextGenerator = Depends(TextGenerator)
    print('**** setting Summary models ****')
    svc = SummaryGenerator()

    @router.get('/sum/hello')
    async def hello(self):
        return 'hello'

    @router.post('/sum/text')
    async def inference_text(self, input: TextInferenceInput):

        input.text = input.text.replace('\n', '')
        
        if input.task == 'summary':
            prompt = ("아래의 텍스트는 요약이 필요한 긴 텍스트입니다.\n"
              "이 텍스트에서 긍정적이거나 부정적인 이유를 중심으로 핵심적인 내용을 요약해 주세요.\n"
              "요약문은 텍스트의 핵심적인 내용을 잘 설명해야 합니다.\n"
              "요약문의 시작은 [ 로, 끝은 ]로 제시하세요."
            )
            modelPrompt = "### Instruction(명령어):\n{}\n\n### Input(입력):\n{}\n\n### Response(응답):".format(prompt, input.text)
        elif input.task == 'Sentiment analysis':
            input.task = 'sa'
            modelPrompt = input.text
        elif input.task == 'Keyword Extraction':
            input.task = 'keyword'
            prompt = "아래 텍스트에 대해 가장 핵심이 되는 key noun phrase 2~4개를 추출해줘. ( 포맷 조건: 각 key phrase는 • 로 시작하며, 개행기호로 구분되어야 합니다.)"
            modelPrompt = "### Instruction(명령어):\n{}\n\n### Input(입력):\n{}\n\n### Response(응답):".format(prompt, input.text)

        result = await self.svc.generateText(input.model, input.task, modelPrompt)

        if input.task == 'summary':
            results = await self.svc.summaryFormatter(result)
        elif input.task == 'sa':
            results = result[0]['label']
        else:
            results = await self.svc.formatter(result)

        return results

    @router.post('/sum/file')
    async def inference_file(self, file: UploadFile, model:str, task:str):

        if input.task == 'summary':
            prompt = ("아래의 텍스트는 요약이 필요한 긴 텍스트입니다.\n"
              "이 텍스트에서 긍정적이거나 부정적인 이유를 중심으로 핵심적인 내용을 요약해 주세요.\n"
              "요약문은 텍스트의 핵심적인 내용을 잘 설명해야 합니다.\n"
              "요약문의 시작은 [ 로, 끝은 ]로 제시하세요."
            )
        elif input.task == 'Sentiment analysis':
            input.task = 'sa'
        elif input.task == 'Keyword Extraction':
            input.task = 'keyword'
            prompt = "아래 텍스트에 대해 가장 핵심이 되는 key noun phrase 2~4개를 추출해줘. ( 포맷 조건: 각 key phrase는 • 로 시작하며, 개행기호로 구분되어야 합니다.)"

        # 업로드한 파일 전처리하기

        contents = file.file.read()
        data = BytesIO(contents)

        df = pd.read_excel(data, header=1)

        data.close()
        file.file.close()

        df.dropna(axis=0, how="any", subset=df.columns[:2])
        df = df.fillna("")

        rowNumColName = df.columns[0]
        inputColName = df.columns[1]

        for i in range(len(df.columns), 3):
            df['empty' + str(i)] = ''

        df.columns = [rowNumColName, inputColName, 'modelOutput:' + model]
        outputColName = df.columns[2]

        inputs = []
        for index, row in df.iterrows():
            if task=='summary':
                modelPrompt = row[inputColName]
            else:
                modelPrompt = "### Instruction(명령어):\n{}\n\n### Input(입력):\n{}\n\n### Response(응답):".format(prompt, row[inputColName])
            inputs.append([index, modelPrompt])

        for idx, data in enumerate(inputs):
            prompt = data[1]
            prompt = prompt.replace('\n', '')

            result = await self.svc.generateText(input.model, input.task, modelPrompt)

            if input.task == 'summary':
                results = await self.svc.summaryFormatter(result)
            elif input.task == 'sa':
                results = result[0]['label']
            else:
                results = await self.svc.formatter(result)
                
            df.loc[data[0], outputColName] = results

        df.to_excel('data/' + file.filename)    

        return FileResponse('data/' + file.filename)