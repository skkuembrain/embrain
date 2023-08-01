from fastapi import APIRouter, File, UploadFile, Depends
from ..service import TextGenerator
import pandas as pd
import xlsxwriter
from pydantic import BaseModel


class TextInferenceInput(BaseModel):
    text: str
    pos: bool
    model: str


router = APIRouter(
	prefix="/oc",
    tags=["opencoding"]
)

class Opencoding:
    svc: TextGenerator = Depends()

    @router.post('/text')
    async def inference_text(self, input: TextInferenceInput):

        if input.pos:
            prompt = '다음 텍스트는 긍정적인 리뷰이다. 다음 텍스트에 대해서 <속성, 의견> 형태로 의견을 추출해줘.'
        else:
            prompt = '다음 텍스트는 부정적인 리뷰이다. 다음 텍스트에 대해서 <속성, 의견> 형태로 의견을 추출해줘.'
        
        modelPrompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
            "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### Instruction(명령어):\n{}\n\n### Input(입력):\n{}\n\n### Response(응답):".format(prompt, input.text)
        )

        result = await self.svc.generate(input.model, modelPrompt)
        results = await self.svc.formatter(result)

        return results


    @router.post('/file')
    async def inference_file(file: UploadFile, pos: bool, model: str):

        if pos:
            prompt = '다음 텍스트는 긍정적인 리뷰이다. 다음 텍스트에 대해서 <속성, 의견> 형태로 의견을 추출해줘.'
        else:
            prompt = '다음 텍스트는 부정적인 리뷰이다. 다음 텍스트에 대해서 <속성, 의견> 형태로 의견을 추출해줘.'

        # 업로드한 파일 전처리하기
        df = pd.read_excel(file, header=1)
        df.dropna(axis=0, how="any", subset=df.columns[:2])
        df = df.fillna("")

        rowNumColName = df.columns[0]
        inputColName = df.columns[1]

        for i in range(len(df.columns), 5):
            df['empty' + str(i)] = ''

        df.columns = [rowNumColName, inputColName, 'output1', 'output2', 'output3']

        input = []
        for index, row in df.iterrows():
            modelPrompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
                "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{}\n\n### Input(입력):\n{}\n\n### Response(응답):".format(prompt, row[inputColName])
            )
            input.append([index, modelPrompt])

        print(df.columns)

        for index, prompt in input:
            df.loc[index, 'output1'] = prompt

        # TODO: connect model code

        return file.filename

    @router.post('/sumsa')
    async def summary_text(text:str):
        result = 'summary test'
        return result