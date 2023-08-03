from fastapi import APIRouter, File, UploadFile, Depends
from service import OpencodingGenerator
from fastapi_restful.inferring_router import InferringRouter
from fastapi_utils.cbv import cbv
import pandas as pd
import xlsxwriter
from pydantic import BaseModel
from io import BytesIO
from fastapi.responses import FileResponse


class TextInferenceInput(BaseModel):
    text: str
    pos: bool
    model: str

'''
router = APIRouter(
	prefix="/oc",
    tags=["opencoding"
'''

router = InferringRouter()

@cbv(router)
class Opencoding:
    # svc: TextGenerator = Depends(TextGenerator)
    print('**** setting Opencoding models ****')
    svc = OpencodingGenerator()

    @router.get('/oc/hello')
    async def hello(self):
        return 'hello'

    @router.post('/oc/text')
    async def inference_text(self, input: TextInferenceInput):

        input.text = input.text.replace('\n', '')

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

        result = await self.svc.generateText(input.model, modelPrompt)
        print(result)
        result = await self.svc.formatter(result[0])
        result = ' '.join(s for s in result)
        return result


    @router.post('/oc/file')
    async def inference_file(self, file: UploadFile, model:str, pos:str):

        if pos == 'True':
            prompt = '다음 텍스트는 긍정적인 리뷰이다. 다음 텍스트에 대해서 <속성, 의견> 형태로 의견을 추출해줘.'
        else:
            prompt = '다음 텍스트는 부정적인 리뷰이다. 다음 텍스트에 대해서 <속성, 의견> 형태로 의견을 추출해줘.'

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

        for i in range(len(df.columns), 5):
            df['empty' + str(i)] = ''

        df.columns = [rowNumColName, inputColName, 'output1', 'output2', 'output3']

        inputs = []
        for index, row in df.iterrows():
            modelPrompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
                "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{}\n\n### Input(입력):\n{}\n\n### Response(응답):".format(prompt, row[inputColName])
            )
            inputs.append([index, modelPrompt])

        for idx, data in enumerate(inputs):
            prompt = data[1]
            result = await self.svc.generateText(model, prompt)
            df.loc[data[0], 'output1'] = result

        df.to_excel('data/' + file.filename)    

        return FileResponse('data/' + file.filename)