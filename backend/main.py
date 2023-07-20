from fastapi import fastapi

app = FastAPI()

@app.get('/')
async def hello():
    result = '안녕하세요'
    return result