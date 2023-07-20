from fastapi import FastAPI

app = FastAPI()

@app.get('/')
async def mainPage():
    result = 'Hello, World :)'
    return result