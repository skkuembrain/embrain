from fastapi import FastAPI
from model import model

app = FastAPI()

# including routers
app.include_router(model.router)

@app.get('/')
async def mainPage():
    result = 'Hello, World :)'
    return result
