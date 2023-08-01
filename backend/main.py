from fastapi import FastAPI
from router import opencoding, summary

app = FastAPI()

# including routers
app.include_router(opencoding.router)
app.include_router(summary.router)

@app.get('/')
async def mainPage():
    result = 'Hello, World :)'
    return result
