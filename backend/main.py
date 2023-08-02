from fastapi import FastAPI
from router import opencoding, summary
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# including routers
app.include_router(opencoding.router)
app.include_router(summary.router)

@app.get('/')
async def mainPage():
    result = 'Hello, World :)'
    return result
