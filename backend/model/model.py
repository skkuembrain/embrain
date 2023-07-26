from fastapi import APIRouter, File, FileUpload

router = APIRouter(
	prefix="/model",
    tags=["models"]
)

@router.post('/opencoding')
async def inference_text(text:str):
    result = 'inference test'
    return result


@router.post('/sumsa')
async def summary_text(text:str):
    result = 'summary test'
    return result