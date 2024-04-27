from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/api/v1/neural", tags=["neural"])


@router.get("/")
async def root(filename: str):
    # async with open('/Users/timi__di/PycharmProjects/code_schema_GAN/media/' + filename, 'rb') as f:
    #     # TODO neural recognize
    #     pass
    return {'filename': filename}
