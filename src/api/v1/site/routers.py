from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from ..neural.routers import root

router = APIRouter(prefix="/api/v1/site", tags=["site"])

@router.get('/')
async def index():
    return {'message': 'Home page'}

@router.get('/support')
async def support():
    return {'message': 'Support page'}

@router.post('/recognize')
async def recognize(file: UploadFile):
    content = await file.read()
    with open(f'/Users/timi__di/PycharmProjects/code_schema_GAN/media/{file.filename}', 'wb') as f:
        f.write(content)
    route = await root(filename=file.filename)
    return {'route': route}
