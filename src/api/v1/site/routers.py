from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
# from ..neural.utils import *
from ..neural.routers import root
import subprocess

router = APIRouter(prefix="/api/v1/site", tags=["site"])

@router.post('/recognize')
async def recognize(file: UploadFile):
    content = await file.read()
    with open(f'../media/{file.filename}', 'wb') as f:
        f.write(content)

    # runpy.run_path(f'E:/VSCode/code_schema_GAN/src/api/v1/neural/utils.py')
    subprocess.run(["python", "./api/v1/neural/utils.py"])
    
    with open('./api/v1/neural/scores.txt', 'r') as f:
        command = f.read()
    # # route = await root(filename=file.filename)
    # try:
    #     command = recognize_file(filename=file.filename)
    #     print(command)
    # except:
    #     command = 'command'
    res = JSONResponse(content={'command': command})
    print(res.body)
    return res
