from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/auth", tags=["/auth"])

@router.get("/")
async def root():
    return {"message": "Hello World"}