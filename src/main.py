from fastapi import FastAPI
from api.v1.auth.routers import router as auth_router
from api.v1.site.routers import router as site_router
from api.v1.neural.routers import router as neural_router
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title='GAN',
    version="0.1.0"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(site_router)
app.include_router(neural_router)


if __name__ == "__main__":
    uvicorn.run('main:app', port=8000, host="0.0.0.0")
