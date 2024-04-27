from fastapi import FastAPI
from api.v1.auth.routers import router as auth_router
from api.v1.site.routers import router as site_router
from api.v1.neural.routers import router as neural_router
import uvicorn

app = FastAPI(
    title='GAN',
    version="0.1.0"
)

app.include_router(auth_router)
app.include_router(site_router)
app.include_router(neural_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
