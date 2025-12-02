from fastapi import FastAPI
from modules import modules_router


app = FastAPI()

app.include_router(
    modules_router,
    prefix='/knn',
    tags=['K nearest neighbors']
)