import pandas as pd
from fastapi import FastAPI, status
from mediapi.models import dbModels, engine
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

from utils.helpers import Helpers

app = FastAPI()
dbModels.base.metadata.create_all(bind=engine)

app.add_middleware(CORSMiddleware,
    allow_origins=['http://0.0.0.0', 'http://0.0.0.0:8000',
                   'http://fedora', 'http://fedora:8000',
                   'https://emart.onrender.com', 'https://emart".onrender.com:443',
                   'http://objective-violet-87944.pktriot.net:22010', ],
    allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

from mediapi.components import DataIngestion, DataTransformation, ModelRecommender
# from mediapi.components.price_model import PriceModel
from mediapi.routes import recomm_router
from os import environ as env


@app.get("/", status_code=status.HTTP_200_OK)
async def index():
    return {"result": f"hello var env['MY_VAR']"}


@app.get("/test", status_code=status.HTTP_200_OK)
async def index():
    di = DataIngestion()
    trdfPath, progsPath, *_ = di.start()
    di.info()
    di.visuals()
    dt = DataTransformation(trdfPath, progsPath)
    X_tr, y_tr, X_te, y_te = dt.start()
    mrec = ModelRecommender(X_tr, y_tr, X_te, y_te)
    mrec.start()
    mrec.predict()
    return {"result": "hello test"}