from fastapi import FastAPI
from pickle import load
from pandas import read_csv
import numpy as np
app=FastAPI()
@app.post("/cancer")
async def create_upload_file(filepath):
    cancer_df=read_csv(filepath)
    cancer=load("../models/cancer.pkl")
    prediction=np.array(cancer.predict(cancer_df))
    preds=np.where(prediction==1,"Yes","No")
    return preds
@app.post("/diabeties")
async def create_upload_file(filepath):
    cancer_df=read_csv(filepath)
    cancer=load("../models/diabeties.pkl")
    prediction=np.array(cancer.predict(cancer_df))
    preds=np.where(prediction==1,"Yes","No")
    return preds
@app.post("/heart")
async def create_upload_file(filepath):
    cancer_df=read_csv(filepath)
    cancer=load("../models/heart.pkl")
    prediction=np.array(cancer.predict(cancer_df))
    preds=np.where(prediction==1,"Yes","No")
    return preds
@app.post("/kidney")
async def create_upload_file(filepath):
    cancer_df=read_csv(filepath)
    cancer=load("../models/kidney.pkl")
    prediction=np.array(cancer.predict(cancer_df))
    preds=np.where(prediction==1,"Yes","No")
    return preds
@app.post("/liver")
async def create_upload_file(filepath):
    cancer_df=read_csv(filepath)
    cancer=load("../models/liver.pkl")
    prediction=np.array(cancer.predict(cancer_df))
    preds=np.where(prediction==1,"Yes","No")
    return preds
