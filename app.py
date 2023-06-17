from fastapi import FastAPI, UploadFile
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error

app = FastAPI()

@app.get('/')
def root():
    return {'message': 'MLOps'}

@app.post("/file")
def upload_file(file: UploadFile):

    df = pd.read_excel(file.file.read())
    X = df[['filename']]
    y = df[['label']]

    model = joblib.load('model.joblib')

    return mean_squared_error(model.predict(X), y)