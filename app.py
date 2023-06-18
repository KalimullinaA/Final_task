from fastapi import FastAPI, UploadFile
import joblib
import pandas as pd
import os.path
# from sklearn.metrics import accuracy

app = FastAPI()

@app.get('/')
def root():
    return {'message': 'MLOps'}

@app.post("/file")
def upload_file(file: UploadFile):
    df = pd.read_csv('data-ver-control/data/prepared/'+file.filename)
    X = df[['filename']]
    y = df[['label']]

    # model = joblib.load(open('data-ver-control/model/model.joblib', 'rb'))
    model = joblib.load('data-ver-control/model/model.joblib')
    # accuracy(model.predict(X), y)
    model.predict(X)
    name, endsw = os.path.splitext(file.filename)

    return {'accuracy': model.predict(X)}