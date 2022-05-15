from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder 
from schema import Data
import uvicorn
import joblib
import pandas as pd

model = joblib.load('src/model/modelo.gzip') 

app = FastAPI()

@app.get('/')
def hello() -> str:
    return 'hello world, go to /docs'

@app.post('/predict')
def predict (data: Data) -> dict:
    df = pd.DataFrame(jsonable_encoder(data), index = [0])
    prediction = model.predict(df)
    return {'prediction':prediction[0]}

def test_import():
    return 'import'

if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 8000)