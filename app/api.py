import uvicorn
from fastapi import FastAPI
from app.model import User
from pathlib import Path
import pickle

from config.config import ARTIFACTS_DIR
from src.predict import prediction

app = FastAPI()


@app.get("/")
def index():
    return {"message": "Hello, World"}

@app.get('/Welcome')
def get_name(name: str):
    return {"Welcome": f"{name}"}

@app.post('/predict')
def predict(data: User):
    received = data.dict()
    type = received['type']
    rpm = received['rpm']
    torque = received['torque']
    tool_wear = received['tool_wear']
    air_temp = received['air_temp']
    process_temp = received['process_temp']

    result1, result2 = prediction(type, rpm, torque, tool_wear, air_temp, process_temp)

    return {"Machine Failure? ": result1,
            "Type of Failure: ": result2}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

