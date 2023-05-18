import uvicorn
from fastapi import FastAPI
from backend.model import User
from pathlib import Path

from src.predict import prediction
from src.metrics import metrics
from src.eda import get_eda_obj
print('after import')
appl = FastAPI()



@appl.get("/")
def index():
    return {"message": "Hello, World"}

@appl.get('/Welcome')
def get_name(name: str):
    return {"Welcome": f"{name}"}

@appl.post('/predict')
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

@appl.get('/metrics')
def get_metrics():
    scores1, report1, best_model_name1, scores2, report2,best_model_name2 = metrics()
    return {"Model 1 Scores": scores1,
            "Model 1 Report": report1,
            "Best Model Name 1": best_model_name1,
            "Model 2 Scores": scores2,
            "Model 2 Report": report2,
            "Best Model Name 2": best_model_name2}

@appl.get('/eda')
def get_eda():
    eda_json = get_eda_obj()
    return eda_json



if __name__ == "__main__":
    uvicorn.run(appl, host="0.0.0.0", port=8000)

