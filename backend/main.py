import uvicorn
from fastapi import FastAPI
from backend.model import User


from src.predict import prediction
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


if __name__ == "__main__":
    uvicorn.run(appl, host="0.0.0.0", port=8000)

