from pydantic import BaseModel

class User(BaseModel):
    type: str
    rpm: float
    torque: float
    tool_wear: float
    air_temp: float
    process_temp: float