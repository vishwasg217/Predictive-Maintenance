import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from config.config import MODEL_DIR

def prediction(type, rpm, torque, tool_wear, air_temp, process_temp):
    with open(Path(MODEL_DIR,'model1.pkl'), 'rb') as f:
        model1 = pickle.load(f)

    if type == 'L':
        type = int(0)
    elif type == 'M':
        type = int(1)
    elif type == 'H':
        type = int(2)

    type = float(type)

    prediction = model1.predict([[type, rpm, torque, tool_wear, air_temp, process_temp]])

    if prediction[0] == 0:
        result = 'No Failure'
    elif prediction[0] == 1:
        result = 'Machine Failure'
    print(result)
    return str(result)

# prediction('M', 0.175738,0.477421,0.823187,0.363062,0.352309)