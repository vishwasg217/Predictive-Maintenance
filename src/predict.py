import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from config.config import ARTIFACTS_DIR, logger

def prediction(type, rpm, torque, tool_wear, air_temp, process_temp):
    with open(Path(ARTIFACTS_DIR,'model1.pkl'), 'rb') as f:
        model1 = pickle.load(f)

    with open(Path(ARTIFACTS_DIR,'model2.pkl'), 'rb') as f:
        model2 = pickle.load(f)

    # type preprocessing
    if type == 'Low':
        type = int(0)
    elif type == 'Medium':
        type = int(1)
    elif type == 'High':
        type = int(2)

    type = float(type)


    # min max scaler
    with open(Path(ARTIFACTS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    scaled_input = scaler.transform([[rpm, torque, tool_wear, air_temp, process_temp]])
    rpm, torque, tool_wear, air_temp, process_temp = scaled_input[0]

    # print(rpm, torque, tool_wear, air_temp, process_temp)

    prediction1 = model1.predict([[type, rpm, torque, tool_wear, air_temp, process_temp]])

    if prediction1[0] == 0:
        result1 = 'No Failure'
    elif prediction1[0] == 1:
        result1 = 'Machine Failure'
    
    prediction2 = model2.predict([[type, rpm, torque, tool_wear, air_temp, process_temp]])
    prediction2 = int(prediction2)

    encoding = {0: 'Heat Dissipation Failure',
                1: 'Overstrain Failure',
                2: 'Power Failure',
                3: 'Random Failure',
                4: 'Tool Wear Failure',
                5: 'No Failure'}
    
    result2 = encoding[prediction2]

    print(result1, result2)

    return result1, result2

prediction('Low', 1412,	52.3,	218,25.15,	34.95)

# Sample Inputs 

# 1412	52.3	218	1	1	25.15	34.95
# 'Low', 1410.0,65.70,191.00,25.75,35.85

# Type                          0.00
# Rotational speed [rpm]     1410.00
# Torque [Nm]                  65.70
# Tool wear [min]             191.00
# Machine failure               1.00
# type_of_failure               2.00
# Air temperature [c]          25.75
# Process temperature [c]      35.85

# {
#   "type": "Low",
#   "rpm": 1412,
#   "torque": 52.3,
#   "tool_wear": 218,
#   "air_temp": 25.15,
#   "process_temp": 34.95
# }