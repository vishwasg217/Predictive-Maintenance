import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from config.config import MODEL_DIR

def prediction(type, rpm, torque, tool_wear, air_temp, process_temp):
    with open(Path(MODEL_DIR,'model1.pkl'), 'rb') as f:
        model1 = pickle.load(f)

    with open(Path(MODEL_DIR,'model2.pkl'), 'rb') as f:
        model2 = pickle.load(f)

    if type == 'Low':
        type = int(0)
    elif type == 'Medium':
        type = int(1)
    elif type == 'High':
        type = int(2)

    type = float(type)

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

# prediction('M', 0.175738,0.477421,0.823187,0.363062,0.352309)