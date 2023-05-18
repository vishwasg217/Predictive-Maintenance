from config.config import ARTIFACTS_DIR
import pickle
from pathlib import Path

with open(Path(ARTIFACTS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

result = scaler.transform([[1410.0, 65.70, 191.00, 25.75, 35.85]])
rpm = result[0, 0]
print(rpm)
print(result)
print('hello')


# Type                          0.00
# Rotational speed [rpm]     1410.00
# Torque [Nm]                  65.70
# Tool wear [min]             191.00
# Machine failure               1.00
# type_of_failure               2.00
# Air temperature [c]          25.75
# Process temperature [c]      35.85
