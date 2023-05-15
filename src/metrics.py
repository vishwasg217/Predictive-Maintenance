import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json
from config.config import ARTIFACTS_DIR
import warnings
warnings.filterwarnings("ignore")

def metrics():
    with open(Path(ARTIFACTS_DIR,'model1_metrics.json'), 'rb') as f:
        model1_metrics = json.load(f)

    scores1 = model1_metrics[0]
    report1 = model1_metrics[1]
    best_model_name1 = model1_metrics[2]


    with open(Path(ARTIFACTS_DIR,'model2_metrics.json'), 'rb') as f:
        model2_metrics = json.load(f)

    scores2 = model2_metrics[0]
    report2 = model2_metrics[1]
    best_model_name2 = model2_metrics[2]
    

    return scores1, report1,best_model_name1, scores2, report2, best_model_name2

    
scores1, report1,best_model_name1, scores2, report2,best_model_name2 = metrics()
print(scores2)
print(report2)
print(best_model_name2)