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
    scores_df1 = pd.read_json(scores1)
    report1 = model1_metrics[1]
    report_df1 = pd.read_json(report1)

    print("MODEL 1")
    print("---------------------------------------")
    print("Scores: ")
    print(scores_df1)
    print("Report: ")
    print(report_df1)

    print("---------------------------------------")
    print("---------------------------------------")


    with open(Path(ARTIFACTS_DIR,'model2_metrics.json'), 'rb') as f:
        model2_metrics = json.load(f)

    scores2 = model2_metrics[0]
    scores_df2 = pd.read_json(scores2)
    report2 = model2_metrics[1]
    report_df2 = pd.read_json(report2)

    print("MODEL 2")
    print("---------------------------------------")
    print("Scores: ")
    print(scores_df2)
    print("Report: ")
    print(report_df2)
    print(type(scores_df2))

metrics()