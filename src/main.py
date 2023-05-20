import warnings
from pathlib import Path
import typer
import pickle
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import config
from config.config import ARTIFACTS_DIR
from data import (
    convert_to_celsius,
    create_target,
    feature_scaling,
    ordinal_encoding,
    sampling,
)
from train import model1, model2
from src.eda import (
    setup,
    question_one,
    question_two,
    question_three,
    question_four,
    question_five,
    question_six
)

warnings.filterwarnings("ignore")
app = typer.Typer()

def get_data():
    df = pd.read_csv("data/raw/data.csv")
    return df

def eda(df):
    df = setup(df)
    q1 = question_one(df)
    q2 = question_two(df)
    q3 = question_three(df)
    q4 = question_four(df)
    q5 = question_five(df)

    json_obj = {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q5": q5
    }

    with open(Path(ARTIFACTS_DIR, "eda.json"), "w+") as f:
        json.dump(json_obj, f)
    
# @app.command()
def preprocess():
    df = pd.read_csv(Path(config.DATA_DIR, "raw/data.csv"))
    df = create_target(df)
    df = convert_to_celsius(df)
    df = ordinal_encoding(df)
    df = feature_scaling(df)
    df = sampling(df)
    df.to_csv(Path(config.DATA_DIR, "processed/preprocessed.csv"), index=False)
    return df

# @app.command()
def split_data():
    df = pd.read_csv(Path(config.DATA_DIR, "processed/preprocessed.csv"))
    target = 'type_of_failure'
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    train_data.to_csv(Path(config.DATA_DIR, "processed/train.csv"), index=False)
    test_data.to_csv(Path(config.DATA_DIR, "processed/test.csv"), index=False)


# @app.command()
def train():
    df = pd.read_csv(Path(config.DATA_DIR, "processed/train.csv"))
    scores_df, best_model, best_model_name, report = model1(df)
    print("Scores")
    print(scores_df)
    print("Best model")
    print(best_model)
    print("Classification report")
    print(report)
    with open(Path(ARTIFACTS_DIR, "model1.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    scores_df = scores_df.to_json()
    report = report.to_json()
    model_metrics = [scores_df, report, best_model_name]
    with open(Path(ARTIFACTS_DIR, "model1_metrics.json"), "w+") as f:
        json.dump(model_metrics, f)


    scores_df, best_model, best_model_name, report = model2(df)
    print("Scores")
    print(scores_df)
    print("Best model")
    print(best_model)
    print("Classification report")
    print(report)

    with open(Path(ARTIFACTS_DIR, "model2.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    scores_df = scores_df.to_json()
    report = report.to_json()
    model_metrics = [scores_df, report, best_model_name]
    with open(Path(ARTIFACTS_DIR, "model2_metrics.json"), "w+") as f:
        json.dump(model_metrics, f)

# @app.command()
# def generate_reports():
#     data_report()
#     model1_report()
#     model2_report()
    

# if __name__ == "__main__":
#     app()


# get_data()
eda(get_data())
# df = preprocess()
# split_data()
# print(df)
# train()
