import warnings
from pathlib import Path
import typer
import pickle

import pandas as pd

from config import config
from config.config import MODEL_DIR
from data import (
    convert_to_celsius,
    create_target,
    feature_scaling,
    ordinal_encoding,
    sampling,
)
from train import model1, model2

warnings.filterwarnings("ignore")
app = typer.Typer()

def get_data():
    df = pd.read_csv("data/raw/data.csv")
    return df

@app.command()
def preprocess():
    df = pd.read_csv(Path(config.DATA_DIR, "raw/data.csv"))
    df = create_target(df)
    df = convert_to_celsius(df)
    df = ordinal_encoding(df)
    df = feature_scaling(df)
    df = sampling(df)
    df.to_csv(Path(config.DATA_DIR, "processed/preprocessed.csv"), index=False)
    return df

@app.command()
def train():
    df = pd.read_csv(Path(config.DATA_DIR, "processed/preprocessed.csv"))
    scores_df, best_model, report = model1(df)
    print("Scores")
    print(scores_df)
    print("Best model")
    print(best_model)
    print("Classification report")
    print(report)
    with open(Path(MODEL_DIR, "model1.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    scores_df, best_model, report = model2(df)
    print("Scores")
    print(scores_df)
    print("Best model")
    print(best_model)
    print("Classification report")
    print(report)

    with open(Path(MODEL_DIR, "model2.pkl"), "wb") as f:
        pickle.dump(best_model, f)


# get_data()
df = preprocess()
print(df)
train()
