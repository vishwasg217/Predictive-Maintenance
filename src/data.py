import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
)

from config.config import logger
from config.config import ARTIFACTS_DIR

warnings.filterwarnings("ignore")

df = pd.read_csv("data/raw/data.csv")


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    def type_of_failure(row_name):
        if df.loc[row_name, "TWF"] == 1:
            df.loc[row_name, "type_of_failure"] = "TWF"
        elif df.loc[row_name, "HDF"] == 1:
            df.loc[row_name, "type_of_failure"] = "HDF"
        elif df.loc[row_name, "PWF"] == 1:
            df.loc[row_name, "type_of_failure"] = "PWF"
        elif df.loc[row_name, "OSF"] == 1:
            df.loc[row_name, "type_of_failure"] = "OSF"
        elif df.loc[row_name, "RNF"] == 1:
            df.loc[row_name, "type_of_failure"] = "RNF"

    df.apply(lambda row: type_of_failure(row.name), axis=1)
    df["type_of_failure"].replace(np.NaN, "no failure", inplace=True)
    df.drop(["TWF", "HDF", "PWF", "OSF", "RNF"], axis=1, inplace=True)
    encoder = LabelEncoder()
    df["type_of_failure"] = encoder.fit_transform(df["type_of_failure"])
    logger.info("Target variable created")
    return df


def convert_to_celsius(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(["UDI", "Product ID"], axis=1, inplace=True)
    df["Air temperature [c]"] = df["Air temperature [K]"] - 273.15
    df["Process temperature [c]"] = df["Process temperature [K]"] - 273.15
    df.drop(["Air temperature [K]", "Process temperature [K]"], axis=1, inplace=True)
    logger.info("Temperature converted to celsius")
    return df


def ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    encoder = OrdinalEncoder(categories=[["L", "M", "H"]])
    df["Type"] = encoder.fit_transform(df[["Type"]])
    logger.info("Type encoded")
    return df


def feature_scaling(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scale_cols = [
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Air temperature [c]",
        "Process temperature [c]",
    ]
    df_scaled = scaler.fit_transform(df[scale_cols])

    with open(Path(ARTIFACTS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)


    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = scale_cols

    df.drop(scale_cols, axis=1, inplace=True)

    df_scaled = pd.concat([df, df_scaled], axis=1)
    logger.info("Features scaled")
    return df_scaled


def sampling(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(["type_of_failure"], axis=1)
    y = df["type_of_failure"]
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    sampled_df = pd.concat([X, y], axis=1)
    logger.info("Data sampled")
    return sampled_df


target_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
celsius_cols = ["UDI", "Product ID", "Air temperature [K]", "Process temperature [K]"]
categorical_cols = ["Type"]

feature_transformer = ColumnTransformer(
    transformers=[
        ("create_target", FunctionTransformer(create_target), target_cols),
        ("convert_to_celsius", FunctionTransformer(convert_to_celsius), celsius_cols),
        ("ordinal_encoding", FunctionTransformer(ordinal_encoding), ["Type"]),
    ],
    remainder="passthrough",
)

scaling_transformer = ColumnTransformer(
    transformers=[("feature_scaling", MinMaxScaler(), [1, 2, 4, 5, 6])], remainder="passthrough"
)


def preprocess(df):
    pipeline = Pipeline(
        steps=[("transformer", feature_transformer), ("scaling_transformer", scaling_transformer)]
    )

    result = pipeline.fit_transform(df)
    result = pd.DataFrame(result)

    X = result.drop(result.columns[5], axis=1)
    y = result[5]

    smote = SMOTE(sampling_strategy="auto")
    X_resampled, y_resampled = smote.fit_resample(X, y)
    result = pd.concat([X_resampled, y_resampled], axis=1)
    result.to_csv("data/processed/proc_data.csv")
