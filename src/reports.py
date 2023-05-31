import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import shutil
import os

from sklearn.model_selection import train_test_split

from evidently import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset

from config.config import logger, REPORTS_DIR, ARTIFACTS_DIR, DATA_DIR


def data_report():
    ref_data = pd.read_csv(Path(DATA_DIR, "processed/train.csv"))
    cur_data = pd.read_csv(Path(DATA_DIR, "processed/test.csv"))

    classification_performance_report = Report(metrics=[
    DataDriftPreset(), DataQualityPreset()
    ])

    classification_performance_report.run(reference_data=ref_data, current_data=cur_data, column_mapping = None)
    classification_performance_report.save_html(Path(REPORTS_DIR, 'data_report.html'))

    

def model1_report():
    ref_data = pd.read_csv(Path(DATA_DIR, "processed/train.csv"))
    cur_data = pd.read_csv(Path(DATA_DIR, "processed/test.csv"))

    with open("artifacts/model1.pkl", "rb") as f:
        model = pickle.load(f)

    ref_X = ref_data.drop(["Machine failure", "type_of_failure"], axis=1)
    ref_y = ref_data["Machine failure"]

    cur_X = cur_data.drop(["Machine failure", "type_of_failure"], axis=1)
    cur_y = cur_data["Machine failure"]
    ref_X_train, ref_X_test, ref_y_train, ref_y_test = train_test_split(ref_X, ref_y, test_size=0.2, random_state=42)

    ref_pred = model.predict(ref_X_test)
    ref_pred = pd.DataFrame(ref_pred, columns=["Prediction"])
    cur_pred = model.predict(cur_X)
    cur_pred = pd.DataFrame(cur_pred, columns=["Prediction"])

    ref_X_test.reset_index(inplace=True, drop=True)
    ref_y_test.reset_index(inplace=True, drop=True)
    ref_merged = pd.concat([ref_X_test, ref_y_test], axis=1)
    ref_merged = pd.concat([ref_merged, ref_pred], axis=1)
    ref_merged

    cur_X.reset_index(inplace=True, drop=True)
    cur_y.reset_index(inplace=True, drop=True)
    cur_merged = pd.concat([cur_X, cur_y], axis=1)
    cur_merged = pd.concat([cur_merged, cur_pred], axis=1)
    cur_merged

    cm = ColumnMapping()
    cm.target = "Machine failure"
    cm.prediction = "Prediction"
    # cm.target_names = list(["No Failure", "Machine Failure"])

    classification_performance_report = Report(metrics=[
    ClassificationPreset()
    ])

    classification_performance_report.run(reference_data=ref_merged, current_data=cur_merged, column_mapping = cm)
    classification_performance_report.save_html(Path(REPORTS_DIR, 'model_1_report.html'))

def model2_report():
    ref_data = pd.read_csv(Path(DATA_DIR, "processed/train.csv"))
    cur_data = pd.read_csv(Path(DATA_DIR, "processed/test.csv"))

    with open("artifacts/model2.pkl", "rb") as f:
        model = pickle.load(f)

    ref_X = ref_data.drop(["Machine failure", "type_of_failure"], axis=1)
    ref_y = ref_data["type_of_failure"]

    cur_X = cur_data.drop(["Machine failure", "type_of_failure"], axis=1)
    cur_y = cur_data["type_of_failure"]
    ref_X_train, ref_X_test, ref_y_train, ref_y_test = train_test_split(ref_X, ref_y, test_size=0.2, random_state=42)

    ref_pred = model.predict(ref_X_test)
    ref_pred = pd.DataFrame(ref_pred, columns=["Prediction"])
    cur_pred = model.predict(cur_X)
    cur_pred = pd.DataFrame(cur_pred, columns=["Prediction"])

    ref_X_test.reset_index(inplace=True, drop=True)
    ref_y_test.reset_index(inplace=True, drop=True)
    ref_merged = pd.concat([ref_X_test, ref_y_test], axis=1)
    ref_merged = pd.concat([ref_merged, ref_pred], axis=1)
    ref_merged

    cur_X.reset_index(inplace=True, drop=True)
    cur_y.reset_index(inplace=True, drop=True)
    cur_merged = pd.concat([cur_X, cur_y], axis=1)
    cur_merged = pd.concat([cur_merged, cur_pred], axis=1)
    cur_merged

    cm = ColumnMapping()
    cm.target = "type_of_failure"
    cm.prediction = "Prediction"
    # cm.target_names = list(["No Failure", "Machine Failure"])

    classification_performance_report = Report(metrics=[
    ClassificationPreset()
    ])

    classification_performance_report.run(reference_data=ref_merged, current_data=cur_merged, column_mapping = cm)
    classification_performance_report.save_html(Path(REPORTS_DIR, 'model_2_report.html'))


def generate_reports():
    data_report()
    model1_report()
    model2_report()

    # dir = 'frontend/reports'
    # for f in os.listdir(dir):
    #     os.remove(os.path.join(dir, f))
    src = Path(REPORTS_DIR)
    dst = 'frontend/reports/'
    shutil.rmtree(dst)
    shutil.copytree(src, dst)

generate_reports()