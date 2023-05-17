import streamlit as st
from annotated_text import annotated_text
import requests
import pandas as pd
import numpy as np
import json

app_url = "http://fastapi:8000"
endpoint = "/metrics"
model_url = f"{app_url}{endpoint}"

st.set_page_config(page_title="Performance Measures",layout="wide")

st.title('Performance Measures')

response = requests.get(model_url)

response_json = response.json()

col1, col2 = st.columns(2)

with col1:
    model1_scores = json.loads(response_json["Model 1 Scores"])
    model1_scores = pd.DataFrame(model1_scores)
    st.subheader("MODEL 1: Machine Failure or not")
    st.write('Model Scores')
    st.dataframe(model1_scores)

    best_model_name1 = response_json["Best Model Name 1"]
    annotated_text((best_model_name1, "Best Model Name"))

    model1_report = json.loads(response_json["Model 1 Report"])
    model1_report = pd.DataFrame(model1_report)
    st.write('Best Model Report')
    st.dataframe(model1_report)


with col2:
    model2_scores = json.loads(response_json["Model 2 Scores"])
    model2_scores = pd.DataFrame(model2_scores)
    st.subheader("MODEL 2: Type of Machine Failure")
    st.write('Model Scores')
    st.dataframe(model2_scores)

    best_model_name2 = response_json["Best Model Name 2"]
    annotated_text(( best_model_name2, "Best Model Name"))

    model2_report = json.loads(response_json["Model 2 Report"])
    model2_report = pd.DataFrame(model2_report)
    st.write('Best Model Report')
    st.dataframe(model2_report)




