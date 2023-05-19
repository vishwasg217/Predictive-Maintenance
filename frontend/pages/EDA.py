import streamlit as st
from annotated_text import annotated_text
import requests
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objs as go

from src.eda import setup

app_url = "http://fastapi:8000"
endpoint = "/eda"
model_url = f"{app_url}{endpoint}"

df = pd.read_csv("data/raw/data.csv")

st.title('Exploratory Data Analysis')

df = setup(df)

st.dataframe(df)

response = requests.get(model_url)
response_json = response.json()

st.header('Question 1')

q1 = json.loads(response_json["q1"])
q1_fig = go.Figure(q1)
st.plotly_chart(q1_fig)

st.header('Question 2')
q2 = json.loads(response_json["q2"])
q2_fig = go.Figure(q2)
st.plotly_chart(q2_fig)

st.header('Question 3')
# q3 = response_json["q3"]
# q3_fig1 = json.loads(q3['fig1'])
# q3_fig2 = json.loads(q3['fig2'])

# q3_fig1 = go.Figure(q3_fig1)
# q3_fig2 = go.Figure(q3_fig2)
# st.plotly_chart(q3_fig1)
# st.plotly_chart(q3_fig2)

st.header('Question 4')
q4 = response_json["q4"]
q4_fig = json.loads(q4['heatmap'])
q4_fig = go.Figure(q4_fig)
st.plotly_chart(q4_fig)

q4_df = json.loads(q4['ttest'])
q4_df = pd.DataFrame(q4_df)
st.dataframe(q4_df)

st.header('Question 5')
q5 = json.loads(response_json["q5"])
q5_fig = go.Figure(q5)
st.plotly_chart(q5_fig)

# st.write(response_json)






