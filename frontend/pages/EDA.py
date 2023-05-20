import streamlit as st
from annotated_text import annotated_text
import requests
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objs as go

app_url = "http://fastapi:8000"
endpoint = "/eda"
model_url = f"{app_url}{endpoint}"

st.title('Exploratory Data Analysis')

response = requests.get(model_url)
response_json = response.json()

st.header('Question 1')
st.write("What is the distribution of the 'machine failure' label in the dataset? How many instances have failed and how many have not failed?")
q1 = json.loads(response_json["q1"])
q1_fig = go.Figure(q1)
st.plotly_chart(q1_fig)
st.markdown("""
            > **The success rate of the machine is 96.52% and the highest type of failure is HDF(Heat Dissipation Failure) with 1.15% failure rate.**
            """)
st.divider()

st.header('Question 2')
st.write("What is the distribution of the 'productID' variable in the dataset? How many instances are of low, medium, and high quality variants?")
q2 = json.loads(response_json["q2"])
q2_fig = go.Figure(q2)
st.plotly_chart(q2_fig)
st.markdown("""
> **low quality varient makes up majority of the dataset with 60% of the data, followed by medium quality varient with 30% and high quality varient with 10%.**""")
st.divider()

st.header('Question 3')
st.write("What is the range of values for the continuous variables 'air temperature', 'process temperature', 'rotational speed', 'torque', and 'tool wear'? Are there any outliers in the dataset?")
q3 = response_json["q3"]
q3_fig1 = json.loads(q3['fig1'])
q3_fig2 = json.loads(q3['fig2'])

q3_fig1 = go.Figure(q3_fig1)
q3_fig2 = go.Figure(q3_fig2)
st.plotly_chart(q3_fig1)
st.plotly_chart(q3_fig2)
st.markdown("""
            > **Rotational speed may or may not be actual outliers, therefore we'll keep them in the dataset for now.**
            """)
st.divider()

st.header('Question 4')
st.write("Is there any correlation between the continuous variables and the 'machine failure' label? For example, does the tool wear increase the likelihood of machine failure?")
q4 = response_json["q4"]
q4_fig = json.loads(q4['heatmap'])
q4_fig = go.Figure(q4_fig)
st.plotly_chart(q4_fig)
st.markdown("""
            > **Null Hypothesis: There is no signifcant relationship between the different columns and Machine Failure.**

            > **Alternate Hypothesis: There is a significant relationship between the tool wear and the machine failure label.**
""")
q4_df = json.loads(q4['ttest'])
q4_df = pd.DataFrame(q4_df)
st.dataframe(q4_df)

st.header('Question 5')
st.write("Is there any correlation between the categorical variable 'productID' and the continuous variables? For example, is the 'rotational speed' higher for high-quality products than for low-quality products?")

q5 = json.loads(response_json["q5"])
q5_fig = go.Figure(q5)
st.plotly_chart(q5_fig)
st.markdown("""
            > **Process Temperature seems to have an effect on high quality variant machines. Therefore we can say that Process Temperature is correlated with machine type.**
            """)
st.divider()

# st.write(response_json)






