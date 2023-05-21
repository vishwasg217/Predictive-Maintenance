import json
import pandas as pd
import numpy as np
import streamlit as st
import requests
# from config.config import logger

app_url = "http://fastapi:8000"
endpoint = "/predict"
model_url = f"{app_url}{endpoint}"

st.set_page_config(page_title="Prediction",layout="wide")

st.title('Predictive Maintenance')

st.write('Please enter the following parameters')

sample_data = ['Low', 1410.0,65.70,191.00,25.75,35.85, 'Machine Failure', 'Power Failure']

st.divider()

type_of_machine = st.selectbox('Type', ['Low', 'Medium', 'High'])

type_of_machine = str(type_of_machine)

rpm = st.number_input('RPM: ', value=sample_data[1])
st.write('The current value of RPM is', rpm)
torque = st.number_input('Torque: ', value=sample_data[2])    
st.write('The current value of Torque is', torque)
tool_wear = st.number_input('Tool Wear: ', value=sample_data[3])
st.write('The current value of Tool Wear is', tool_wear)
air_temp = st.number_input('Air Temperature: ', value=sample_data[4])
st.write('The current value of Air Temperature is', air_temp)
process_temp = st.number_input('Process Temperature: ', value=sample_data[5])
st.write('The current value of Process Temperature is', process_temp)


data = {
    "type": type_of_machine,
    "rpm": rpm,
    "torque": torque,
    "tool_wear": tool_wear,
    "air_temp": air_temp,
    "process_temp": process_temp
}

    

if st.button("Predict"):
    response = requests.post(model_url, json=data)
    if response.ok:
        response_json = response.json()
        st.write()
        st.write('Machine Failure?  ', response_json.get('Machine Failure? '))
        st.write('Type of Failure: ', response_json.get('Type of Failure: ')) 
    else:
        st.write('Error:', response.status_code, response.text)
         
        


