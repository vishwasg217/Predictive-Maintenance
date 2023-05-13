import json
import pandas as pd
import numpy as np
import streamlit as st
import requests
# from config.config import logger

app_url = "http://localhost:8000"
endpoint = "/predict"
model_url = f"{app_url}{endpoint}"

st.title('Predictive Maintenance')

st.write('Please enter the following parameters')

sample_data = ['Low', 1410.0,65.70,191.00,25.75,35.85, 'Machine Failure', 'Power Failure']

st.write('Sample input: ')
st.write('Type: ',sample_data[0], 'RPM: ',sample_data[1],'Torque: ',sample_data[2],'Tool Wear: ',sample_data[3],'Air Temperature: ',sample_data[4],'Process Temperature: ',sample_data[5])
st.write('Sample Output: ')
st.write('Machine Failure? ', sample_data[6])
st.write('Type of Machine Failure: ', sample_data[7])

st.divider()

type_of_machine = st.selectbox('Type', ['Low', 'Medium', 'High'])

type_of_machine = str(type_of_machine)

rpm = st.number_input('RPM: ')
st.write('The current value of RPM is', rpm)
torque = st.number_input('Torque: ')    
st.write('The current value of Torque is', torque)
tool_wear = st.number_input('Tool Wear: ')
st.write('The current value of Tool Wear is', tool_wear)
air_temp = st.number_input('Air Temperature: ')
st.write('The current value of Air Temperature is', air_temp)
process_temp = st.number_input('Process Temperature: ')
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
        st.write('Machine Failure?  ', response_json.get('Machine Failure? '))
        st.write('Type of Failure: ', response_json.get('Type of Failure: ')) 
    else:
        st.write('Error:', response.status_code, response.text)
         
        


