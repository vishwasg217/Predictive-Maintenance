import json
import pandas as pd
import numpy as np
import streamlit as st
import requests

app_url = "http://localhost:8000"
endpoint = "/predict"
model_url = f"{app_url}{endpoint}"

st.title('Predictive Maintenance')

st.write('Please enter the following parameters')

st.write('Sample input: ')
st.write('Low', 'RPM',1410,'Torque',65.70,'Tool Wear',191.00,'Air Temperature',25.75,'Process Temperature',35.85)
st.write('Sample Output: ')
st.write('Machine Failure? Machine Failure')
st.write('Type of Machine Failure: Power Failure')

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
json_data = json.dumps(data)

def get_prediction():
    response = requests.post(model_url, json=data)
    response = json.loads(response.text)
    st.write('Machine Failure?  ', response['Machine Failure? '])
    st.write('Type of Failure: ', response['Type of Failure: '])    

if st.button("Predict"):
    get_prediction()
        


