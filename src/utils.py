import pandas as pd
import numpy as np


def get_columns():
    df = pd.read_csv('data/raw/data.csv')
    return df.columns

def get_num_columns():
    return ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
            'Tool wear [min]']
