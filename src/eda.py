import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.stats import ttest_ind
import plotly
import json


from config.config import logger
from config.config import ARTIFACTS_DIR

warnings.filterwarnings("ignore")


def setup(df: pd.DataFrame) -> pd.DataFrame:
    def type_of_failure(row_name):
        if df.loc[row_name, 'TWF'] == 1:
            df.loc[row_name, 'type_of_failure'] = 'TWF'
        elif df.loc[row_name, 'HDF'] == 1:
            df.loc[row_name, 'type_of_failure'] = 'HDF'
        elif df.loc[row_name, 'PWF'] == 1:
            df.loc[row_name, 'type_of_failure'] = 'PWF'
        elif df.loc[row_name, 'OSF'] == 1:
            df.loc[row_name, 'type_of_failure'] = 'OSF'
        elif df.loc[row_name, 'RNF'] == 1:
            df.loc[row_name, 'type_of_failure'] = 'RNF'

    df.apply(lambda row: type_of_failure(row.name), axis=1)

    df['type_of_failure'].replace(np.NaN, 'no failure', inplace=True)
    df.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)
    logger.info("Created type_of_failure column")
    return df

def question_one(df):
    category_counts = df['type_of_failure'].value_counts()
    total_samples = len(df)
    category_percentages = (category_counts / total_samples) * 100
    categories = list(category_percentages.index)
    percentage_labels = list(category_percentages)
    percentage_labels = [f'{num:.2f}%' for num in percentage_labels]
    fig = px.histogram(df, x='type_of_failure', category_orders={'type_of_failure': categories})
    fig.update_traces(text=percentage_labels, textposition='auto')
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    logger.info("EDA Question 1 complete")
    return plot_json

def question_two(df):
    category_counts = df['Type'].value_counts()
    total_samples = len(df)
    category_percentages = (category_counts / total_samples) * 100
    categories = list(category_percentages.index)
    percentage_labels = list(category_percentages)
    percentage_labels = [f'{num:.2f}%' for num in percentage_labels]
    fig = px.histogram(df, x='Type', category_orders={'Type': categories})
    fig.update_traces(text=percentage_labels, textposition='auto')
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    logger.info("EDA Question 2 complete")
    return plot_json

def question_three(df):

    num_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    fig1 = make_subplots(rows=5, cols=1, subplot_titles=num_cols, vertical_spacing=0.04)


    for i, col in enumerate(num_cols):
        box_plot = go.Box(x=df[col], name=col)
        fig1.add_trace(box_plot, row=i+1, col=1)

    fig1.update_layout(
        title="Distribution of Numerical Features",
        height=1200,
        width=900,
        title_text="Box plots"
    )

    plot_json1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    outlier_cols = ['Torque [Nm]', 'Rotational speed [rpm]']

    fig2 = make_subplots(rows=1, cols=2, subplot_titles=outlier_cols, vertical_spacing=0.03)

    for i, col in enumerate(outlier_cols):
        box_plot = go.Histogram(x=df[col], name=col)
        fig2.add_trace(box_plot, row=1, col=i+1)

    # fig2 = px.histogram(df, x=outlier_cols, nbins=50, marginal='box', opacity=0.7)

    fig2.update_layout(
        title='Distribution of Torque and Rotational speed',
        yaxis_title='Frequency',
        title_text="Histograms",
        width=900
    )

    plot_json2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    logger.info("EDA Question 3 complete")
    return {'fig1': plot_json1, 'fig2': plot_json2}

def question_four(df):
    corr_matrix = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']].corr()

    fig = px.imshow(corr_matrix, zmin=-1, zmax=1, text_auto=True)

    fig.update_layout(
        title='Correlation Matrix',
        height=600,
        width=800
    )
    
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    test_cols = ['Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]','Tool wear [min]']
    values = []
    for col in test_cols:
        failed = df[df['Machine failure'] == 1][col]
        non_failed = df[df['Machine failure'] == 0][col]

        t, p = ttest_ind(failed, non_failed)
        values.append([t, p])

    values = pd.DataFrame(values, columns=['test-statistic', 'p-value'], index=test_cols)
    alpha = 0.05
    values['Hypothesis'] = values['p-value'].apply(lambda p: 'Reject null hypothesis' if p < alpha else 'Accept null hypothesis')
    value = values.to_json()
    logger.info("EDA Question 4 complete")
    return {'heatmap': plot_json, 'ttest': value}

def question_five(df):
    num_cols = ['Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]','Tool wear [min]']
    fig = make_subplots(rows=5, cols=1, subplot_titles=num_cols, vertical_spacing=0.03, horizontal_spacing=0.01)

    for i, col in enumerate(num_cols):
        violin_trace = go.Violin(x=df['Type'], y=df[col], name=col, box_visible=True, meanline_visible=True)
        fig.add_trace(violin_trace, row=i+1, col=1)
    fig.update_layout(height=2000, width=800, title_text="Subplots")
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    logger.info("EDA Question 5 complete")
    return plot_json

def question_six(df):
    num_cols = df[['Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]','Tool wear [min]']]
    sns.pairplot(num_cols)
    fig = go.Figure(data=pio.to_plotly(sns))  
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    logger.info("EDA Question 6 complete")
    return plot_json

def get_eda_obj():
    with open(Path(ARTIFACTS_DIR, 'eda.json'), 'r') as f:
        eda_json = json.load(f)

    return eda_json


