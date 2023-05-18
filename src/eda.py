import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
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
    fig = px.histogram(df, x='type_of_failure')
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    logger.info("EDA Question 1 complete")
    return plot_json

def question_two(df):
    fig = px.histogram(df, x='Type')
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    logger.info("EDA Question 2 complete")
    return plot_json

def question_three(df):

    num_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    fig1 = go.Figure()

    for i, col in enumerate(num_cols):
        fig1.add_trace(go.Box(x=df[col], name=col))

    fig1.update_layout(
        title="Distribution of Numerical Features",
        height=1200,
        width=800,
        xaxis=dict(title="Feature"),
        yaxis=dict(title="Value"),
        showlegend=False
    )

    plot_json1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    outlier_cols = ['Torque [Nm]', 'Rotational speed [rpm]']

    fig2 = px.histogram(df, x=outlier_cols, nbins=50, marginal='box', opacity=0.7)

    fig2.update_layout(
        title='Distribution of Torque and Rotational speed',
        xaxis_title='Values',
        yaxis_title='Frequency'
    )

    plot_json2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    logger.info("EDA Question 3 complete")
    return {'fig1': plot_json1, 'fig2': plot_json2}

def question_four(df):
    corr_matrix = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']].corr()

    fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title='Correlation')
                    ))

    fig.update_layout(
        title='Correlation Matrix',
        xaxis=dict(title='Variables'),
        yaxis=dict(title='Variables')
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
    fig = px.violin(df, y=num_cols, x='Type', box=True, points="all", hover_data=df.columns)
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


