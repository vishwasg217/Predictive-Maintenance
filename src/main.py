import pandas as pd
from pathlib import Path
import warnings

from config import config
from data import create_target, convert_to_celsius, ordinal_encoding, feature_scaling, sampling
from train import model1, model2

warnings.filterwarnings('ignore')

def get_data():
    df = pd.read_csv('data/raw/data.csv')
    return df

def preprocess():
    df = pd.read_csv(Path(config.DATA_DIR, 'raw/data.csv'))
    df = create_target(df)
    df = convert_to_celsius(df)
    df = ordinal_encoding(df)
    df = feature_scaling(df)
    df = sampling(df)
    df.to_csv(Path(config.DATA_DIR, 'processed/preprocessed.csv'), index=False)
    return df
    
def train(df):

    scores_df, best_model, report = model1(df)
    print('Scores')
    print(scores_df)
    print('Best model')
    print(best_model)
    print('Classification report')
    print(report)

    scores_df, best_model, report = model2(df)
    print('Scores')
    print(scores_df)
    print('Best model')
    print(best_model)
    print('Classification report')
    print(report)


# get_data()
df = preprocess()
print(df)
train(df)





