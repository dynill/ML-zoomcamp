import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/spotify-2023.csv'
MODEL_PATH = 'model.bin'

AUDIO = ['bpm', 'danceability', 'valence', 'energy', 'acousticness',
         'instrumentalness', 'liveness', 'speechiness']
FEATURES = AUDIO + ['key', 'mode', 'artist_count', 'released_month', 'released_year']

XGB_PARAMS = {
    'eta': 0.1,
    'max_depth': 2,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 1,
    'verbosity': 0,
}
NUM_ROUNDS = 25


def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')

    df.columns = (
        df.columns.str.lower()
        .str.replace('_%', '', regex=False)
        .str.replace('(s)', '', regex=False)
    )

    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    df = df.dropna(subset=['streams']).reset_index(drop=True)
    df['streams'] = df['streams'].astype(int)

    df['key'] = df['key'].fillna('Unknown')
    df = df.drop_duplicates(subset=['track_name', 'artist_name']).reset_index(drop=True)

    threshold = df['streams'].quantile(0.75)
    df['mega_hit'] = (df['streams'] >= threshold).astype(int)

    df['released_month'] = df['released_month'].astype(str)

    return df


def train(df_train, y_train):
    dicts = df_train[FEATURES].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    feature_names = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_ROUNDS)

    return dv, model


def predict(df, dv, model):
    dicts = df[FEATURES].to_dict(orient='records')
    X = dv.transform(dicts)

    feature_names = list(dv.get_feature_names_out())
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)

    return model.predict(dmatrix)


if __name__ == '__main__':
    df = load_data(DATA_PATH)
    print(f'rows after cleaning: {len(df)}, mega-hit share: {df.mega_hit.mean():.3f}')

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train['mega_hit'].values
    y_test = df_test['mega_hit'].values

    dv, model = train(df_full_train, y_full_train)

    y_pred = predict(df_test, dv, model)
    auc = roc_auc_score(y_test, y_pred)
    print(f'test AUC: {auc:.4f}')

    with open(MODEL_PATH, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

    print(f'model saved to {MODEL_PATH}')