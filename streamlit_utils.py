import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from joblib import load




def feature_engineering(df):
    # Historique des prix
    N = 10
    for i in range(1, N+1):
        df['Open_t-'+ str(i)] = df['Open'].shift(i)

    # Moyenne mobile des prix (5 jours, 10, 20)
    MA = [5, 10, 20]

    for i in MA:
        df['MA(' + str(i) + ')'] = df['Open'].rolling(i).mean()


    # ROI - Return on Investment
    R = [1, 5, 10, 20]

    for i in R:
        df['R_t(' + str(i) + ')'] = df['Open'].pct_change(i)

    # Volatilités à 5, 10 et 20 jours
    V = [5, 10, 20]

    for i in V : 
        df['V_t(' + str(i) + ')'] = df['R_t(1)'].rolling(i).std()

    # Historique Volume 
    N = 10
    for i in range(1, N+1):
        df['Volume_t-'+ str(i)] = df['Volume'].shift(i)

    # Variable Cible
    df['target'] = df['V_t(5)'].shift(-5)
    df['target'] = df['target'].apply(lambda x : 1 if x >= 0.015 else 0)

    df = df.dropna()
    
    return df


def drop_columns(df):
    columns = ['High', 'Low', 'Close', 'Volume']
    
    return df.drop(columns, axis = 1)



def make_pipeline():
    col_dropper = FunctionTransformer(drop_columns)
    scaler = load("mon_scaler.joblib")

    model = load("mon_model.joblib")
    
    pipeline = Pipeline(
        steps = [
            ("drop_columns", col_dropper), # On supprime les colonnes
            ("scaling", scaler),           # Normalisation
            ("model", model)               # Prediction
        ]
    )
    
    return pipeline