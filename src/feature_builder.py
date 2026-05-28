import holidays
import numpy as np
import pandas as pd
from utils.config import Config

def add_features(df):
    df = df.copy()
    target_name = Config.TARGET_COL[0] if isinstance(Config.TARGET_COL, list) else Config.TARGET_COL
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    us_holidays = holidays.US()
    df['is_holiday'] = df.index.map(lambda x: 1 if x in us_holidays else 0)
    df['rolling_mean_24h'] = df[Config.TARGET_COL].rolling(window=24).mean()

    df['rolling_mean_24h'] = df[target_name].rolling(window=24).mean()
    df['lag_24h'] = df[target_name].shift(24)
    
    return df.dropna()

def create_sequences(data, window_size=Config.WINDOW_SIZE, horizon=Config.HORIZON):
    #(Samples, Time_steps, Features)
    X = []
    y = []
    for i in range(window_size, len(data) - horizon + 1):
        X.append(data[i-window_size:i, :])
        y.append(data[i:i+horizon, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], -1))
    
    return X, y