import holidays
import numpy as np
import pandas as pd
from utils.config import Config

def add_features(df):
   
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    us_holidays = holidays.US()
    df['holiday'] = df.index.map(lambda x: 1 if x in us_holidays else 0)
    df['rolling_mean_24h'] = df[Config.TARGET_COL].rolling(window=24).mean()
    df['lag_24h'] = df[Config.TARGET_COL].shift(24)
    df = df.dropna() # Loại bỏ các dòng có giá trị NaN do rolling và shift tạo ra
    return df

def create_sequences(data, window_size=Config.WINDOW_SIZE):
    """
    Biến mảng 1 chiều thành mảng 3 chiều (Samples, Time_steps, Features)
    để nạp vào mạng GRU/LSTM.
    """
    X = []
    y = []

    # Chạy vòng lặp để cắt dữ liệu thành từng ô cửa sổ
    for i in range(window_size, len(data)):
        # Lấy window_size dòng trước đó làm đầu vào (X)
        X.append(data[i-window_size:i, :])
        # Lấy giá trị hiện tại làm mục tiêu dự báo (y)
        y.append(data[i, 0])

    # Chuyển về định dạng mảng Numpy
    X, y = np.array(X), np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], -1))
    
    print(f"✅ Tạo chuỗi thành công: X shape {X.shape}, y shape {y.shape}")
    return X, y