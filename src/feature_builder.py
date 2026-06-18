import holidays
import numpy as np
import pandas as pd
from utils.config import Config

def add_features(df):
    """
    Trích xuất đặc trưng chu kỳ, lịch lễ và các biến kỹ thuật nâng cao.
    Hỗ trợ tính toán động biến mục tiêu phụ tải và tốc độ thay đổi phụ tải (MW_diff).
    """
    df = df.copy()
    target_col = Config.TARGET_COL
    
    # 1. Tính toán Tốc độ thay đổi phụ tải điện (Sai phân bậc 1)
    df['MW_diff'] = df[target_col].diff()
    
    # 2. Trích xuất đặc trưng chu kỳ thời gian phẳng (Sin/Cos)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # 3. Tích hợp lịch nghỉ lễ quốc gia Mỹ nhằm giải thích các biến động tải sâu
    us_holidays = holidays.US()
    df['is_holiday'] = df.index.map(lambda x: 1 if x in us_holidays else 0)
    
    # 4. Xây dựng các đặc trưng trễ và trung bình trượt dựa trên cột phụ tải động của vùng
    df['rolling_mean_24h'] = df[target_col].rolling(window=24).mean()
    df['lag_24h'] = df[target_col].shift(24)
    
    # Lọc bỏ các dòng dữ liệu bị khuyết (NaN) do cơ chế dịch cửa sổ diff/shift/rolling
    return df.dropna()

def create_sequences(data, window_size=Config.WINDOW_SIZE, horizon=Config.HORIZON):
    """
    Đóng gói ma trận dữ liệu mảng phẳng thành các tập dữ liệu chuỗi tensor 3D.
    Đầu ra nhãn y được trích xuất động theo số lượng biến cấu hình trong Config.TARGET_FEATURES.
    
    Shape đầu ra mong muốn:
        X: (Samples, WINDOW_SIZE, len(Config.FEATURES))
        y: (Samples, HORIZON, len(Config.TARGET_FEATURES)) -> (Samples, 24, 2)
    """
    X = []
    y = []
    
    # Xác định động vị trí cột chỉ mục của các biến mục tiêu đầu ra nằm trong mảng FEATURES
    target_indices = [Config.FEATURES.index(feature) for feature in Config.TARGET_FEATURES]
    
    for i in range(window_size, len(data) - horizon + 1):
        # Trích xuất toàn bộ các đặc trưng trong quá khứ làm đầu vào X
        X.append(data[i-window_size:i, :])
        # Trích xuất đồng thời đa biến mục tiêu tương lai (Công suất & Tốc độ thay đổi) làm nhãn y
        y.append(data[i:i+horizon, target_indices])
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], -1))
    
    return X, y