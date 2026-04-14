import pandas as pd
from utils.config import Config

def load_raw_data(file_path=Config.RAW_DATA_PATH):
    """Đọc dữ liệu thô từ CSV"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Đã nạp dữ liệu từ: {file_path}")
        return df
    except Exception as e:
        print(f"❌ Lỗi nạp dữ liệu: {e}")
        return None

def preprocess_datetime(df, time_col='Datetime'):
    """Xử lý định dạng thời gian và sắp xếp"""
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    df = df.set_index(time_col)
    return df