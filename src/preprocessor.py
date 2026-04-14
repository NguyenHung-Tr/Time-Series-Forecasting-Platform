import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.config import Config

class Preprocessor:
    def __init__(self):
        # Khởi tạo scaler để đưa dữ liệu về khoảng [0, 1]
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def clean_data(self, df):
        """Xử lý giá trị trống và trùng lặp"""
        # Xóa bỏ các dòng bị NaN (nếu có)
        df = df.dropna()
        # Loại bỏ các dòng trùng mốc thời gian
        df = df[~df.index.duplicated(keep='first')]
        print("✅ Dữ liệu đã được làm sạch (NaN & Duplicates).")
        return df

    def scale_data(self, df):
        """Chuẩn hóa dữ liệu MW về khoảng [0, 1]"""
        # GRU/LSTM yêu cầu đầu vào phải được chuẩn hóa để dễ hội tụ
        data = df[[Config.TARGET_COL]].values
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data

    def split_data(self, scaled_data):
        """Chia dữ liệu thành tập Train và Test"""
        train_size = int(len(scaled_data) * (1 - Config.TEST_SPLIT))
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        print(f"✅ Chia dữ liệu thành công: Train({len(train_data)}), Test({len(test_data)})")
        return train_data, test_data