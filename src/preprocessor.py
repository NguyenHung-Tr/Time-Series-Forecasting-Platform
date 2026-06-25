import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from utils.config import Config

class Preprocessor:
    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.scaler_dir = os.path.join(Config.BASE_DIR, 'models', 'saved_models')
        if not os.path.exists(self.scaler_dir):
            os.makedirs(self.scaler_dir)

    def clean_data(self, df):
        df = df.dropna()
        df = df[~df.index.duplicated(keep='first')]
        return df

    def scale_data(self, df, is_training=True):
        feature_cols = Config.FEATURES
        target_cols = Config.TARGET_FEATURES  # Chuyển đổi sang đa biến mục tiêu động

        # Cấu hình đường dẫn lưu Scaler định danh riêng biệt theo từng vùng miền (ZONE)
        feat_scaler_path = os.path.join(self.scaler_dir, f'{Config.ZONE}_feature_scaler.pkl')
        targ_scaler_path = os.path.join(self.scaler_dir, f'{Config.ZONE}_target_scaler.pkl')

        if is_training:
            scaled_features = self.feature_scaler.fit_transform(df[feature_cols])
            self.target_scaler.fit(df[target_cols]) 
            
            joblib.dump(self.feature_scaler, feat_scaler_path)
            joblib.dump(self.target_scaler, targ_scaler_path)
            return scaled_features
        else:
            # Nạp bộ chuẩn hóa tương ứng của phân vùng đang cấu hình
            if os.path.exists(feat_scaler_path) and os.path.exists(targ_scaler_path):
                self.feature_scaler = joblib.load(feat_scaler_path)
                self.target_scaler = joblib.load(targ_scaler_path)
            else:
                raise FileNotFoundError(f"❌ Không tìm thấy bộ scaler cho vùng {Config.ZONE}. Vui lòng chạy train trước!")
            
            if df.empty:
                return None
                
            return self.feature_scaler.transform(df[feature_cols])

    def split_data(self, scaled_data):
        train_size = int(len(scaled_data) * (1 - Config.TEST_SPLIT))
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        print(f"✅ Chia dữ liệu thành công: Train({len(train_data)}), Test({len(test_data)})")
        return train_data, test_data