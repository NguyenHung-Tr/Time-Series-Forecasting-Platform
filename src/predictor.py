import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils.config import Config
from src.preprocessor import Preprocessor
from src.data_loader import load_raw_data, preprocess_datetime
from src.feature_builder import add_features

class Predictor:
    def __init__(self):
        print("📂 Đang nạp mô hình đa biến và cấu hình thang đo...")
        self.model = load_model(Config.MODEL_SAVE_PATH)
        self.pre = Preprocessor()
        
        # Fit lại scaler với đầy đủ các đặc trưng
        raw_df = load_raw_data()
        clean_df = self.pre.clean_data(preprocess_datetime(raw_df))
        df_with_features = add_features(clean_df)
        self.pre.scale_data(df_with_features) 

    def predict_next_hours(self, last_window_data, n_steps=24):
        predictions = []
        num_features = len(Config.FEATURES)
        current_batch = last_window_data.reshape((1, Config.WINDOW_SIZE, num_features))

        for i in range(n_steps):
            # 1. Dự báo và lấy giá trị số thực duy nhất bằng .flatten()[0]
            res = self.model.predict(current_batch, verbose=0)
            current_pred = float(res.flatten()[0]) 
            predictions.append(current_pred)
            
            # 2. Lấy dòng cuối cùng của cửa sổ hiện tại
            last_row = current_batch[:, -1:, :].copy()
            
            # 3. Cập nhật giá trị MW mới (vị trí 0)
            # Dùng index cụ thể để tránh lỗi "setting an array element with a sequence"
            last_row[0, 0, 0] = current_pred 
            
            # 4. (Tùy chọn nâng cao) Cập nhật lại đặc trưng 'hour' cho dòng tiếp theo
            # Nếu Hưng muốn chính xác hơn, có thể cộng thêm 1 giờ vào đặc trưng 'hour'
            # current_hour = last_row[0, 0, 1] 
            # last_row[0, 0, 1] = (current_hour + 1) % 24
            
            # 5. Đẩy cửa sổ trượt
            current_batch = np.append(current_batch[:, 1:, :], last_row, axis=1)

        # Nghịch đảo thang đo để về đơn vị MW
        dummy_array = np.zeros((len(predictions), num_features))
        dummy_array[:, 0] = predictions
        inverse_params = self.pre.scaler.inverse_transform(dummy_array)
        
        return inverse_params[:, 0].reshape(-1, 1)