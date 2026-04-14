import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils.config import Config
from src.preprocessor import Preprocessor
from src.data_loader import load_raw_data, preprocess_datetime

class Predictor:
    def __init__(self):
        print("📂 Đ đang nạp mô hình và cấu hình thang đo...")
        self.model = load_model(Config.MODEL_SAVE_PATH)
        self.pre = Preprocessor()
        
        # --- BƯỚC QUAN TRỌNG: Fit lại scaler ---
        # Để inverse_transform được, scaler phải biết Max/Min của dữ liệu gốc
        raw_df = load_raw_data()
        clean_df = self.pre.clean_data(preprocess_datetime(raw_df))
        self.pre.scale_data(clean_df) # Hàm này sẽ gọi self.scaler.fit() ngầm bên trong
        # ---------------------------------------

    def predict_next_hours(self, last_window_data, n_steps=24):
        predictions = []
        # Đảm bảo dữ liệu đầu vào có dạng [1, window_size, 1]
        current_batch = last_window_data.reshape((1, Config.WINDOW_SIZE, 1))

        for i in range(n_steps):
            current_pred = self.model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred)
            
            # Cập nhật cửa sổ trượt
            current_pred_reshaped = current_pred.reshape((1, 1, 1))
            current_batch = np.append(current_batch[:, 1:, :], current_pred_reshaped, axis=1)

        # Bây giờ scaler đã được "fit", lệnh này sẽ chạy ngon lành
        predictions_mw = self.pre.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions_mw