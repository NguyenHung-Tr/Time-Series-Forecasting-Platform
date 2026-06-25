import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils.config import Config
from src.preprocessor import Preprocessor
from src.data_loader import load_raw_data

class Predictor:
    def __init__(self):
        # TUÂN THỦ NGUYÊN TẮC: Sử dụng compile=False để tránh lỗi giải tuần tự hóa metric trên Keras 3
        self.model = load_model(Config.MODEL_SAVE_PATH, compile=False)
        self.pre = Preprocessor()
        
        # Gọi hàm mồi để tự động nạp các tệp Scaler .pkl thích ứng với vùng tương ứng vào bộ nhớ RAM
        self.pre.scale_data(pd.DataFrame(), is_training=False)

    def predict_next_hours(self, last_window_data, n_steps=24, start_time=None):
        num_features = len(Config.FEATURES)
        num_targets = len(Config.TARGET_FEATURES)
        
        # Chuẩn bị Tensor đầu vào bám theo cấu trúc 3D: (Batch_size, Window_size, Features)
        current_batch = last_window_data.reshape((1, Config.WINDOW_SIZE, num_features))
        
        # Thực hiện dự báo đồng thời 24 bước song song (Joint Multi-step Forecasting)
        res = self.model.predict(current_batch, verbose=0)
        
        # SỬA LỖI CỐT LÕI: Định hình lại cấu trúc từ dạng chuỗi 3D (1, 24, 2) về dạng ma trận 2D phẳng (24, 2)
        # Tuyệt đối không dùng reshape(-1, 1) vì sẽ làm sai lệch cấu trúc thành 48 mẫu đơn lẻ
        predictions = res.reshape(Config.HORIZON, num_targets) 
        
        # Tiến hành giải nén ma trận đồng thời cho toàn bộ các biến mục tiêu đầu ra về đơn vị gốc
        predictions_inverse = self.pre.target_scaler.inverse_transform(predictions)

        return predictions_inverse