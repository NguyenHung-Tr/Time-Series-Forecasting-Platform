from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from utils.config import Config

def build_gru_model(input_shape):
    """
    Xây dựng kiến trúc mạng GRU
    input_shape: (window_size, num_features)
    """
    model = Sequential([
        # Lớp GRU đầu tiên
        # return_sequences=True nếu bạn muốn chồng thêm 1 lớp GRU nữa bên dưới
        GRU(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2), # Chống Overfitting (học vẹt)
        
        # Lớp GRU thứ hai
        GRU(units=50, return_sequences=False),
        Dropout(0.2),
        
        # Lớp Dense (Đầu ra) - Trả về 1 con số MW duy nhất
        Dense(units=1)
    ])
    
    # Cấu hình bộ tối ưu và hàm mất mát
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='mean_squared_error' # MSE: Thước đo chuẩn cho bài toán hồi quy (Regression)
    )
    
    return model