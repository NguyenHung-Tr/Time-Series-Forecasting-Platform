import numpy as np
from utils.config import Config

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
        X.append(data[i-window_size:i, 0])
        # Lấy giá trị hiện tại làm mục tiêu dự báo (y)
        y.append(data[i, 0])

    # Chuyển về định dạng mảng Numpy
    X, y = np.array(X), np.array(y)

    # Quan trọng: Reshape X về 3D [Samples, Time_steps, Features]
    # GRU yêu cầu đầu vào 3D. Ở đây features = 1 (chỉ có MW)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    print(f"✅ Tạo chuỗi thành công: X shape {X.shape}, y shape {y.shape}")
    return X, y