from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, RepeatVector, TimeDistributed 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from utils.config import Config

def build_model(input_shape):
    """
    Khởi tạo cấu trúc mạng Seq2Seq GRU hỗ trợ cơ chế dự báo đồng thời đa biến đầu ra.
    Mô hình triệt tiêu hoàn toàn sai số tích lũy của cấu trúc đệ quy cũ.
    """
    model = Sequential()
    
    # 1. ENCODER: Tiếp nhận không gian đặc trưng đa biến đầu vào và nén thành vector ngữ cảnh
    model.add(GRU(64, activation='relu', input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    
    # 2. BRIDGE: Nhân bản vector ngữ cảnh tương thích với kích thước bước thời gian tương lai (HORIZON)
    model.add(RepeatVector(Config.HORIZON))
    
    # 3. DECODER: Giải mã chuỗi trạng thái, tái cấu trúc hình thái đồ thị chuỗi thời gian tương lai
    model.add(GRU(64, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    
    # 4. MULTIVARIATE OUTPUT LAYER: Dự báo song song nhiều biến mục tiêu cùng một lúc
    # Output Shape chuyển dịch từ dạng đơn biến (None, 24, 1) sang cấu trúc đa biến (None, 24, 2)
    num_targets = len(Config.TARGET_FEATURES)
    model.add(TimeDistributed(Dense(num_targets)))

    # Khởi tạo bộ tối ưu hóa và biên dịch mô hình theo chuẩn kiến trúc Keras 3 mới nhất
    optimizer = Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer, 
        loss=MeanSquaredError(), 
        metrics=[MeanAbsoluteError()]
    )

    model.summary()
    return model