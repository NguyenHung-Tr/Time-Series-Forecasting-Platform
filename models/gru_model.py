from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, RepeatVector, TimeDistributed 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from utils.config import Config


def build_model(input_shape):
    model = Sequential()
    
    # 1. ENCODER: Đọc và nén bối cảnh quá khứ
    model.add(GRU(64, activation='relu', input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    
    # 2. BRIDGE: Lặp lại trạng thái nén đúng bằng số bước cần dự báo (HORIZON = 24)
    model.add(RepeatVector(Config.HORIZON))
    
    # 3. DECODER: Bung ma trận trạng thái ra chuỗi thời gian tương lai
    model.add(GRU(64, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    
    # 4. OUTPUT LAYER: Sinh ra 1 con số dự báo cho mỗi bước thời gian
    model.add(TimeDistributed(Dense(1)))

    optimizer=Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

    model.summary()
    return model