import utils.config as cfg
Config = cfg.Config
from src.data_loader import load_raw_data, preprocess_datetime, load_and_merge_weather
from src.preprocessor import Preprocessor
from models.gru_model import build_model
from src.feature_builder import add_features, create_sequences
from tensorflow.keras.callbacks import EarlyStopping
import os

def train_pipeline():
    # 1. Pipeline nạp và tích hợp dữ liệu hệ thống
    df = load_raw_data()
    df = preprocess_datetime(df)
    df_combined = load_and_merge_weather(df)
    
    pre = Preprocessor()
    df_clean = pre.clean_data(df_combined)
    df_features = add_features(df_clean)    
    
    # 2. Thực thi chuẩn hóa ma trận đặc trưng nâng cao và chia tập dữ liệu
    scaled_data = pre.scale_data(df_features, is_training=True)
    train_data, test_data = pre.split_data(scaled_data)
    
    # 3. Đóng gói ma trận chuỗi thời gian 3D dạng Sliding Window
    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)
    
    # 4. Khởi tạo kiến trúc mạng mạng Neural Network Seq2Seq GRU
    # Định dạng Tensor đầu vào: (WINDOW_SIZE=24, NUM_FEATURES=11)
    input_shape = (X_train.shape[1], X_train.shape[2]) 
    model = build_model(input_shape)

    # 5. Thiết lập cơ chế dừng sớm (Early Stopping) kiểm soát hiện tượng quá khớp (Overfitting)
    early_stop = EarlyStopping(
        monitor='val_loss',  
        patience=Config.PATIENCE,
        restore_best_weights=True
    )
    
    print(f"🚀 Bắt đầu huấn luyện mạng Seq2Seq GRU Đa biến cho phân vùng: {Config.ZONE}...")
    model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 6. Đóng gói lưu trữ mô hình tối ưu
    if not os.path.exists(os.path.dirname(Config.MODEL_SAVE_PATH)):
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH))
    model.save(Config.MODEL_SAVE_PATH)
    print(f"✅ Mô hình tối ưu đã được lưu thành công tại: {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_pipeline()