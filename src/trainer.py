import utils.config as cfg
Config = cfg.Config
from src.data_loader import load_raw_data, preprocess_datetime
from src.preprocessor import Preprocessor
from src.feature_builder import create_sequences
from models.gru_model import build_gru_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def train_pipeline():
    # 1. Load & Clean
    df = load_raw_data()
    df = preprocess_datetime(df)
    
    pre = Preprocessor()
    df_clean = pre.clean_data(df)
    
    # 2. Scale & Split
    scaled_data = pre.scale_data(df_clean)
    train_data, test_data = pre.split_data(scaled_data)
    
    # 3. Create Sequences (Sliding Window)
    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)
    
    # 4. Build & Train Model
    input_shape = (X_train.shape[1], X_train.shape[2]) # (24, 1)
    model = build_gru_model(input_shape)

    # 5. Thiết lập cơ chế dừng sớm và lưu mô hình tốt nhất
    early_stop =EarlyStopping(
        monitor='val_loss',  
        patience=Config.PATIENCE,
        restore_best_weights=True
    )
    
    print("🚀 Bắt đầu huấn luyện mạng GRU...")
    history = model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 5. Save Model
    if not os.path.exists(os.path.dirname(Config.MODEL_SAVE_PATH)):
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH))
    model.save(Config.MODEL_SAVE_PATH)
    print(f"✅ Mô hình đã được lưu tại: {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_pipeline()