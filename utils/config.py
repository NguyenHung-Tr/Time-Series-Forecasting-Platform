import os

class Config:
    # 1. Đường dẫn gốc
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. Đường dẫn dữ liệu
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'AEP_hourly.csv')
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_data.csv')
    
    # 3. Đường dẫn lưu mô hình
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'seq2seq_model.h5')

    # 4. Tham số mô hình 
    TARGET_COL = 'AEP_MW'

    FEATURES = [
        'AEP_MW',          
        'hour_sin', 'hour_cos', 
        'day_sin', 'day_cos', 
        'is_holiday', 
        'rolling_mean_24h', 
        'lag_24h'          
]
    WINDOW_SIZE = 24           
    HORIZON = 24               
    
    TEST_SPLIT = 0.2
    PATIENCE = 5
    # 5. Tham số huấn luyện
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001