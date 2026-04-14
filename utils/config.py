import os

class Config:
    # 1. Đường dẫn gốc
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. Đường dẫn dữ liệu
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'AEP_hourly.csv')
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_data.csv')
    
    # 3. Đường dẫn lưu mô hình
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'best_gru_model.h5')

    # 4. Tham số mô hình (Để sau này Hưng mở rộng đa biến chỉ cần sửa ở đây)
    TARGET_COL = 'AEP_MW'      # Tên cột công suất trong file AEP
    WINDOW_SIZE = 24           # Dùng 24 tiếng quá khứ
    HORIZON = 1                # Dự báo 1 tiếng tương lai
    
    TEST_SPLIT = 0.2
    PATIENCE = 5
    # 5. Tham số huấn luyện
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001