import os

class ConfigMeta(type):
    @property
    def RAW_DATA_PATH(cls):
        return os.path.join(cls.BASE_DIR, 'data', 'raw', cls.ZONE, f'{cls.ZONE}_hourly.csv')

    @property
    def WEATHER_DATA_PATH(cls):
        return os.path.join(cls.BASE_DIR, 'data', 'raw', cls.ZONE, 'weather_hourly.csv')

    @property
    def TARGET_COL(cls):
        return f'{cls.ZONE}_MW'

    @property
    def TARGET_FEATURES(cls):
        return [f'{cls.ZONE}_MW', 'MW_diff']

    @property
    def FEATURES(cls):
        return [
            f'{cls.ZONE}_MW',
            'MW_diff',
            'temperature',
            'humidity',
            'hour_sin',
            'hour_cos',
            'day_sin',
            'day_cos',
            'is_holiday',
            'rolling_mean_24h',
            'lag_24h'
        ]

class Config(metaclass=ConfigMeta):
    # 1. Đường dẫn gốc của dự án
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. Cấu hình Vùng dữ liệu động (Có thể đổi thành 'COMED' hoặc 'DAYTON')
    ZONE = 'DAYTON'
    
    # 3. Đường dẫn dữ liệu đầu ra sau xử lý và lưu mô hình
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_data.csv')
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'seq2seq_model.h5')

    # 4. Tham số cấu trúc chuỗi Thời gian (Joint Multi-step)
    WINDOW_SIZE = 24           
    HORIZON = 24               
    
    # 5. Tham số kiểm soát huấn luyện
    TEST_SPLIT = 0.2
    PATIENCE = 5
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001