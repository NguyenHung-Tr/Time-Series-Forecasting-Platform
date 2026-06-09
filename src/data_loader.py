import pandas as pd
from utils.config import Config
import os

def load_raw_data(file_path=Config.RAW_DATA_PATH):
    """Đọc dữ liệu thô từ CSV"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Đã nạp dữ liệu từ: {file_path}")
        return df
    except Exception as e:
        print(f"❌ Lỗi nạp dữ liệu: {e}")
        return None

def preprocess_datetime(df, time_col='Datetime'):
    """Xử lý định dạng thời gian và sắp xếp"""
    if df is None:
        return None
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    df = df.set_index(time_col)
    return df

def load_and_merge_weather(df_energy, weather_filepath=Config.WEATHER_DATA_PATH):
    """
    Tải file thời tiết, xử lý cấu trúc và gộp đồng bộ (Inner Join) với dữ liệu điện năng.
    """
    if df_energy is None:
        print("❌ Dữ liệu điện năng đầu vào bị trống (None). Không thể gộp thời tiết.")
        return None

    # Hỗ trợ tìm file dự phòng ở thư mục gốc nếu không thấy ở thư mục data/raw/
    if not os.path.exists(weather_filepath):
        backup_path = os.path.basename(weather_filepath)  
        if os.path.exists(backup_path):
            weather_filepath = backup_path
        else:
            print(f"⚠️ Không tìm thấy file thời tiết tại {weather_filepath}. Hệ thống giữ nguyên dữ liệu cũ.")
            return df_energy
        
    try:
        # Đọc dữ liệu từ dòng thứ 4 để bỏ qua metadata định vị của trạm đo
        df_weather = pd.read_csv(weather_filepath, skiprows=3)
        print(f"✅ Đã nạp dữ liệu thời tiết từ: {weather_filepath}")
        
        # Định dạng lại thời gian và đưa lên làm Index
        df_weather['Datetime'] = pd.to_datetime(df_weather['time'])
        df_weather.set_index('Datetime', inplace=True)
        
        # Đổi tên các cột sang tiếng Anh ngắn gọn để khớp với Config.FEATURES
        df_weather = df_weather.rename(columns={
            'temperature_2m (°C)': 'temperature',
            'relative_humidity_2m (%)': 'humidity'
        })
        
        # Lọc lấy các cột cần thiết cho mô hình
        df_weather = df_weather[['temperature', 'humidity']]
        
        # Nội suy tuyến tính phòng trường hợp mất dữ liệu thời tiết ở một số giờ cá biệt
        df_weather = df_weather.interpolate(method='linear')
        
        # Thực hiện INNER JOIN: Tự động giữ lại các khung giờ trùng khớp và xóa bỏ dòng thừa
        df_merged = df_energy.join(df_weather, how='inner')
        print(f"✅ Tích hợp thời tiết thành công! Kích thước dữ liệu sau gộp: {df_merged.shape}")
        
        return df_merged

    except Exception as e:
        print(f"❌ Lỗi trong quá trình xử lý và gộp dữ liệu thời tiết: {e}")
        return df_energy