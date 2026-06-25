import sys
import os
import argparse

# Đảm bảo hệ thống nhận diện đúng đường dẫn package nội bộ khi thực thi
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_builder import add_features
from src.trainer import train_pipeline
from src.predictor import Predictor
from src.evaluator import evaluate_model
from src.data_loader import load_raw_data, preprocess_datetime, load_and_merge_weather
from src.preprocessor import Preprocessor
from utils.config import Config

def main():
    parser = argparse.ArgumentParser(description="Hệ thống Dự báo Phụ tải Điện Đa vùng miền - Kiến trúc Seq2Seq")
    parser.add_argument('--mode', type=str, default='test', help='Chế độ vận hành: train hoặc test')
    args = parser.parse_args()

    if args.mode == 'train':
        print("🏗️  Đang khởi động quá trình huấn luyện hệ thống Seq2Seq Đa biến...")
        train_pipeline()
        
    elif args.mode == 'test':
        print(f"📊 Đang triển khai kiểm tra mô hình dự báo đa biến cho phân vùng: {Config.ZONE}...")
        
        # 1. Pipeline nạp và đồng bộ dữ liệu thô
        df_raw = load_raw_data()
        df_time = preprocess_datetime(df_raw)
        df_combined = load_and_merge_weather(df_time)

        # 2. Xử lý làm sạch và trích xuất ma trận đặc trưng nâng cao
        pre = Preprocessor()
        df_clean = pre.clean_data(df_combined)
        df_features = add_features(df_clean)
        scaled_data = pre.scale_data(df_features, is_training=False)
        
        # 3. Tách chuỗi thực tế (y_true) và cửa sổ quá khứ đầu vào (input_window) theo tham số động
        # Đảm bảo y_true lấy đầy đủ cả 2 cột mục tiêu cấu hình trong TARGET_FEATURES
        y_true = df_features[Config.TARGET_FEATURES].values[-Config.HORIZON:]
        
        # Xác định vị trí động để bóc tách cửa sổ dữ liệu đầu vào quá khứ (WINDOW_SIZE = 24)
        start_idx = -(Config.WINDOW_SIZE + Config.HORIZON)
        end_idx = -Config.HORIZON
        input_window = scaled_data[start_idx:end_idx]
        
        # 4. Thực thi dự báo đồng thời qua Predictor (Loại bỏ hoàn toàn đệ quy)
        pd_tool = Predictor()
        start_forecast_time = df_features.index[end_idx]
        forecast_out = pd_tool.predict_next_hours(input_window, n_steps=Config.HORIZON, start_time=start_forecast_time)
        
        # Cơ chế phòng vệ cấu hình (Shape Defense): Đảm bảo ma trận đầu ra giữ đúng cấu trúc 2D (24, 2)
        num_targets = len(Config.TARGET_FEATURES)
        if hasattr(forecast_out, 'reshape') and forecast_out.size == (Config.HORIZON * num_targets):
            forecast_out = forecast_out.reshape(Config.HORIZON, num_targets)
        
        # 5. Xuất kết quả đối chứng trực quan song song cả 2 biến mục tiêu đầu ra
        print("\n🔮 KẾT QUẢ DỰ BÁO VS THỰC TẾ ĐA BIẾN (24h cuối):")
        print(f"{'Thời gian':<8} | {'Dự báo MW':<12} | {'Thực tế MW':<12} | {'Dự báo Diff':<12} | {'Thực tế Diff':<12}")
        print("-" * 75)
        for i in range(Config.HORIZON):
            print(f"Giờ {i+1:<5} | "
                  f"{forecast_out[i][0]:<12.2f} | "
                  f"{y_true[i][0]:<12.2f} | "
                  f"{forecast_out[i][1]:<12.2f} | "
                  f"{y_true[i][1]:<12.2f}")
            
        # 6. Tính toán hệ thống sai số tích hợp đa mục tiêu (Multi-output Evaluation)
        evaluate_model(y_true, forecast_out)
        
    else:
        print("❌ Chế độ điều khiển không hợp lệ! Vui lòng sử dụng --mode train hoặc --mode test")

if __name__ == "__main__":
    main()