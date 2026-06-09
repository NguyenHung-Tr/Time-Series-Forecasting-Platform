import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_builder import add_features
from src.trainer import train_pipeline
from src.predictor import Predictor
from src.evaluator import evaluate_model
from src.data_loader import load_raw_data, preprocess_datetime, load_and_merge_weather
from src.preprocessor import Preprocessor
from utils.config import Config

def main():
    parser = argparse.ArgumentParser(description="Hệ thống dự báo phụ tải điện HCMUS")
    parser.add_argument('--mode', type=str, default='test', help='Chế độ: train hoặc test')
    args = parser.parse_args()

    if args.mode == 'train':
        print("🏗️  Đang khởi động quá trình huấn luyện...")
        train_pipeline()
        
    elif args.mode == 'test':
        print("📊 Đang kiểm tra mô hình và dự báo...")
        
        # 1. Nạp dữ liệu thực tế 
        df_raw = load_raw_data()
        df_time = preprocess_datetime(df_raw)
        df_combined = load_and_merge_weather(df_time)

        pre = Preprocessor()
        df_clean = pre.clean_data(df_combined)
        df_features = add_features(df_clean)
        scaled_data = pre.scale_data(df_features, is_training=False)
        
        # 2. Tách dữ liệu thực tế (y_true) và dữ liệu đầu vào (input)
        y_true_mw = df_features[Config.TARGET_COL].values[-24:].reshape(-1, 1)
        input_window = scaled_data[-48:-24]
        
        # 3. Gọi Predictor để đoán 24 giờ tiếp theo
        pd_tool = Predictor()
        start_forecast_time = df_features.index[-48]
        forecast_mw = pd_tool.predict_next_hours(input_window, n_steps=24, start_time=start_forecast_time)
        
        # 4. In kết quả dự báo
        print("\n🔮 KẾT QUẢ DỰ BÁO VS THỰC TẾ (24h cuối):")
        print(f"{'Giờ':<10} | {'Dự báo (MW)':<15} | {'Thực tế (MW)':<15}")
        print("-" * 45)
        for i in range(24):
            print(f"Giờ {i+1:<6} | {forecast_mw[i][0]:<15.2f} | {y_true_mw[i][0]:<15.2f}")
            
        # 5. TÍNH TOÁN SAI SỐ (MAPE, RMSE, MAE)
        evaluate_model(y_true_mw, forecast_mw)
    else:
        print("❌ Chế độ không hợp lệ. Hãy dùng --mode train hoặc --mode test")

if __name__ == "__main__":
    main()