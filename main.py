import sys
import os
import argparse

# Fix lỗi import cho môi trường Windows/Linux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trainer import train_pipeline
from src.predictor import Predictor
from src.evaluator import evaluate_model
from src.data_loader import load_raw_data, preprocess_datetime
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
        
        # 1. Nạp dữ liệu thực tế để lấy cửa sổ cuối cùng (last window)
        df_raw = load_raw_data()
        df_time = preprocess_datetime(df_raw)
        
        pre = Preprocessor()
        df_clean = pre.clean_data(df_time)
        scaled_data = pre.scale_data(df_clean)
        
        # 2. Lấy 24 giờ cuối cùng trong tập dữ liệu để làm mồi dự báo
        last_window = scaled_data[-Config.WINDOW_SIZE:]
        
        # 3. Gọi Predictor để đoán 24 giờ tiếp theo
        pd_tool = Predictor()
        forecast = pd_tool.predict_next_hours(last_window, n_steps=24)
        
        print("\n🔮 KẾT QUẢ DỰ BÁO 24 GIỜ TIẾP THEO (MW):")
        for i, val in enumerate(forecast):
            print(f"Giờ {i+1}: {val[0]:.2f} MW")
            
        # Lưu ý: Ở đây bạn có thể gọi thêm evaluator nếu có dữ liệu thực tế đối chứng
    else:
        print("❌ Chế độ không hợp lệ. Hãy dùng --mode train hoặc --mode test")

if __name__ == "__main__":
    main()