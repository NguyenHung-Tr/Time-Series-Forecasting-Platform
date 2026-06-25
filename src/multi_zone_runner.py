import os
import sys

# Đảm bảo nhận diện package nội bộ dự án
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from src.trainer import train_pipeline

def run_regional_forecasting():
    """
    Script điều phối trung tâm: Tự động lặp qua danh sách các vùng miền tại Mỹ,
    thiết lập bối cảnh dynamic và kích hoạt tiến trình huấn luyện biệt lập.
    """
    print("=========================================================================")
    print("🌍 KHỞI ĐỘNG HỆ THỐNG DỰ BÁO PHỤ TẢI ĐA VÙNG MIỀN LƯỚI ĐIỆN PJM (MỸ)")
    print("=========================================================================\n")
    
    # Danh sách các vùng miền hệ thống đã chuẩn bị sẵn hạ tầng
    regions_to_train = ["AEP", "COMED", "DAYTON"]
    
    for zone in regions_to_train:
        print(f"\n[TIẾN TRÌNH] ----------------> Đang xử lý phân vùng miền: {zone}")
        
        # Cơ chế ép cấu hình động (Dynamic Context Overriding)
        Config.ZONE = zone
        
        # Kiểm tra sự tồn tại của dữ liệu thô trước khi kích hoạt pipeline
        if not os.path.exists(Config.RAW_DATA_PATH):
            print(f"⚠️ Bỏ qua vùng [{zone}]: Không tìm thấy file phụ tải điện tại {Config.RAW_DATA_PATH}")
            continue
            
        if not os.path.exists(Config.WEATHER_DATA_PATH):
            print(f"⚠️ Bỏ qua vùng [{zone}]: Không tìm thấy file thời tiết tại {Config.WEATHER_DATA_PATH}")
            continue
            
        try:
            # Cập nhật đường dẫn lưu mô hình biệt lập cho từng vùng miền, tránh ghi đè
            Config.MODEL_SAVE_PATH = os.path.join(
                Config.BASE_DIR, 'models', 'saved_models', f'seq2seq_model_{zone}.h5'
            )
            
            # Kích hoạt luồng huấn luyện chuẩn cho phân vùng hiện tại
            train_pipeline()
            print(f"🔹 Huấn luyện thành công vùng [{zone}]. Mô hình lưu tại: {Config.MODEL_SAVE_PATH}")
            
        except Exception as e:
            print(f"❌ Tiến trình huấn luyện vùng [{zone}] gặp sự cố kỹ thuật: {e}")
            
    print("\n=========================================================================")
    print("✅ HOÀN THÀNH TOÀN BỘ TIẾN TRÌNH HUẤN LUYỆN ĐA VÙNG MIỀN (REGIONAL FORECASTING)")
    print("=========================================================================")

if __name__ == "__main__":
    run_regional_forecasting()