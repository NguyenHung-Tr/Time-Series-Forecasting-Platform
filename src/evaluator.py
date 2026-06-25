import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from utils.config import Config

def evaluate_model(y_true, y_pred):
    """
    Tính toán và hiển thị hệ thống chỉ số sai số đo lường (Metrics).
    Hỗ trợ bóc tách chi tiết hiệu năng cho từng biến mục tiêu đầu ra.
    """
    # 1. Tính toán sai số tích hợp tổng quan toàn hệ thống
    rmse_total = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_total = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae_total = mean_absolute_error(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP ĐA BIẾN MỤC TIÊU:")
    print(f"🔹 RMSE Tổng Hợp: {rmse_total:.2f}")
    print(f"🔹 MAPE Tổng Hợp: {mape_total:.2f} %")
    print(f"🔹 MAE Tổng Hợp:  {mae_total:.2f}")
    print("="*40)
    
    # 2. Bóc tách chi tiết sai số đo lường cho từng kênh biến đầu ra độc lập
    print("🔍 CHI TIẾT HIỆU NĂNG THEO TỪNG ĐẦU RA:")
    for idx, feature_name in enumerate(Config.TARGET_FEATURES):
        y_true_feat = y_true[:, idx]
        y_pred_feat = y_pred[:, idx]
        
        feat_rmse = np.sqrt(mean_squared_error(y_true_feat, y_pred_feat))
        feat_mape = mean_absolute_percentage_error(y_true_feat, y_pred_feat) * 100
        
        unit = "MW" if "MW" in feature_name else "Đơn vị/h"
        print(f"📍 Đặc trưng [{feature_name}]:")
        print(f"   -> RMSE: {feat_rmse:.2f} {unit}")
        print(f"   -> MAPE: {feat_mape:.2f} %")
    print("="*40)
    
    return rmse_total, mape_total