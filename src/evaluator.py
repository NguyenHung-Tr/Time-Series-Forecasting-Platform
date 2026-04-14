import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def evaluate_model(y_true, y_pred):
    # Tính toán các chỉ số quan trọng
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print("-" * 30)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ:")
    print(f"🔹 RMSE: {rmse:.2f} MW")
    print(f"🔹 MAPE: {mape:.2f} %")
    print("-" * 30)
    
    return rmse, mape