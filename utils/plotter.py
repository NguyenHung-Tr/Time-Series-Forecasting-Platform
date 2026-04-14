import matplotlib.pyplot as plt

def plot_forecast(history_data, forecast_data):
    plt.figure(figsize=(12, 6))
    
    # Vẽ 48 giờ quá khứ để làm nền
    plt.plot(range(len(history_data)), history_data, label='Dữ liệu quá khứ', color='blue')
    
    # Vẽ 24 giờ dự báo nối tiếp vào
    plt.plot(range(len(history_data), len(history_data) + len(forecast_data)), 
             forecast_data, label='AI Dự báo', color='red', linestyle='--')
    
    plt.title('Dự báo phụ tải điện 24 giờ tới')
    plt.xlabel('Thời gian (Giờ)')
    plt.ylabel('Công suất (MW)')
    plt.legend()
    plt.grid(True)
    plt.show()