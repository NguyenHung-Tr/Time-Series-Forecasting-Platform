# Time-Series-Forecasting-Platform

# Hệ thống Dự báo Phụ tải Điện Đa biến Dài hạn (Time-Series Forecasting Platform)

- **Ngành học:** Máy tính & Hệ thống nhúng (Khoa Điện tử Viễn thông) - Đồ án Tốt nghiệp
- **Kiến trúc AI:** Encoder-Decoder Sequence-to-Sequence (Seq2Seq) GRU đa biến.
- **Tập dữ liệu:** Phụ tải điện khu vực PJM AEP (Mỹ) kết hợp dữ liệu thời tiết lịch sử theo giờ (Open-Meteo).
- **Mục tiêu mở rộng:** Dự báo phụ tải điện theo từng vùng (Regional Load Forecasting) và áp dụng cấu trúc Đa biến dự báo Đa biến (Multivariate-to-Multivariate).

---

## 1. 📁 Cấu trúc Thư mục Hiện tại

```text
Time-Series-Forecasting-Platform/
│
├── data/
│   └── raw/
│       ├── AEP_hourly.csv          # Dữ liệu phụ tải điện gốc (MW)
│       └── weather_hourly.csv      # Dữ liệu thời tiết lịch sử vùng Columbus, Ohio
│
├── models/
│   ├── saved_models/
│   │   ├── feature_scaler.pkl      # Thước đo chuẩn hóa 9 đặc trưng đầu vào
│   │   ├── target_scaler.pkl       # Thước đo chuẩn hóa riêng cho cột mục tiêu (MW)
│   │   └── seq2seq_model.h5        # File đóng gói mô hình Seq2Seq GRU tối ưu
│   └── gru_model.py                # Kiến trúc mạng mạng Neural Network (Keras/TensorFlow)
│
├── src/
│   ├── data_loader.py              # Đọc dữ liệu thô, lọc bỏ metadata và gộp (Inner Join) thời tiết
│   ├── preprocessor.py             # Lưu/tải Scaler bằng Joblib, chia tập Train/Test dữ liệu
│   ├── feature_builder.py          # Trích xuất đặc trưng chu kỳ (Sin/Cos), ngày lễ, tạo ma trận chuỗi 3D
│   ├── trainer.py                  # Luồng huấn luyện tự động (EarlyStopping, ModelCheckpoint)
│   ├── predictor.py                # Khởi tạo mô hình và dự báo đồng thời 24h không dùng đệ quy
│   └── evaluator.py                # Tính toán sai số đo lường hệ thống (RMSE, MAE, MAPE)
│
├── utils/
│   ├── config.py                   # Quản lý siêu tham số hệ thống (Hyperparameters)
│   └── plotter.py                  # Module xuất biểu đồ đối chứng định dạng PNG
│
├── main.py                         # File thực thi và điều hướng hệ thống (--mode train / --mode test)
└── README.md                       # Nhật ký logic và hướng dẫn dự án (File này)

🛠️ 2. Các Công nghệ & Thư viện Đang dùng
Hạ tầng AI/Deep Learning: tensorflow, keras (Cấu hình tương thích Keras 3 sử dụng đối tượng MeanSquaredError() và cơ chế compile=False khi nạp mô hình).
Xử lý Dữ liệu: pandas, numpy, scikit-learn (MinMaxScaler).
Lưu trữ Trạng thái bộ nhớ: joblib (Đóng gói trạng thái Scaler phục vụ môi trường chạy độc lập).
Trích xuất bối cảnh: holidays (Tự động nhận diện lịch nghỉ lễ quốc gia).
Môi trường vận hành: Python Virtual Environment (venv), argparse điều khiển Terminal.

🧠 3. Logic Vận hành của các File Quan trọng
🔹 utils/config.py
Nơi cấu hình tập trung siêu tham số. Đầu vào hệ thống (FEATURES) gồm 9 cột: ['AEP_MW', 'temperature', 'humidity', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_holiday', 'rolling_mean_24h']. Cửa sổ quá khứ WINDOW_SIZE = 24 và chuỗi dự báo tương lai HORIZON = 24.

🔹 src/data_loader.py
Xử lý đồng bộ dữ liệu. Hàm load_and_merge_weather thực hiện đọc file thời tiết, bỏ qua 3 dòng metadata đầu trạm đo (skiprows=3), xử lý nội suy tuyến tính điền khuyết dữ liệu (interpolate), sau đó dùng join(how='inner') dựa trên Datetime Index để ép khớp thời gian hoàn hảo với file điện năng.

🔹 src/preprocessor.py
Quản lý thang đo độc lập. Tách biệt feature_scaler (9 đặc trưng vào) và target_scaler (riêng cột MW ra). Cơ chế kiểm tra df.empty ở chế độ test cho phép nạp trực tiếp Scaler từ file .pkl vào bộ nhớ RAM mà không cần phụ thuộc vào dữ liệu mồi đầu vào.

🔹 src/feature_builder.py
Xây dựng ma trận không gian 3D. Hàm add_features chuyển đổi cột giờ và thứ sang tọa độ phẳng qua hàm Sin/Cos để AI hiểu tính tuần hoàn thời gian. Hàm create_sequences đóng gói dữ liệu thành dạng tensor đầu vào $X$ (Số_mẫu, 24, 9) và nhãn chuỗi tương lai $y$ (Số_mẫu, 24).

🔹 models/gru_model.py
Mô hình Seq2Seq (Joint Multi-step Forecasting) loại bỏ hoàn toàn sai số đệ quy:
Encoder: Lớp GRU (64 neurons) nén toàn bộ bối cảnh quá khứ đa biến thành vector trạng thái phẳng.
Bridge: Lớp RepeatVector(24) sao chép vector trạng thái ra đúng số bước thời gian cần dự báo.
Decoder: Lớp GRU (64 neurons, return_sequences=True) giải mã chuỗi.
Output Layer: Lớp TimeDistributed(Dense(1)) tính toán song song giá trị cho tất cả các giờ đầu ra.

🔹 src/predictor.py
Nạp mô hình tối ưu bằng lệnh load_model(..., compile=False). Thực hiện dự báo nhanh bằng một lệnh model.predict() duy nhất, sau đó dùng target_scaler.inverse_transform để giải nén trực tiếp ma trận kết quả về đơn vị MW thực tế.

## 📈 4. Lịch sử Tiến độ & Kết quả Đánh giá

1. **Giai đoạn 1: Baseline Đệ quy (Nhánh `baseline-recursive` - Đã đóng băng)**
   - Mô hình: Đa biến dự báo đơn bước, lặp `for` 24 lần cuốn chiếu để đoán tương lai.
   - Kết quả: Bị bùng nổ sai số tích lũy (Error Accumulation). Giờ dự báo thứ 24 bị lệch nghiêm trọng lên mức 31,000 MW. **MAPE: 38.64%**.

2. **Giai đoạn 2: Kiến trúc Seq2Seq Đơn biến (Nhánh `main` cũ)**
   - Mô hình: Encoder-Decoder GRU dự báo đồng thời (Joint Multi-step) trọn vẹn 24 bước song song, loại bỏ vòng lặp đệ quy. Chỉ sử dụng duy nhất 1 biến đầu vào là Phụ tải điện (`AEP_MW`).
   - Kết quả: Triệt tiêu hoàn toàn hiện tượng bùng nổ sai số. **RMSE: 980.15 MW | MAPE: 4.33%**.

3. **Giai đoạn 3: Kiến trúc Seq2Seq + Tích hợp Thời tiết (Nhánh `main` hiện tại)**
   - Mô hình: Giữ nguyên cơ chế dự báo đồng thời 24 bước của Seq2Seq GRU, mở rộng ma trận đầu vào lên 9 đặc trưng (`FEATURES`), tích hợp sâu dữ liệu ngoại lai gồm Nhiệt độ và Độ ẩm lịch sử theo giờ. Mô hình tự động đóng băng huấn luyện nhờ Early Stopping tại Epoch 19/50.
   - Kết quả: Mô hình bám sát hình thái đồ thị phụ tải thực tế, phản ánh chính xác tác động của thời tiết lên lưới điện mà không bị quá khớp (Overfitting). **RMSE: 1261.33 MW | MAPE: 6.44%** (Đạt chuẩn xuất sắc cho mô hình đa biến thực tế).

## 🔮 5. Hướng Mở Rộng Tiếp Theo
Cải tiến Output (Đa biến dự báo Đa biến): Cấu hình lại hệ thống để mô hình dự báo song song hai cột mục tiêu đầu ra gồm: Công suất thực tế (AEP_MW) và Tốc độ thay đổi phụ tải điện (MW_diff).
Dự báo theo từng vùng miền (Regional Forecasting): Thiết lập kiến trúc nạp song song dữ liệu phụ tải và thời tiết của các khu vực/vùng lưới điện khác nhau, chứng minh tính tổng quát hóa trên diện rộng của mô hình Seq2Seq.  Hãy viết tất cả dưới dạng markdown để tôi copy bỏ vô vscode không bị lỗi text dính liền tát cả trên dòng
```
