# 📡 Market Anomaly Radar

Ứng dụng Web chuyên nghiệp xây dựng bằng **Python**, **Streamlit**, **Plotly** và **PyTorch** nhằm phát hiện các biến động bất thường (Anomaly Detection) trên chuỗi thời gian dữ liệu tài chính thực tế bằng mạng nơ-ron hồi quy **LSTM Autoencoder**, đồng thời so sánh trực tiếp với phương pháp thống kê truyền thống **Z-Score Baseline**.

---

## 🌟 Tính Năng Nổi Bật

1. **Kiến Trúc Học Sâu LSTM Autoencoder**:
   - Sử dụng mô hình nén và giải nén chuỗi thời gian (Encoder-Decoder) để học hành vi bình thường của thị trường từ dữ liệu lịch sử.
   - Khi gặp một cấu trúc thị trường lạ (sốc giá, xả hàng đột biến, thiên nga đen), lỗi tái tạo (Reconstruction Error - MSE) sẽ tăng đột biến, kích hoạt tín hiệu cảnh báo bất thường.

2. **Chống Rò Rỉ Dữ Liệu Tuyệt Đối (No Data Leakage)**:
   - Dữ liệu được chia tuần tự theo thời gian (`Train Split` mặc định 70%, `Test Split` 30%).
   - `StandardScaler` và các thông số thống kê ($\mu, \sigma$) được `fit` **duy nhất** trên tập Train, sau đó mới dùng để chuẩn hóa tập Test và dữ liệu tương lai.

3. **Hai Chế Độ Phát Hiện Chuyên Sâu**:
   - **Last-Day (Nhạy, bắt sốc tức thời)**: Đánh giá MSE tại ngày cuối cùng trong cửa sổ trượt. Tối ưu cho việc phát hiện các cú flash crash, tin tức chấn động hoặc bơm xả trong ngày.
   - **Window Average (Bắt khủng hoảng kéo dài)**: Tính trung bình MSE trên toàn bộ cửa sổ trượt. Tối ưu cho việc nhận diện sự thay đổi chế độ thị trường (Regime Shift) và khủng hoảng dài hạn.

4. **Tương Tác Thời Gian Thực Không Cần Retrain**:
   - Tối ưu hóa với `st.session_state` để lưu trữ mô hình và ma trận lỗi.
   - Khi người dùng thay đổi thanh trượt **Threshold Percentile** (90% – 99.5%), các thẻ KPI, biểu đồ Plotly và bảng Top 15 được cập nhật ngay lập tức chỉ trong vài mili-giây mà **không phải huấn luyện lại mạng nơ-ron**.

5. **Trực Quan Hóa Đa Chiều Bằng Plotly**:
   - **Biểu đồ Giá Đóng Cửa**: Phân loại trực quan 3 kịch bản:
     - 🔴 **Chấm đỏ**: Chỉ có mô hình LSTM Autoencoder phát hiện bất thường.
     - 🟢 **Dấu X xanh lá**: Chỉ có phương pháp thống kê Z-Score (|Z| > 2.0) phát hiện.
     - ⭐ **Ngôi sao tím**: Cả hai phương pháp cùng đồng thuận phát hiện.
   - **Biểu đồ Reconstruction Error**: Thể hiện chuỗi sai số MSE kèm đường ngưỡng nét đứt động.

6. **Bảng Xếp Hạng & Xuất Dữ Liệu**:
   - Bảng hiển thị Top 15 phiên có mức độ bất thường cao nhất kèm đầy đủ các chỉ số: Ngày, Giá đóng cửa, Log Return, Volume Change, MSE Error, Z-Score.
   - Hỗ trợ nút tải xuống file CSV toàn bộ kết quả phân tích.

---

## 📁 Cấu Trúc Dự Án

```
market-anomaly-radar/
├── app.py              # Giao diện người dùng Streamlit, Dashboard tương tác, Plotly Charts
├── model.py            # Kiến trúc PyTorch LSTMAutoencoder, hàm train, hàm tính MSE Error
├── data_utils.py       # Tải dữ liệu yfinance, tính log_return & vol_change, scaling không rò rỉ, tạo cửa sổ 3D
├── requirements.txt    # Danh sách thư viện phụ thuộc
└── README.md           # Tài liệu hướng dẫn sử dụng và thuyết minh kiến trúc
```

---

## 🧠 Nguyên Lý Hoạt Động Của LSTM Autoencoder

Mô hình bao gồm hai khối chính:

```
[ Chuỗi Đầu Vào X ]  (Batch, Window Size, 2)
        │
        ▼
   [ LSTM ENCODER ]   (2 lớp LSTM, Hidden Dim = 48)
        │
        ▼
 [ Latent Bottleneck ] (Vector biểu diễn ẩn 16 chiều)
        │
        ▼
   [ LSTM DECODER ]   (Repeat Vector -> 2 lớp LSTM -> Linear)
        │
        ▼
[ Chuỗi Tái Tạo X̂ ]  (Batch, Window Size, 2)
```

- **Hàm mất mát (Loss Function)**: Mean Squared Error (MSE):
  $$\mathcal{L}(X, \hat{X}) = \frac{1}{W \times F} \sum_{w=1}^{W} \sum_{f=1}^{F} (X_{w,f} - \hat{X}_{w,f})^2$$
- Trong giai đoạn huấn luyện trên tập Train (thời kỳ thị trường tương đối bình thường), mạng Autoencoder học được cách nén và tái tạo chính xác các dao động thông thường.
- Trong giai đoạn kiểm thử (Test Set), các phiên xuất hiện hiện tượng biến động bất thường sẽ khiến bộ giải mã không thể tái tạo lại được chính xác, dẫn đến giá trị **MSE Error vượt qua ngưỡng phân vị (Threshold Percentile)**.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Chuẩn bị môi trường Python (Python 3.10+)

Mở Terminal / PowerShell và điều hướng vào thư mục dự án:
```bash
cd market-anomaly-radar
```

*(Khuyến nghị) Khởi tạo và kích hoạt môi trường ảo:*
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### 3. Khởi chạy Web App

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 📊 Hướng Dẫn Sử Dụng

1. **Nhập Ticker**: Điền mã cổ phiếu/crypto mong muốn (ví dụ: `BTC-USD`, `NVDA`, `AAPL`, `^GSPC`, `ETH-USD`, `TSLA`).
2. **Chọn Khoảng Thời Gian**: Chọn ngày bắt đầu và kết thúc (khuyến nghị từ 1 đến 3 năm để có đủ dữ liệu học mẫu).
3. **Chọn Chế Độ**: Chọn `Last-Day` nếu muốn bắt các cú sốc trong phiên, hoặc `Window Average` nếu muốn theo dõi khủng hoảng dài ngày.
4. **Bấm "🚀 Train & Detect Anomalies"**: Hệ thống sẽ tự động tải dữ liệu, tiền xử lý, huấn luyện mô hình PyTorch và hiển thị dashboard.
5. **Kéo thanh trượt Threshold Percentile**: Điều chỉnh độ nhạy của bộ lọc (90% - 99.5%) để xem các điểm bất thường biến thiên trực tiếp trên biểu đồ.
6. **Tải Dữ Liệu**: Tải file CSV kết quả để phục vụ nghiên cứu định lượng hoặc backtesting chiến lược.

---

## 🛠️ Công Nghệ Sử Dụng

- **Python 3.10+**
- **PyTorch**: Xây dựng kiến trúc Deep Learning LSTM Autoencoder.
- **Streamlit**: Giao diện người dùng Web tương tác cao, phản hồi nhanh.
- **Plotly**: Biểu đồ tài chính tương tác, hỗ trợ hover, zoom và crosshair.
- **yfinance**: Tải dữ liệu thị trường theo thời gian thực từ Yahoo Finance.
- **Scikit-learn**: Chuẩn hóa dữ liệu `StandardScaler` tuân thủ chuẩn phòng chống data leakage.
