import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Đọc dữ liệu từ file CSV
df = pd.read_csv('Data_Number_4.csv')

# Chuyển đổi cột timestamp thành định dạng datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Tiền xử lý dữ liệu
# Đảm bảo không có giá trị thiếu
df = df.dropna()

# 1. Phân cụm các khu vực bằng DBSCAN dựa trên tọa độ (x, y) và nồng độ PM2.5
X = df[['x_coord', 'y_coord', 'pm25']].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Sử dụng DBSCAN với epsilon = 0.5 và min_samples = 3
db = DBSCAN(eps=0.5, min_samples=3)
df['cluster'] = db.fit_predict(X_scaled)

# 2. Tính toán chỉ số "rủi ro ô nhiễm" cho mỗi khu vực
threshold = 50  # Ngưỡng an toàn PM2.5
df['risk_index'] = (df['pm25'] > threshold).astype(int)

# 3. Tạo đặc trưng "chỉ số thời tiết bất lợi"
# Định nghĩa chỉ số thời tiết bất lợi
df['weather_risk'] = (df['humidity'] > 70) & (df['wind_speed'] < 10)  # Cách tính có thể thay đổi

# 4. Tạo đặc trưng "xu hướng ô nhiễm"
df['pm25_slope'] = df['pm25'].diff()  # Độ dốc của PM2.5 trong 1 giờ qua

# 5. Xây dựng mô hình LSTM để dự đoán PM2.5 trong 6 giờ tới
# Tạo dữ liệu huấn luyện cho mô hình LSTM
def create_lstm_data(df, hours=24):
    X, y = [], []
    for i in range(hours, len(df) - 6):  # Lấy dữ liệu từ 24 giờ trước để dự đoán 6 giờ sau
        X.append(df['pm25'].iloc[i-hours:i].values)
        y.append(df['pm25'].iloc[i+6])  # Dự đoán PM2.5 sau 6 giờ
    return np.array(X), np.array(y)

X_train, y_train = create_lstm_data(df)

# Chuẩn bị dữ liệu cho LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Dự đoán PM2.5 cho 6 giờ tới
predictions = model.predict(X_train)

# Đánh giá mô hình
rmse = np.sqrt(mean_squared_error(y_train, predictions))
print(f'RMSE: {rmse}')

# Vẽ biểu đồ so sánh giá trị thực tế và dự đoán
plt.plot(y_train, label='Thực tế')
plt.plot(predictions, label='Dự đoán')
plt.legend()
plt.show()

# Lưu kết quả phân cụm vào file CSV
df.to_csv('clustered_data.csv', index=False)
