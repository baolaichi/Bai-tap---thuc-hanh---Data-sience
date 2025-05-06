import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Sinh dữ liệu giả lập
np.random.seed(42)

# Sinh 5000 bản ghi giả lập
timestamps = pd.date_range("2025-04-01 00:00:00", periods=5000, freq="H")
x_coords = np.random.uniform(0, 10, 5000)
y_coords = np.random.uniform(0, 10, 5000)
vehicle_types = np.random.choice(["Xe máy", "Ô tô", "Xe buýt"], size=5000)
speeds = np.random.uniform(0, 60, 5000)  # Tốc độ trung bình từ 0 đến 60 km/h
traffic_density = np.random.choice(["Thấp", "Trung bình", "Cao"], size=5000)

data = pd.DataFrame({
    'timestamp': timestamps,
    'x_coord': x_coords,
    'y_coord': y_coords,
    'vehicle_type': vehicle_types,
    'speed': speeds,
    'traffic_density': traffic_density
})

# 2. Phân cụm bằng K-Means
X = data[['x_coord', 'y_coord', 'speed']].values
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

# 3. Tính toán mức độ nghiêm trọng của tắc nghẽn
data['congestion_severity'] = np.where(
    (data['traffic_density'] == 'Cao') & (data['speed'] < 10),  # Mật độ cao và tốc độ thấp
    1, 0
)

# 4. Tạo đặc trưng "giờ cao điểm"
# Giả định giờ cao điểm từ 7:00 AM đến 9:00 AM
data['hour'] = data['timestamp'].dt.hour
data['rush_hour'] = np.where((data['hour'] >= 7) & (data['hour'] <= 9), 1, 0)

# 5. Tạo đặc trưng "tỷ lệ xe lớn"
vehicle_type_map = {'Xe máy': 0, 'Ô tô': 1, 'Xe buýt': 1}
data['is_large_vehicle'] = data['vehicle_type'].map(vehicle_type_map)
data['large_vehicle_ratio'] = data.groupby(['x_coord', 'y_coord'])['is_large_vehicle'].transform('mean')

# 6. Dự đoán mật độ giao thông sử dụng Gradient Boosting
# Chuyển đổi nhãn mật độ giao thông thành dạng số
traffic_density_map = {'Thấp': 0, 'Trung bình': 1, 'Cao': 2}
data['traffic_density_label'] = data['traffic_density'].map(traffic_density_map)

# Chọn đặc trưng và nhãn
X = data[['speed', 'rush_hour', 'large_vehicle_ratio', 'hour']]
y = data['traffic_density_label']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Gradient Boosting
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Hiển thị kết quả phân cụm và mức độ nghiêm trọng của tắc nghẽn
plt.figure(figsize=(10, 6))
plt.scatter(data['x_coord'], data['y_coord'], c=data['cluster'], cmap='viridis', label='Cluster')
plt.scatter(data[data['congestion_severity'] == 1]['x_coord'], data[data['congestion_severity'] == 1]['y_coord'],
            color='red', label='Congestion Points')
plt.title('Clustering and Congestion Severity')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()

