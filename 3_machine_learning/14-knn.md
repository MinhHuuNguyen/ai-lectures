---
slug: knn
time: 11/09/2024
title: Mô hình K-Nearest Neighbors (KNN)
description:
author: Nguyễn Hữu Minh
banner_url: 
tags: [machine-learning]
is_highlight: false
is_published: false
---

# K-Nearest Neighbors


## 1. Ý tưởng chung:

- KNN là một thuật toán Supervised Learning, thường được sử dụng cho cả Classification và Regression.
- Nguyên tắc hoạt động của KNN dựa trên việc đo đạc khoảng cách giữa các điểm dữ liệu.
- Dữ liệu được biểu diễn trong không gian nhiều chiều (n chiều), và KNN đo đạc khoảng cách giữa các điểm dữ liệu để đưa ra dự đoán.
- Các điểm gần nhau nhất sẽ có ảnh hưởng lớn đến việc đưa ra dự đoán.

## 2. Các bước thực hiện:

### 2.1. Quá trình huấn luyện:

- **Bước 1:** Chuẩn bị và lưu trữ dữ liệu:
    - Tiền xử lý dữ liệu để loại bỏ nhiễu và chuẩn hóa các đặc trưng.
    - KNN không huấn luyện một mô hình trước, mà chỉ lưu trữ dữ liệu đào tạo.
    - Mỗi điểm dữ liệu được biểu diễn trong không gian nhiều chiều với các đặc trưng tương ứng.
- **Bước 2:** Chọn giá trị K:
    - Quyết định giá trị K, tức là số lượng láng giềng gần nhất sẽ được sử dụng để đưa ra quyết định.

### 2.2. Quá trình dự đoán:

Từ kết quả đã có từ quá trình huấn luyện,

- **Bước 3:** Đưa điểm dữ liệu mới vào:
    - Mã hoá điểm dữ liệu mới thành vector đặc trưng.
- **Bước 4:** Đo đạc khoảng cách:
    - Đo đạc khoảng cách giữa điểm dữ liệu mới và các điểm dữ liệu huấn luyện đã có.
    - Phương pháp đo khoảng cách thường sử dụng là Euclidean distance.
    - Các phương pháp khác bao gồm Manhattan distance, Minkowski distance, Cosine similarity ...
- **Bước 5:** Đưa ra dự đoán:
    - Lựa chọn K láng giềng gần nhất theo khoảng cách của điểm mới.
    - Tính toán và lựa chọn lớp hoặc giá trị dự đoán dựa trên đa số lớp hoặc giá trị của các láng giềng.
        - Trong trường hợp phân loại, lớp xuất hiện nhiều nhất trong K láng giềng sẽ được chọn là kết quả.
        - Trong trường hợp hồi quy, giá trị trung bình của K láng giềng sẽ được chọn là kết quả.

### 2.3. Cải thiện mô hình KNN:

- Weighted KNN:
    - Có thể sử dụng trọng số cho các láng giềng dựa trên khoảng cách, nghĩa là láng giềng gần hơn sẽ có ảnh hưởng lớn hơn đối với quyết định.
    - Ví dụ: Có thể sử dụng trọng số bằng nghịch đảo của khoảng cách.
- Quyết Định Mức Độ Tin Cậy:
    - Có thể sử dụng thông tin về mức độ tin cậy của K láng giềng để ước lượng độ chắc chắn của dự đoán.
    - Ví dụ: KNN có thể đưa ra dự đoán dựa trên 5 láng giềng gần nhất, nhưng chỉ có 3 láng giềng gần nhất là chắc chắn.

## 3. Ưu Điểm và Nhược Điểm:

- Ưu Điểm:
    - Dễ hiểu và dễ triển khai.
    - Không cần huấn luyện mô hình, do đó tiết kiệm thời gian huấn luyện.

- Nhược Điểm:
    - Nhạy cảm với nhiễu và outliers.
    - Yêu cầu nhiều tài nguyên tính toán khi dữ liệu lớn.
    - Lớp dữ liệu không cân bằng có thể tạo ra các kết quả chệch.
