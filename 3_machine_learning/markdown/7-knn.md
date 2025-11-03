---
time: 08/05/2022
title: Mô hình K Nearest Neighbors (KNN)
description: KNN là một trong những mô hình machine learning đơn giản và dễ hiểu nhất, có thể được sử dụng cho cả bài toán phân loại và hồi quy.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/banner.jpeg
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Tổng quan

K-Nearest Neighbors (KNN) là một trong những thuật toán học có giám sát (supervised learning) đơn giản và trực quan, có thể áp dụng cho cả bài toán phân loại (classification) và hồi quy (regression).

Nguyên lý cơ bản của KNN: để dự đoán nhãn (class đối với bài toán classification) hay giá trị (value đối với bài toán regression) của một điểm dữ liệu mới, KNN tìm K điểm dữ liệu “gần nhất” trong tập huấn luyện rồi dùng thông tin từ các điểm đó để đưa ra dự đoán.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/classification_vs_regression.jpeg" style="width: 1000px;"/>

## 2. Các bước của thuật toán

- **Bước 1:** Chọn số K là số lượng láng giềng gần nhất mà ta sẽ xét với mỗi điểm dữ liệu đầu vào trong tương lai.

- **Bước 2:** Mã hoá các điểm dữ liệu trong bộ dữ liệu train thành các vector đặc trưng trong không gian nhiều chiều.

- **Bước 3:** Lưu trữ các vector đặc trưng của các điểm dữ liệu trong bộ dữ liệu train.

- **Bước 4:** Với một điểm dữ liệu mới, mã hoá nó thành vector đặc trưng trong không gian nhiều chiều giống như các điểm dữ liệu trong bộ dữ liệu train.

- **Bước 5:** Tính toán khoảng cách giữa vector đặc trưng của điểm dữ liệu mới và các vector đặc trưng của các điểm dữ liệu trong bộ dữ liệu train.

- **Bước 6:** Sắp xếp và chọn ra K điểm dữ liệu gần nhất với điểm dữ liệu mới.

- **Bước 7:**
    - Đối với bài toán classification: Chọn lớp xuất hiện nhiều nhất trong K điểm gần nhất làm dự đoán cho điểm dữ liệu mới.
    - Đối với bài toán regression: Tính giá trị trung bình của K điểm gần nhất làm dự đoán cho điểm dữ liệu mới.

## 3. Công thức tính khoảng cách

Để xác định “gần nhất”, ta cần định nghĩa hàm khoảng cách giữa hai điểm $x$ và $y$ trong không gian nhiều chiều.

### 3.1. Khoảng cách Euclidean

Khoảng cách Euclidean là khoảng cách phổ biến nhất được sử dụng trong KNN. Nó được tính bằng công thức:

$$ d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} $$

trong đó:
- $x_i$ và $y_i$ là các thành phần của vector $x$ và $y$.

Khoảng cách Euclidean thường được sử dụng trong đa số các trường hợp.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/euclidean.jpeg" style="width: 500px;"/>

### 3.2. Khoảng cách Manhattan

Khoảng cách Manhattan (hay còn gọi là khoảng cách L1) được tính bằng tổng các khoảng cách tuyệt đối giữa các thành phần của hai vector. Nó được tính bằng công thức:

$$ d(x, y) = \sum_{i=1}^{n} |x_i - y_i| $$

trong đó:
- $x_i$ và $y_i$ là các thành phần của vector $x$ và $y$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/manhattan.jpeg" style="width: 500px;"/>

### 3.3. Khoảng cách Minkowski

Khoảng cách Minkowski là một tổng quát của khoảng cách Euclidean và Manhattan. Nó được tính bằng công thức:

$$ d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}} $$

trong đó:
- $p$ là một tham số điều chỉnh khoảng cách.
Khi $p=1$, nó trở thành khoảng cách Manhattan, và khi $p=2$, nó trở thành khoảng cách Euclidean.
- $x_i$ và $y_i$ là các thành phần của vector $x$ và $y$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/minkowski_dist.jpeg" style="width: 500px;"/>

### 3.4. Khoảng cách Cosine

Khoảng cách Cosine đo lường độ tương đồng giữa hai vector bằng cách tính cosine của góc giữa chúng. Nó được tính bằng công thức:

$$ d(x, y) = 1 - \frac{x \cdot y}{||x|| \cdot ||y||} $$

trong đó:
- $x \cdot y$ là tích vô hướng của hai vector $x$ và $y$.
- $||x||$ và $||y||$ là độ dài (norm) của vector $x$ và $y$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/cosine.jpeg" style="width: 500px;"/>

## 4. Ưu điểm và nhược điểm của KNN

- Ưu Điểm:
    - Dễ hiểu và dễ triển khai.
    - Không cần huấn luyện mô hình, do đó tiết kiệm thời gian huấn luyện.

- Nhược Điểm:
    - Yêu cầu nhiều tài nguyên tính toán khi dữ liệu lớn.
    - Nhạy cảm với nhiễu và outliers.
    - Lớp dữ liệu không cân bằng có thể tạo ra các kết quả chệch.

## 5. Các biến thể nâng cấp của KNN

### 5.1. Weighted KNN:

Có thể sử dụng trọng số cho các láng giềng dựa trên khoảng cách, nghĩa là láng giềng gần hơn sẽ có ảnh hưởng lớn hơn đối với quyết định.
Trọng số có thể được tính bằng cách sử dụng hàm khoảng cách, ví dụ như:
$$ w(x_i) = \frac{1}{d(x, x_i)} $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/weighted_knn.jpeg" style="width: 500px;"/>

Ví dụ: Xét 1 điểm dữ liệu mới $x$ và 3 điểm láng giềng gần nhất $x_1$, $x_2$, $x_3$ thuộc các lớp $C_1$, $C_1$, $C_2$ với khoảng cách tương ứng là $d_1$, $d_2$, $d_3$. Ta có thể tính trọng số cho các lớp như sau:
$$ w(C_1) = \frac{1}{d_1} + \frac{1}{d_2} $$
$$ w(C_2) = \frac{1}{d_3} $$
Sau đó, lớp có trọng số lớn nhất sẽ được chọn làm dự đoán cho điểm dữ liệu mới.

### 5.2. Dynamic-K KNN (Adaptive KNN)

Trong một số vùng của không gian dữ liệu, các điểm huấn luyện có thể tập trung dày đặc (mật độ cao), trong khi ở vùng khác lại thưa thớt.
Dùng cùng một giá trị K cho tất cả các điểm dữ liệu có thể không phải là lựa chọn tốt nhất, có thể khiến ở vùng thưa dữ liệu ta phải “vươn” rất xa để lấy đủ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/dynamic-k-knn.jpeg" style="width: 500px;"/>

Ta có một số chiến lược để điều chỉnh giá trị K cho từng điểm dữ liệu:
- **Dựa vào khoảng cách:** Thay vì đếm đúng K láng giềng, ta hướng tới "lấy tất cả các điểm trong bán kính R cố định", hoặc lập bán kính sao cho luôn có ít nhất K_min hoặc K_max điểm trong bán kính đó.
- **Elbow method:** Tính toán khoảng cách trung bình đến K láng giềng gần nhất và chọn K sao cho khoảng cách này là nhỏ nhất.
