---
time: 08/13/2022
title: Mô hình K-means Clustering
description: Bên cạnh các mô hình học có giám sát, mô hình học không giám sát cũng đóng một vai trò quan trọng trong Machine Learning. Trong bài viết này, chúng ta sẽ tìm hiểu về mô hình phân cụm K-means Clustering, mô hình giúp phân chia dữ liệu thành các cụm dựa trên đặc trưng của chúng.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/banner.png
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Tổng quan

K-means Clustering là thuật toán unsupervised learning dùng để nhóm các điểm dữ liệu thành các cụm dựa trên đặc trưng của chúng.

Nói cách khác, mô hình K-means Clustering giúp phân chia dữ liệu thành các cụm sao cho các điểm dữ liệu trong cùng một cụm có đặc trưng tương tự nhau và các điểm dữ liệu ở các cụm khác nhau có đặc trưng khác biệt.

## 2. Các bước của thuật toán

- **Bước 1**: Chọn số K mà chúng ta muốn phân chia dữ liệu.

- **Bước 2**: Chọn ngẫu nhiên K điểm dữ liệu từ tập dữ liệu làm các tâm, đại diện cho các cụm (centroids).

- **Bước 3**: Với mỗi điểm dữ liệu, ta tìm tâm gần nhất và gán điểm dữ liệu đó vào cụm tương ứng với tâm đó.

- **Bước 4**: Tính toán tâm mới của mỗi cụm bằng cách lấy trung bình của tất cả các điểm dữ liệu trong cụm đó.

- **Bước 5**: Lặp lại **Bước 3** và **Bước 4** cho đến khi không có sự thay đổi nào trong việc gán điểm dữ liệu vào các cụm.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/example_step_init.png" style="width: 400px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/example_step_1.png" style="width: 800px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/example_step_2.png" style="width: 800px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/example_step_3.png" style="width: 800px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/example_step_4.png" style="width: 800px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/example_step_5.png" style="width: 800px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/example_animation.gif" style="width: 400px;"/>

## 3. Công thức tính khoảng cách

Để xác định “gần nhất”, ta cần định nghĩa hàm khoảng cách giữa hai điểm $x$ và $y$ trong không gian nhiều chiều.

### 3.1. Khoảng cách Euclidean

Khoảng cách Euclidean là khoảng cách phổ biến nhất được sử dụng trong KNN. Nó được tính bằng công thức:

$$ d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} $$

trong đó:
- $x_i$ và $y_i$ là các thành phần của vector $x$ và $y$.

Khoảng cách Euclidean thường được sử dụng trong đa số các trường hợp.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/euclidean.png" style="width: 500px;"/>

### 3.2. Khoảng cách Manhattan

Khoảng cách Manhattan (hay còn gọi là khoảng cách L1) được tính bằng tổng các khoảng cách tuyệt đối giữa các thành phần của hai vector. Nó được tính bằng công thức:

$$ d(x, y) = \sum_{i=1}^{n} |x_i - y_i| $$

trong đó:
- $x_i$ và $y_i$ là các thành phần của vector $x$ và $y$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/manhattan.png" style="width: 500px;"/>

### 3.3. Khoảng cách Minkowski

Khoảng cách Minkowski là một tổng quát của khoảng cách Euclidean và Manhattan. Nó được tính bằng công thức:

$$ d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}} $$

trong đó:
- $p$ là một tham số điều chỉnh khoảng cách.
Khi $p=1$, nó trở thành khoảng cách Manhattan, và khi $p=2$, nó trở thành khoảng cách Euclidean.
- $x_i$ và $y_i$ là các thành phần của vector $x$ và $y$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/minkowski_dist.png" style="width: 500px;"/>

### 3.4. Khoảng cách Cosine

Khoảng cách Cosine đo lường độ tương đồng giữa hai vector bằng cách tính cosine của góc giữa chúng. Nó được tính bằng công thức:

$$ d(x, y) = 1 - \frac{x \cdot y}{||x|| \cdot ||y||} $$

trong đó:
- $x \cdot y$ là tích vô hướng của hai vector $x$ và $y$.
- $||x||$ và $||y||$ là độ dài (norm) của vector $x$ và $y$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/cosine.png" style="width: 500px;"/>

## 4. Tối ưu trong mô hình K-means

## 5. Phương pháp lựa chọn số lượng cụm

### 5.1. Phương pháp Elbow

### 5.2. Phương pháp Silhouette

## 6. Ưu điểm và nhược điểm của mô hình

## 7. Các biến thể nâng cấp của mô hình K-means Clustering

### 7.1. K-means++

### 7.2. Mini-batch K-means

### 7.3. K-means Parallel

### 7.4. K-means Spectral

### 7.5. K-means Time Series

### 7.6. K-means Fuzzy

### 7.7. Gaussian Mixture Model (GMM)


