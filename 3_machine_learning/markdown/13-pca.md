---
time: 08/19/2022
title: Mô hình PCA
description: Làm việc trực tiếp trên dữ liệu có số chiều cao gây ra khó khăn cả về việc lưu trữ và tốc độ tính toán. Do đó, giảm chiều dữ liệu là một bài toán có tính ứng dụng cao trong Machine Learning, giúp lưu trữ và xử lý dữ liệu với hiệu năng tốt hơn. PCA là mô hình giảm chiều dữ liệu đại diện cho nhóm các mô hình tuyến tính, dựa vào các phép toán trên ma trận để giảm chiều dữ liệu.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/13-pca/banner.png
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Tổng quan

Giảm chiều dữ liệu (Dimensionality reduction) là quá trình giảm số chiều của dữ liệu trong không gian đa chiều.
Mục tiêu của quá trình này là làm cho việc xử lý và phân tích dữ liệu trở nên hiệu quả hơn mà không làm mất đi quá nhiều thông tin quan trọng.

Có hai loại chính của giảm chiều dữ liệu:
- Giảm chiều dữ liệu tuyến tính:
Chiếu dữ liệu từ không gian nhiều chiều sang không gian ít chiều hơn thông qua các phép biến đổi tuyến tính.
Đại diện là mô hình Principal Component Analysis (PCA).
- Giảm chiều dữ liệu phi tuyến tính:
Tìm ra mối quan hệ giữa các điểm dữ liệu và cố gắng duy trì được mối quan hệ này trên không gian mới có số chiều thấp hơn.
Đại diện là mô hình t-distributed Stochastic Neighbor Embedding (t-SNE).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/13-pca/idea.png" width="400"/>

Principal Component Analysis (PCA) là một phương pháp giảm chiều dữ liệu tuyến tính được ra đời từ năm 1901 bởi Karl Pearson.

PCA giúp giảm số chiều của dữ liệu bằng cách tìm kiếm các thành phần chính (principal components) mà giữ lại nhiều phương sai nhất trong dữ liệu.
Điều này giúp bảo toàn sự biến động lớn nhất trong dữ liệu.

Nói cách khác, PCA sẽ đánh giá mức độ quan trọng của từng chiều dữ liệu và loại bỏ những chiều không quan trọng, từ đó giảm số chiều của dữ liệu.

## 2. Giá trị riêng và vector riêng (Eigenvalues and Eigenvectors)

Tham khảo về khái niệm, ý nghĩa và cách tính toán giá trị riêng (eigenvalues) và vector riêng (eigenvectors) trong bài viết [này](/blog/gia-tri-rieng-eigenvalues-va-vector-rieng-eigenvectors).

## 3. Ma trận hiệp phương sai (Covariance matrix)

Tham khảo về khái niệm, ý nghĩa và cách tính toán ma trận hiệp phương sai (covariance matrix) trong bài viết [này](/blog/cac-phan-phoi-xac-suat).

## 4. Các bước của thuật toán

Giả sử ta có một bộ dữ liệu gồm $m$ điểm dữ liệu $x_1, x_2, \dots, x_m \in R^n$.
Ta cần giảm số chiều của dữ liệu từ $n$ xuống $k$ với $k < n$.

Nghĩa là bộ dữ liệu sau khi giảm chiều sẽ có dạng $x_1, x_2, \dots, x_m \in R^k$.

### 4.1. Bước 1: Chuẩn hóa dữ liệu

Tính vector kỳ vọng trên toàn bộ bộ dữ liệu.

$$ \bar{x} = \frac{1}{m} \sum_{i=1}^m x_i $$

trong đó:
- $\bar{x}$ là vector kỳ vọng của bộ dữ liệu.
- $x_i$ là vector dữ liệu thứ $i$ trong bộ dữ liệu.

Chuẩn hoá dữ liệu bằng cách trừ đi giá trị dữ liệu kỳ vọng của bộ dữ liệu.

$$ x^{norm}_i = x_i - \bar{x} $$
trong đó:
- $x^{norm}_i$ là vector dữ liệu thứ $i$ sau khi chuẩn hoá.
- $x_i$ là vector dữ liệu thứ $i$ trong bộ dữ liệu.
- $\bar{x}$ là vector kỳ vọng của bộ dữ liệu.

Khái quát trên toàn bộ bộ dữ liệu, ta có:

$$ X^{norm} = X - \bar{x} $$
trong đó:
- $X^{norm}$ là ma trận dữ liệu sau khi chuẩn hoá.
- $X$ là ma trận dữ liệu ban đầu.
- $\bar{x}$ là vector kỳ vọng của bộ dữ liệu.

### 4.2. Bước 2: Tính ma trận hiệp phương sai

Ta tính ma trận hiệp phương sai của bộ dữ liệu đã chuẩn hoá để giúp đo đạc phương sai giữa các chiều dữ liệu, giúp phát hiện các hướng mà dữ liệu phân tán mạnh nhất.

$$ S = \frac{1}{m} X^{norm} (X^{norm})^T $$
trong đó:
- $S$ là ma trận hiệp phương sai có kích thước $n \times n$.
- $X^{norm}$ là ma trận dữ liệu đã chuẩn hoá.
- $(X^{norm})^T$ là ma trận chuyển vị của ma trận dữ liệu đã chuẩn hoá.
- $m$ là số lượng điểm dữ liệu trong bộ dữ liệu.

### 4.3. Bước 3: Tính các vector riêng và giá trị riêng của ma trận hiệp phương sai


### 4.4. Bước 4: Sắp xếp giá trị riêng và chọn các thành phần chính

- Sắp xếp giá trị riêng giảm dần:
Sắp xếp giá trị riêng theo thứ tự giảm dần để đặt ưu tiên cho các thành phần chính quan trọng nhất.
- Chọn $K$ vector riêng tương ứng với $K$ giá trị riêng lớn nhất.
Các vector này được gọi là các thành phần chính (principal components), giúp tạo thành không gian mới ít chiều hơn cho dữ liệu.
- Tạo ra ma trận chiếu từ $K$ vector riêng đã chọn.

$$
U
= \begin{bmatrix} u_1 & u_2 & \dots & u_k \end{bmatrix}
= \begin{bmatrix}
u^1_1 & u^1_2 & \dots & u^1_k \\
u^2_1 & u^2_2 & \dots & u^2_k \\
\vdots & \vdots & \ddots & \vdots \\
u^n_1 & u^n_2 & \dots & u^n_k
\end{bmatrix}
$$
trong đó:
- $U$ là ma trận chiếu có kích thước là $n \times k$.
- $u_i$ là vector riêng của $S$ có số chiều là $n$.
- $k$ là số chiều của không gian mới.
- $n$ là số chiều của không gian ban đầu.

### 4.5. Bước 5: Chiếu dữ liệu lên không gian mới

Chiếu dữ liệu lên không gian mới bằng cách nhân ma trận chiếu với dữ liệu.

$$ Z = U^T X $$

Trong đó:
- $Z$ là ma trận dữ liệu trong không gian mới có kích thước là $k \times m$.
- $U$ là ma trận chiếu có kích thước là $n \times k$.
- $U^T$ là ma trận chuyển vị của ma trận chiếu $U$ có kích thước là $k \times n$.
- $X$ là ma trận dữ liệu ban đầu có số chiều là $n \times m$.

## 5. Ví dụ minh hoạ

## 6. Ưu và nhược điểm của mô hình

- **Ưu điểm**:
    - Giữ lại được các thông tin quan trọng nhất trong dữ liệu.

- **Nhược điểm**:
    - Đối với các bộ dữ liệu mà vai trò của các đặc trưng là như nhau, việc loại bỏ đi một số đặc trưng sẽ làm mất đi lượng thông tin lớn.
    - Việc tính toán với toàn bộ bộ dữ liệu có thể gây tốn lượng lớn tài nguyên với những bộ dữ liệu lớn.

## 7. Các biến thể nâng cấp của mô hình

- **Kernel PCA (KPCA)**: Sử dụng hàm kernel để biến đổi dữ liệu sang không gian cao hơn, giúp tìm kiếm các thành phần chính phi tuyến tính.
- **Incremental PCA (IPCA)**: Thực hiện PCA theo từng batch, không cần toàn bộ dữ liệu nằm trong RAM.
Giúp dùng được mô hình với tập dữ liệu lớn.
- **Multilinear PCA (MPCA)**: Mở rộng PCA cho dữ liệu có nhiều chiều như hình ảnh, video.
- **...**
