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

Có hai loại chính của mô hình giảm chiều dữ liệu:
- **Giảm chiều dữ liệu tuyến tính**:
Chiếu dữ liệu từ không gian nhiều chiều sang không gian ít chiều hơn thông qua các phép biến đổi tuyến tính.
Đại diện là mô hình Principal Component Analysis (PCA).
- **Giảm chiều dữ liệu phi tuyến tính**:
Tìm ra mối quan hệ giữa các điểm dữ liệu và cố gắng duy trì được mối quan hệ này trên không gian mới có số chiều thấp hơn.
Đại diện là mô hình t-distributed Stochastic Neighbor Embedding (t-SNE).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/13-pca/idea.png" width="400"/>

Principal Component Analysis (PCA) là một phương pháp giảm chiều dữ liệu tuyến tính được ra đời từ năm 1901 bởi Karl Pearson.

PCA giúp giảm số chiều của dữ liệu bằng cách tìm kiếm các thành phần chính (principal components) mà giữ lại nhiều phương sai nhất trong dữ liệu.
Điều này giúp bảo toàn sự biến động lớn nhất trong dữ liệu.

Nói cách khác, PCA sẽ đánh giá mức độ quan trọng của từng chiều dữ liệu và loại bỏ những chiều không quan trọng, từ đó giảm số chiều của dữ liệu.

## 2. Giá trị riêng và vector riêng (Eigenvalues và Eigenvectors)

Tham khảo về khái niệm, ý nghĩa và cách tính toán giá trị riêng (eigenvalues) và vector riêng (eigenvectors) trong bài viết [này](/blog/gia-tri-rieng-eigenvalues-va-vector-rieng-eigenvectors).

## 3. Ma trận hiệp phương sai (Covariance matrix)

Tham khảo về khái niệm, ý nghĩa và cách tính toán ma trận hiệp phương sai (covariance matrix) trong bài viết [này](/blog/cac-phan-phoi-xac-suat).

## 4. Các bước của thuật toán

Giả sử ta có một bộ dữ liệu gồm $m$ điểm dữ liệu $x_1, x_2, ..., x_m \in R^n$.
Ta cần giảm số chiều của dữ liệu từ $n$ xuống $k$ với $k < n$.

Nghĩa là bộ dữ liệu sau khi giảm chiều sẽ có dạng $x_1, x_2, ..., x_m \in R^k$.

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
- $X^{norm}$ là ma trận dữ liệu sau khi chuẩn hoá với mỗi hàng là một vector dữ liệu, mỗi cột là một đặc trưng dữ liệu.
- $X$ là ma trận dữ liệu ban đầu với mỗi hàng là một vector dữ liệu, mỗi cột là một đặc trưng dữ liệu.
- $\bar{x}$ là vector kỳ vọng của bộ dữ liệu.

### 4.2. Bước 2: Tính ma trận hiệp phương sai

Ta tính ma trận hiệp phương sai của bộ dữ liệu đã chuẩn hoá để giúp đo đạc phương sai giữa các chiều dữ liệu, giúp phát hiện các hướng mà dữ liệu phân tán mạnh nhất.

$$ S = \frac{1}{m} (X^{norm})^T X^{norm} $$
trong đó:
- $S$ là ma trận hiệp phương sai có kích thước $n \times n$.
- $X^{norm}$ là ma trận dữ liệu đã chuẩn hoá.
- $(X^{norm})^T$ là ma trận chuyển vị của ma trận dữ liệu đã chuẩn hoá.
- $m$ là số lượng điểm dữ liệu trong bộ dữ liệu.

### 4.3. Bước 3: Tính các vector riêng và giá trị riêng của ma trận hiệp phương sai

Từ ma trận hiệp phương sai $S$, ta tính các vector riêng và giá trị riêng của ma trận hiệp phương sai.
Ta thu được các vector riêng $u_i$ và giá trị riêng $\lambda_i$ của ma trận hiệp phương sai $S$.

$$ S u_i = \lambda_i u_i $$
trong đó:
- $S$ là ma trận hiệp phương sai có kích thước $n \times n$.
- $u_i$ là vector riêng thứ $i$ của ma trận hiệp phương sai $S$.
- $\lambda_i$ là giá trị riêng thứ $i$ của ma trận hiệp phương sai $S$.

### 4.4. Bước 4: Sắp xếp giá trị riêng và chọn các thành phần chính

- Sắp xếp giá trị riêng giảm dần:
Sắp xếp giá trị riêng theo thứ tự giảm dần để đặt ưu tiên cho các thành phần chính quan trọng nhất.
- Chọn $K$ vector riêng tương ứng với $K$ giá trị riêng lớn nhất.
Các vector này được gọi là các thành phần chính (principal components), giúp tạo thành không gian mới ít chiều hơn cho dữ liệu.
- Tạo ra ma trận chiếu từ $K$ vector riêng đã chọn.

$$
U
= \begin{bmatrix} u_1 & u_2 & ... & u_k \end{bmatrix}
= \begin{bmatrix}
u^1_1 & u^1_2 & ... & u^1_k \\
u^2_1 & u^2_2 & ... & u^2_k \\
\vdots & \vdots & \ddots & \vdots \\
u^n_1 & u^n_2 & ... & u^n_k
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

Xét bộ dữ liệu gồm 4 điểm dữ liệu trong không gian 2 chiều như sau:

| Điểm dữ liệu | X1 | X2 | X3 | X4 |
| ------------ | -- | -- | -- | -- |
| A            | 2  | 0  | 0  | 4  |
| B            | 7  | 9  | 1  | 9  |
| C            | 1  | 1  | 5  | 1  |
| D            | 8  | 1  | 1  | 2  |

### 5.1. Bước 1: Chuẩn hóa dữ liệu

Tính vector kỳ vọng trên toàn bộ bộ dữ liệu.

$$ \bar{x} = \frac{1}{4} (A + B + C + D) $$
$$ \bar{x} = \frac{1}{4} ([2, 0, 0, 4] + [7, 9, 1, 9] + [1, 1, 5, 1] + [8, 1, 1, 2]) $$
$$ \bar{x} = [4.5, 2.75, 1.75, 4] $$

Chuẩn hoá dữ liệu bằng cách trừ đi giá trị dữ liệu kỳ vọng của bộ dữ liệu.

$$ X^{norm} = X - \bar{x} $$
$$
X^{norm} = \begin{bmatrix}
-2.5 & -2.75 & -1.75 & 0 \\ 2.5 & 6.25 & -0.75 & 5 \\ -3.5 & -1.75 & 3.25 & -3 \\ 3.5 & -1.75 & -0.75 & -2
\end{bmatrix}
$$

### 5.2. Bước 2: Tính ma trận hiệp phương sai

Ma trận hiệp phương sai của bộ dữ liệu đã chuẩn hoá là:

$$ S = \frac{1}{4} (X^{norm})^T X^{norm} $$
$$ S = \frac{1}{4} \begin{bmatrix}
-2.5 & 2.5 & -3.5 & 3.5 \\ -2.75 & 6.25 & -1.75 & -1.75 \\ -1.75 & -0.75 & 3.25 & -0.75 \\ 0 & 5 & -3 & -2
\end{bmatrix} \begin{bmatrix}
-2.5 & -2.75 & -1.75 & 0 \\ 2.5 & 6.25 & -0.75 & 5 \\ -3.5 & -1.75 & 3.25 & -3 \\ 3.5 & -1.75 & -0.75 & -2
\end{bmatrix} $$
$$
S = \begin{bmatrix} 9.25 & 5.625 & -2.875 & 4 \\ 5.625 & 13.1875 & -1.0625 & 10 \\ -2.875 & -1.0625 & 3.6875 & -3 \\ 4 & 10 & -3 & 9.5 \end{bmatrix}
$$

### 5.3. Bước 3: Tính các vector riêng và giá trị riêng của ma trận hiệp phương sai

Ta tính được các vector riêng và giá trị riêng của ma trận hiệp phương sai $S$.

Các giá trị riêng là:

$$ \lambda_1 = 25.23, \lambda_2 = 6.76, \lambda_3 = 3.64, \lambda_4 = 0 $$

Các vector riêng tương ứng là:

$$ u_1 = [0.42, 0.81, 0.36, 0.21] $$
$$ u_2 = [0.69, -0.37, 0.34, -0.53] $$
$$ u_3 = [-0.17, -0.38, 0.74, 0.53] $$
$$ u_4 = [0.57, -0.26, -0.45, 0.63] $$

### 5.4. Bước 4: Sắp xếp giá trị riêng và chọn các thành phần chính

Ta sắp xếp các giá trị riêng theo thứ tự giảm dần và chọn 2 vector riêng tương ứng với 2 giá trị riêng lớn nhất.
Hai giá trị riêng lớn nhất là $\lambda_1 = 25.23$ và $\lambda_2 = 6.76$ với các vector riêng tương ứng là $u_1$ và $u_2$.

Ma trận chiếu $U$ sẽ có dạng:
$$
U = \begin{bmatrix} u_1 & u_2 \end{bmatrix} = \begin{bmatrix} 0.42 & 0.69 \\ 0.81 & -0.37 \\ 0.36 & 0.34 \\ 0.21 & -0.53 \end{bmatrix}
$$

### 5.5. Bước 5: Chiếu dữ liệu lên không gian mới

Cuối cùng, ta chiếu dữ liệu lên không gian mới bằng cách nhân ma trận chiếu $U$ với dữ liệu đã chuẩn hoá $X^{norm}$.

$$ Z = U^T X $$
$$ Z = \begin{bmatrix} 0.42 & 0.81 & 0.36 & 0.21 \\ 0.69 & -0.37 & 0.34 & -0.53 \end{bmatrix} \begin{bmatrix} 2 & 0 & 0 & 4 \\ 7 & 9 & 1 & 9 \\ 1 & 1 & 5 & 1 \\ 8 & 1 & 1 & 2 \end{bmatrix} $$
$$ Z = \begin{bmatrix} 8.55 & 7.86 & 2.82 & 9.75 \\ -5.11 & -3.52 & 0.8 & -1.29 \end{bmatrix} $$

Từ ma trận dữ liệu ban đầu:

| Điểm dữ liệu | X1 | X2 | X3 | X4 |
| ------------ | -- | -- | -- | -- |
| A            | 2  | 0  | 0  | 4  |
| B            | 7  | 9  | 1  | 9  |
| C            | 1  | 1  | 5  | 1  |
| D            | 8  | 1  | 1  | 2  |

Ta đã thu được ma trận dữ liệu mới sẽ gồm các điểm dữ liệu như sau:

| Điểm dữ liệu | Z1   | Z2   |
| ------------ | ---- | ---- |
| A            | 8.55 | -5.11 |
| B            | 7.86 | -3.52 |
| C            | 2.82 | 0.8  |
| D            | 9.75 | -1.29 |

## 6. Ưu và nhược điểm của mô hình

- **Ưu điểm**:
    - **Đơn giản và hiệu quả**: PCA là phương pháp tuyến tính, dễ hiểu, dễ triển khai và tính toán nhanh.
    - **Bảo toàn phương sai tối đa**: PCA giữ lại nhiều thông tin nhất có thể bằng cách tối đa hóa phương sai.
    - **Có thể sử dụng trong mô hình hóa**: Các thành phần chính (principal components) có thể được dùng trong các mô hình học máy khác như hồi quy, phân loại.

- **Nhược điểm**:
    - **Tuyến tính**: Không nắm bắt được các mối quan hệ phi tuyến trong dữ liệu.
    - **Bộ dữ liệu phức tạp**: Đối với các bộ dữ liệu mà vai trò của các đặc trưng là như nhau, việc loại bỏ đi một số đặc trưng sẽ làm mất đi lượng thông tin lớn.
    - **Hiệu năng tính toán**: Việc tính toán với toàn bộ bộ dữ liệu có thể gây tốn lượng lớn tài nguyên với những bộ dữ liệu lớn.

## 7. Các biến thể nâng cấp của mô hình

- **Kernel PCA (KPCA)**: Sử dụng hàm kernel để biến đổi dữ liệu sang không gian cao hơn, giúp tìm kiếm các thành phần chính phi tuyến tính.
- **Incremental PCA (IPCA)**: Thực hiện PCA theo từng batch, không cần toàn bộ dữ liệu nằm trong RAM.
Giúp dùng được mô hình với tập dữ liệu lớn.
- **Multilinear PCA (MPCA)**: Mở rộng PCA cho dữ liệu có nhiều chiều như hình ảnh, video.
- **...**
