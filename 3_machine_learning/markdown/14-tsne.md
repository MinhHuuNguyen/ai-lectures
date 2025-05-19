---
time: 08/23/2022
title: Mô hình tSNE
description: Các feature vectors trong các bài toán machine learning thực tế có thể có số chiều rất lớn và số lượng các điểm dữ liệu cũng lớn dần theo thời gian. Điều này có thể được gọi là Curse of Dimensionality, Lời nguyền của số chiều. Trong các thuật toán giảm chiều dữ liệu, t-SNE là một đại diện nổi bật cho phương pháp giảm chiều dữ liệu phi tuyến tính.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/13-pca/banner.png
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Tổng quan

Curse of Dimensionality (Lời nguyền của số chiều) là cách nói ám chỉ hiện tượng bùng nổ tổ hợp trong lưu trữ và tính toán theo số chiều của biểu diễn.

Vấn đề này xảy ra khi ta phải làm việc với các bộ dữ liệu có các feature vectors có số chiều rất lớn, tới vài nghìn và số lượng các điểm dữ liệu rất lớn.
Vì vậy, giảm số chiều dữ liệu là một bước quan trọng trong nhiều bài toán.

Có hai loại chính của mô hình giảm chiều dữ liệu:
- **Giảm chiều dữ liệu tuyến tính**:
Chiếu dữ liệu từ không gian nhiều chiều sang không gian ít chiều hơn thông qua các phép biến đổi tuyến tính.
Đại diện là mô hình Principal Component Analysis (PCA).
- **Giảm chiều dữ liệu phi tuyến tính**:
Tìm ra mối quan hệ giữa các điểm dữ liệu và cố gắng duy trì được mối quan hệ này trên không gian mới có số chiều thấp hơn.
Đại diện là mô hình t-distributed Stochastic Neighbor Embedding (t-SNE).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/14-tsne/idea.png" width="600"/>

t-distributed Stochastic Neighbor Embedding (t-SNE) là một phương pháp giảm chiều dữ liệu phi tuyến tính được ra đời từ năm 2008 bởi Laurens van der Maaten và Geoffrey Hinton.

t-SNE duy trì các cặp điểm dữ liệu gần nhau trong không gian ban đầu và tạo ra các cặp điểm dữ liệu tương tự trong không gian mới.

## 2. Công thức tính khoảng cách KL divergence

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/3-distribution/kl_divergence.png" width="700"/>

Tham khảo về khái niệm, ý nghĩa và cách tính toán giá trị khoảng cách KL divergence trong bài viết [này](/blog/cac-phan-phoi-xac-suat).

## 3. Các bước của thuật toán

Giả sử ta có một bộ dữ liệu gồm $m$ điểm dữ liệu $x_1, x_2, ..., x_m \in R^n$.
Ta cần giảm số chiều của dữ liệu từ $n$ xuống $k$ với $k < n$.

Nghĩa là bộ dữ liệu sau khi giảm chiều sẽ có dạng $x_1, x_2, ..., x_m \in R^k$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/3-distribution/data_3d.png" width="500"/>

### 3.1. Bước 1: Tính ma trận xác suất tương đồng

Xác suất tương đồng là xác suất điểm $x_i$ được chọn làm điểm gần nhất của điểm $x_j$.

Với mỗi điểm dữ liệu $x_i$, t-SNE tính xác suất điều kiện $p_{j|i}$ để điểm $x_j$ là điểm gần nhất của điểm $x_i$.
Ta dùng phân phối Gaussian để tính phương sai của điểm $x_i$.
Ta có công thức tính xác suất điều kiện như sau:

$$ p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)} $$
trong đó:
- $p_{j|i}$ là xác suất điểm $x_i$ được chọn làm điểm gần nhất của điểm $x_j$.
- $\sigma_i$ là phương sai của điểm $x_i$.

Sau đó, ta tính xác suất tương đồng $p_{ij}$ giữa hai điểm $x_i$ và $x_j$ bằng cách đối xứng hóa xác suất điều kiện:

$$ p_{ij} = \frac{p_{j|i} + p_{i|j}}{2m} $$
trong đó:
- $p_{ij}$ là xác suất tương đồng giữa hai điểm $x_i$ và $x_j$.
- $m$ là số lượng điểm dữ liệu trong bộ dữ liệu.
- $p_{i|j}$ là xác suất điểm $x_j$ được chọn làm điểm gần nhất của điểm $x_i$.
- $p_{j|i}$ là xác suất điểm $x_i$ được chọn làm điểm gần nhất của điểm $x_j$.

Các giá trị trên đường chéo của ma trận xác suất tương đồng $P$ là $p_{ii} = 0$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/3-distribution/p_matrixx.png" width="500"/>

Mô hình t-SNE sẽ tính toán ma trận xác suất tương đồng $P$ để đo đạc khoảng cách (sự tương đồng) giữa các điểm dữ liệu trong không gian nhiều chiều.

Lý do sử dụng giá trị xác suất thay vì các công thức tính khoảng cách thông thường như Euclidean distance là vì các giá trị xác suất có thể được sử dụng trong các không gian dữ liệu có số chiều khác nhau.

Nghĩa là, hai điểm dữ liệu có xác suất tương đồng trong không gian nhiều chiều sẽ có xác suất tương đồng giữ nguyên như vậy trong không gian ít chiều và ngược lại.
Trong khi đó, khoảng cách giữa hai điểm dữ liệu trong không gian nhiều chiều không thể sẽ cần phải được tính toán lại trong không gian ít chiều.

### 3.2. Bước 2: Khởi tạo ma trận xác suất tương đồng trong không gian ít chiều

Với mỗi điểm dữ liệu $x_i \in R^n$, t-SNE sẽ khởi tạo một điểm dữ liệu $y_i \in R^k$ trong không gian ít chiều với $k < n$.

Từ các điểm dữ liệu $y_i$, t-SNE sẽ tính ma trận xác suất tương đồng $Q$ giữa các điểm dữ liệu trong không gian ít chiều.

$$ q_{j|i} = \frac{\exp(-\|y_i - y_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|y_i - y_k\|^2 / 2\sigma_i^2)} $$
trong đó:
- $q_{j|i}$ là xác suất điểm $y_i$ được chọn làm điểm gần nhất của điểm $y_j$.
- $\sigma_i$ là phương sai của điểm $y_i$.
- $y_i$ là điểm dữ liệu thứ $i$ trong không gian ít chiều.
- $y_j$ là điểm dữ liệu thứ $j$ trong không gian ít chiều.

Trong không gian ít chiều, t-SNE khởi tạo các điểm dữ liệu $y_i$ với phương sai $\sigma_i = \frac{1}{\sqrt{2}}$.
Từ đó, ta có công thức tính xác suất tương đồng giữa hai điểm dữ liệu $y_i$ và $y_j$ như sau:

$$ q_{j|i} = \frac{\exp(-\|y_i - y_j\|^2)}{\sum_{k \neq i} \exp(-\|y_i - y_k\|^2)} $$

Các giá trị trên đường chéo của ma trận xác suất tương đồng $Q$ là $q_{ii} = 0$.

### 3.3. Bước 3: Tối ưu hàm mất mát

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/3-distribution/train_progress.gif" width="1200"/>

Hàm mất mát của t-SNE được định nghĩa bằng hàm KL divergence giữa ma trận xác suất tương đồng $P$ trong không gian nhiều chiều và ma trận xác suất tương đồng $Q$ trong không gian ít chiều.
Ta có công thức tính KL divergence như sau:

$$ L = KL(P || Q) $$
trong đó:
- $L$ là hàm mất mát của t-SNE.
- $KL(P || Q)$ là KL divergence giữa ma trận xác suất tương đồng $P$ trong không gian nhiều chiều và ma trận xác suất tương đồng $Q$ trong không gian ít chiều.
- $P$ là ma trận xác suất tương đồng trong không gian nhiều chiều.
- $Q$ là ma trận xác suất tương đồng trong không gian ít chiều.

Khi ta cực tiểu hoá hàm mất mát này, t-SNE sẽ tìm ra các điểm dữ liệu $y_i$ trong không gian ít chiều sao cho xác suất tương đồng giữa các điểm dữ liệu trong không gian nhiều chiều và không gian ít chiều là giống nhau.

Đến đây, ta sẽ sử dụng thuật toán gradient descent để tối ưu hàm mất mát này.

## 4. Ưu và nhược điểm của mô hình

- **Ưu điểm**:
    - **Trực quan hóa dữ liệu tốt**: Rất hiệu quả trong việc trực quan hóa dữ liệu có chiều cao (ví dụ như embedding), thường dùng để biểu diễn dữ liệu 2D hoặc 3D.
    - **Bảo toàn cấu trúc cục bộ**: t-SNE giữ được các cụm gần nhau, làm nổi bật cấu trúc phân cụm trong dữ liệu.
    - **Hiệu quả với dữ liệu phi tuyến**: Xử lý tốt các mối quan hệ phi tuyến.

- **Nhược điểm**:
    - **Tính toán chậm**: Đặc biệt với tập dữ liệu lớn, t-SNE tiêu tốn nhiều tài nguyên.
    - **Không bảo toàn cấu trúc toàn cục**: Mối quan hệ giữa các cụm (clusters) có thể không phản ánh đúng khoảng cách thực sự.
    - **Không dùng để huấn luyện mô hình khác**: t-SNE chủ yếu chỉ dùng để trực quan hóa, không thể sử dụng như feature đầu vào cho các mô hình khác.
    - **Kết quả không ổn định**: Mỗi lần chạy có thể cho kết quả khác nhau do tính ngẫu nhiên trong khởi tạo.

## 5. Các biến thể nâng cấp của mô hình

- **Barnes-Hut t-SNE** và **FFT-accelerated Interpolation-based t-SNE (FIt-SNE)**: Tăng tốc quá trình tính toán giữa các điểm dữ liệu bằng việc sự dụng thuật toán Barnes-Hut (đối với Barnes-Hut t-SNE) và phép nội suy và biến đổi Fourier nhanh FFT (đối với FIt-SNE).
- **Parametric t-SNE**: Thay vì tối ưu trực tiếp toạ độ điểm, mô hình huấn luyện một mạng neural để học ánh xạ từ không gian gốc sang không gian thấp chiều.
Từ đó, có thể sử dụng kết quả của quá trình giảm chiều dữ liệu để huấn luyện các mô hình khác.
- **...**
