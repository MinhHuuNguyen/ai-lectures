---
slug: machine-learning
time: "11/09/2024"
title: "Machine learning"
description: "Machine Learning (ML) là một phần của trí tuệ nhân tạo (AI) mà chúng ta dùng để xây dựng các mô hình hoặc chương trình máy tính có khả năng tự học từ dữ liệu."
author: "Nguyễn Hữu Minh"
banner_url: "https://tenten.vn/tin-tuc/wp-content/uploads/2023/08/1cG6U1qstYDijh9bPL42e-Q.jpg"
tags:
is_highlight: false
---

# Dimensionality reduction

## 1. Giới thiệu

Các feature vectors trong các bài toán thực tế có thể có số chiều rất lớn, tới vài nghìn.
Ngoài ra, số lượng các điểm dữ liệu cũng thường rất lớn.
Nếu thực hiện lưu trữ và tính toán trực tiếp trên dữ liệu có số chiều cao này thì sẽ gặp khó khăn cả về việc lưu trữ và tốc độ tính toán.

Điều này có thể được gọi là Curse of Dimensionality (Lời nguyền của số chiều – cách nói ám chỉ hiện tượng bùng nổ tổ hợp trong lưu trữ và tính toán theo số chiều của biểu diễn).

Giảm chiều dữ liệu (Dimensionality reduction) là quá trình giảm số chiều của dữ liệu trong không gian đa chiều.
Mục tiêu của quá trình này là làm cho việc xử lý và phân tích dữ liệu trở nên hiệu quả hơn mà không làm mất đi quá nhiều thông tin quan trọng.

Vì vậy, giảm số chiều dữ liệu là một bước quan trọng trong nhiều bài toán.

Đây cũng được coi là một phương pháp nén dữ liệu.

Có hai loại chính của giảm chiều dữ liệu:
- Giảm chiều dữ liệu tuyến tính:
    - Ý tưởng chính của phương pháp này là chiếu dữ liệu từ không gian nhiều chiều sang không gian ít chiều hơn thông qua các phép biến đổi tuyến tính.
    - Thuật toán nổi bật là Principal Component Analysis (PCA).
- Giảm chiều dữ liệu phi tuyến tính:
    - Ý tưởng chính của phương pháp này là tìm ra mối quan hệ giữa các điểm dữ liệu và cố gắng duy trì được mối quan hệ này trên không gian mới có số chiều thấp hơn.
    - Thuật toán nổi bật là t-distributed Stochastic Neighbor Embedding (t-SNE).


## 2. Principal Component Analysis (PCA)

### 2.1. Giới thiệu

Principal Component Analysis (PCA) là một phương pháp giảm chiều dữ liệu tuyến tính được ra đời từ năm 1901 bởi Karl Pearson.

Mục tiêu của PCA:
- Tối ưu hóa phương sai:
PCA tìm kiếm các trục mới (thành phần chính) sao cho phương sai của dữ liệu được tối đa hóa trên các trục này.
Điều này giúp bảo toàn sự biến động lớn nhất trong dữ liệu.
- Giảm số chiều:
PCA giúp giảm số chiều của dữ liệu bằng cách chọn ra các thành phần chính quan trọng nhất để biểu diễn dữ liệu.


### 2.2. Các bước thực hiện

Giả sử ta có một tập dữ liệu gồm $m$ điểm dữ liệu $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_m \in \mathbb{R}^n$.

Ta cần giảm số chiều của dữ liệu từ $n$ xuống $k$ với $k < n$.

Các bước thực hiện của PCA như sau:

#### 2.2.1: Bước 1: Chuẩn hóa dữ liệu

Chuẩn hoá dữ liệu về trung bình bằng cách trừ đi giá trị trung bình của mỗi chiều

$$
\mathbf{x}_i \leftarrow \mathbf{x}_i - \frac{1}{m} \sum_{j=1}^m \mathbf{x}_j
$$

#### 2.2.2: Bước 2: Tính ma trận hiệp phương sai

$$
\mathbf{S} = \frac{1}{m} \sum_{i=1}^m \mathbf{x}_i \mathbf{x}_i^T
$$

#### 2.2.3: Bước 3: Tính các vector riêng và giá trị riêng của ma trận hiệp phương sai

$$
\mathbf{S} \mathbf{u}_i = \lambda_i \mathbf{u}_i
$$

với $\mathbf{u}_i$ là vector riêng của $\mathbf{S}$ và $\lambda_i$ là giá trị riêng của $\mathbf{S}$.

trong đó:
- **Giá trị riêng (Eigenvalues)**:
    - Đo lường độ lớn của phương sai:
    Giá trị riêng đại diện cho độ lớn của phương sai trong mỗi hướng của dữ liệu.
    Càng lớn, càng quan trọng về mặt thống kê.
    - Quyết định độ quan trọng của thành phần chính:
    Các giá trị riêng được sắp xếp theo thứ tự giảm dần, cho biết mức độ quan trọng của từng thành phần chính.
    Các thành phần chính đầu tiên (có giá trị riêng lớn) giữ lại nhiều phương sai hơn.
- **Vector riêng (Eigenvectors)**:
    - Xác định hướng của các thành phần chính:
    Các vector riêng cho biết hướng của các thành phần chính mới trong không gian dữ liệu.
    Mỗi vector riêng tương ứng với một giá trị riêng và xác định hướng tối ưu để biểu diễn phương sai.
    - Không gian chiều mới:
    Vector riêng được sử dụng để tạo ma trận chiếu, chuyển đổi dữ liệu từ không gian ban đầu sang không gian mới của các thành phần chính.
    Các thành phần chính này thường được chọn dựa trên vector riêng tương ứng với các giá trị riêng lớn nhất.

#### 2.2.4: Bước 4: Sắp xếp giả trị riêng và chọn thành phần chính

- Sắp xếp giá trị riêng giảm dần:
Sắp xếp giá trị riêng theo thứ tự giảm dần để đặt ưu tiên cho các thành phần chính quan trọng nhất.
- Chọn số lượng thành phần chính:
Chọn số lượng thành phần chính dựa trên tỷ lệ giữ lại phương sai mong muốn (ví dụ: giữ lại 95% phương sai).

#### 2.2.5: Bước 5: Tạo ma trận chiếu

Tạo ma trận chiếu $\mathbf{W}$ bằng cách chọn $k$ vector riêng đại diện cho $k$ giá trị riêng lớn nhất.

$$
\mathbf{W} = \begin{bmatrix}
\mathbf{|} & \mathbf{|} & \dots & \mathbf{|} \\
\mathbf{|} & \mathbf{|} & \dots & \mathbf{|} \\
\mathbf{u}_1 & \mathbf{u}_2 & \dots & \mathbf{u}_k \\
\mathbf{|} & \mathbf{|} & \dots & \mathbf{|} \\
\mathbf{|} & \mathbf{|} & \dots & \mathbf{|} \\
\end{bmatrix}
$$

Trong đó:
- $\mathbf{W}$ là ma trận chiếu có số chiều là $n \times k$.
- $\mathbf{u}_i$ là vector riêng của $\mathbf{S}$ có số chiều là $n$.
- $k$ là số chiều của không gian mới.

#### 2.2.6: Bước 6: Chiếu dữ liệu lên không gian mới

Chiếu dữ liệu lên không gian mới bằng cách nhân ma trận chiếu với dữ liệu.

$$
\mathbf{Z} = \mathbf{W}^T \mathbf{X}
$$

Trong đó:
- $\mathbf{Z}$ là ma trận kết quả có số chiều là $k$.
- $\mathbf{W}$ là ma trận chiếu có số chiều là $n \times k$.
- $\mathbf{X}$ là ma trận dữ liệu có số chiều là $n \times m$.

## 3. t-distributed Stochastic Neighbor Embedding (t-SNE)

### 3.1. Giới thiệu

t-distributed Stochastic Neighbor Embedding (t-SNE) là một phương pháp giảm chiều dữ liệu phi tuyến tính được ra đời từ năm 2008 bởi Laurens van der Maaten và Geoffrey Hinton.
t-SNE duy trì các cặp điểm dữ liệu gần nhau trong không gian ban đầu và tạo ra các cặp điểm dữ liệu tương tự trong không gian mới.

### 3.2. Các bước thực hiện

#### 3.2.1: Bước 1: Xác định độ tương đồng (khoảng cách) giữa các điểm

Đo lường độ tương đồng giữa các điểm dữ liệu trong không gian nhiều chiều.

Thông thường, sử dụng hàm Gaussian để đo độ tương đồng.

Các điểm gần nhau sẽ có độ tương đồng lớn, trong khi các điểm xa nhau sẽ có độ tương đồng nhỏ.

#### 3.2.2: Bước 2: Tính ma trận xác suất tương đồng

Xác suất tương đồng là xác suất điểm $\mathbf{x}_i$ được chọn làm điểm gần nhất của điểm $\mathbf{x}_j$.

$$
p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}
$$

Trong đó:
- $p_{j|i}$ là xác suất điểm $\mathbf{x}_i$ được chọn làm điểm gần nhất của điểm $\mathbf{x}_j$.
- $\sigma_i$ là độ lớn của Gaussian.

#### 3.2.3: Bước 3: Tính ma trận xác suất tương đồng trong không gian mới

Ta khởi tạo các điểm trong không gian ít chiều và tính ma trận xác suất tương đồng giữa các điểm trong không gian ít chiều tương tự như đã tính trong không gian nhiều chiều.

#### 3.2.4: Bước 4: Tối ưu hàm mất mát

Ta sử dụng hàm loss KL divergence để đo đạc sự khác biệt giữa phân phối xác suất tương đồng trong không gian ban đầu và phân phối xác suất tương đồng trong không gian mới.

Cuối cùng, ta sử dụng gradient descent để tối ưu hàm mất mát này.

## 4. Linear Discriminatory Analysis (LDA)

### 4.1. Giới thiệu

Linear Discriminatory Analysis (LDA) là một phương pháp giảm chiều dữ liệu tuyến tính.

<img src="https://machinelearningcoban.com/assets/29_lda/lda.png" style="width: 800px;"/>

Với ý tưởng của PCA, PCA cho rằng các đặc trưng có phương sai lớn nhất là những đặc trưng quan trọng nhất, tuy nhiên, điều này ko phải lúc nào cũng đúng.
Đối với những bộ dữ liệu mà ta biết label của từng điểm dữ liệu, ta có thể sử dụng LDA để giảm chiều dữ liệu.

LDA cũng tìm các thành phần chính như PCA, tuy nhiên, LDA tìm các thành phần chính sao cho phân bố của các điểm dữ liệu thuộc các lớp khác nhau là tối đa.
Nói cách khác, đặc trưng dữ liệu được giữ lại bởi LDA là những đặc trưng có thể phân biệt được giữa các lớp.

<img src="https://machinelearningcoban.com/assets/29_lda/lda4.png" style="width: 800px;"/>


### 4.2. Các bước thực hiện

#### 4.2.1: Bước 1: Tính ma trận phân tán giữa các lớp (Between-class scatter matrix - SB)

Đối với mỗi lớp, tính toán vector trung bình (trung bình giá trị các đặc trưng) và vector trung bình trên toàn bộ bộ dữ liệu.

Tính ma trận phân tán giữa các lớp (SB), đo lường sự lan rộng giữa các giá trị trung bình của các lớp.

$$
\mathbf{S}_B = \sum_{i=1}^c (\mathbf{m}_i - \mathbf{m})(\mathbf{m}_i - \mathbf{m})^T
$$

Trong đó:
- $\mathbf{S}_B$ là ma trận phân tán giữa các lớp có số chiều là $n \times n$.
- $\mathbf{m}_i$ là vector trung bình của lớp thứ $i$ có số chiều là $n$.
- $\mathbf{m}$ là vector trung bình của toàn bộ bộ dữ liệu có số chiều là $n$.
- $c$ là số lượng lớp.

#### 4.2.2: Bước 2: Tính ma trận phân tán trong lớp (Within-class scatter matrix - SW)

Tính ma trận phân tán trong lớp (SW), đo lường sự lan rộng của các điểm dữ liệu trong cùng một lớp.

$$
\mathbf{S}_W = \sum_{i=1}^c \sum_{\mathbf{x} \in D_i} (\mathbf{x} - \mathbf{m}_i)(\mathbf{x} - \mathbf{m}_i)^T
$$

Trong đó:
- $\mathbf{S}_W$ là ma trận phân tán trong lớp có số chiều là $n \times n$.
- $\mathbf{m}_i$ là vector trung bình của lớp thứ $i$ có số chiều là $n$.
- $D_i$ là tập hợp các điểm dữ liệu thuộc lớp thứ $i$.
- $c$ là số lượng lớp.

#### 4.2.3: Bước 3: Tính giá trị riêng và vector riêng

- **\( S_W \) (Ma trận Phân Tán Trong Lớp):** Đo lường biến động của các điểm dữ liệu trong cùng một lớp. \( S_W \) là tổng của các ma trận phân tán (variance-covariance matrix) của từng lớp.
- **\( S_B \) (Ma trận Phân Tán Giữa Các Lớp):** Đo lường sự khác biệt giữa các trung bình của các lớp. \( S_B \) là ma trận phân tán giữa các lớp.

Nhân \( S_W^{-1}S_B \) giúp "điều chỉnh" sự biến động trong lớp (\( S_W^{-1} \)) và sự khác biệt giữa các lớp (\( S_B \)).
Nếu tỷ lệ giữa \( S_B \) và \( S_W \) lớn, thì ma trận \( S_W^{-1}S_B \) sẽ có các giá trị riêng lớn, tạo ra các vectơ riêng (biến phân loại tuyến tính) mà khi chiếu dữ liệu lên chúng, sự phân loại giữa các lớp trở nên tốt hơn.

Do đó, tính giá trị riêng và vector riêng của ma trận $\mathbf{S}_W^{-1} \mathbf{S}_B$.

#### 4.2.4: Bước 4: Chọn Các Biến Phân Loại Tuyến Tính

Sắp xếp các giá trị riêng theo thứ tự giảm dần và chọn \( k \) giá trị riêng và vector riêng tương ứng. Các vector riêng này trở thành các biến phân loại tuyến tính.
