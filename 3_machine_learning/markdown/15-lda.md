---
time: 09/06/2022
title: 
description:
banner_url:
tags: [machine-learning]
is_highlight: false
is_published: false
---

## 4. Linear Discriminatory Analysis (LDA)

### 4.1. Giới thiệu

Linear Discriminatory Analysis (LDA) là một phương pháp giảm chiều dữ liệu tuyến tính.

<img src="https://machinelearningcoban.com/assets/29_lda/lda.jpeg" style="width: 1200px;"/>

Với ý tưởng của PCA, PCA cho rằng các đặc trưng có phương sai lớn nhất là những đặc trưng quan trọng nhất, tuy nhiên, điều này ko phải lúc nào cũng đúng.
Đối với những bộ dữ liệu mà ta biết label của từng điểm dữ liệu, ta có thể sử dụng LDA để giảm chiều dữ liệu.

LDA cũng tìm các thành phần chính như PCA, tuy nhiên, LDA tìm các thành phần chính sao cho phân bố của các điểm dữ liệu thuộc các lớp khác nhau là tối đa.
Nói cách khác, đặc trưng dữ liệu được giữ lại bởi LDA là những đặc trưng có thể phân biệt được giữa các lớp.

<img src="https://machinelearningcoban.com/assets/29_lda/lda4.jpeg" style="width: 1200px;"/>


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
