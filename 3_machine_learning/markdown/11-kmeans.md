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

Phân cụm (clustering) là bài toán học máy không giám sát nhằm nhóm các điểm dữ liệu thành những cụm sao cho các điểm trong cùng cụm tương đồng với nhau hơn so với các điểm ở cụm khác.

K-means Clustering là một trong những thuật toán phân cụm đơn giản và phổ biến nhất
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

## 3. Công thức tính khoảng cách Euclidean

Khoảng cách Euclidean là khoảng cách phổ biến nhất được sử dụng trong KNN. Nó được tính bằng công thức:

$$ d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} $$

trong đó:
- $x_i$ và $y_i$ là các thành phần của vector $x$ và $y$.

Khoảng cách Euclidean thường được sử dụng trong đa số các trường hợp.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/7-knn/euclidean.png" style="width: 500px;"/>

## 4. Tối ưu trong mô hình K-means

Xét bộ dữ liệu $X = \{x_1, x_2, \ldots, x_n\}$ với $n$ điểm dữ liệu và $K$ cụm, tâm của các cụm được ký hiệu là $C = \{c_1, c_2, \ldots, c_K\}$ với $c_j$ là tâm của cụm $j$.

Với mỗi điểm dữ liệu $x_i$ thuộc cụm $k$ với tâm là $c_k$ nào đó, label của điểm dữ liệu $x_i$ được xác định là $y_i = k$.
Ta có thể biểu diễn label của điểm dữ liệu $x_i$ bằng một vector nhị phân $y_i$ với kích thước $K$.

$$ y_i = \begin{bmatrix} y_{i1} & \ldots & y_{ik} \ldots & y_{iK} \end{bmatrix}^T $$

$$ y_i = \begin{bmatrix} 0 & \ldots & 1 \ldots & 0 \end{bmatrix}^T $$

trong đó, $1$ nằm ở vị trí $k$ tương ứng với cụm mà điểm dữ liệu $x_i$ thuộc về.

Ta có khoảng cách giữa điểm dữ liệu $x_i$ và tâm của một cụm nào đó:

$$ d(x_i, c_j) = (x_{i} - c_{j})^2 $$

và tổng khoảng cách giữa điểm dữ liệu $x_i$ và tất cả các tâm cụm:

$$ d(x_i, C) = \sum_{j=1}^{K} (x_{i} - c_{j})^2 $$

Mục tiêu chính của K-means là cực tiểu hoá khoảng cách giữa các điểm dữ liệu và tâm của cụm mà chúng thuộc về.
Từ đó, ta xây dựng hàm mục tiêu của K-means trên toàn bộ tập dữ liệu $X$:

$$ L(Y, C) = \sum_{i=1}^{n} \sum_{j=1}^{K} y_{ij} d(x_i, c_j) = Y d(X, C) $$

Phát biểu lại hàm mục tiêu trên: "Ta cần tìm tâm của các cụm $C$ và cụm mà các điểm dữ liệu $X$ thuộc về $Y$ sao cho tổng khoảng cách giữa các điểm dữ liệu và tâm của cụm mà chúng thuộc về là nhỏ nhất".
Hàm mục tiêu trên có thể được hiểu là tổng khoảng cách giữa các điểm dữ liệu và tâm của cụm mà chúng thuộc về.

Để tối ưu hàm mục tiêu trên, ta cần lần lượt: Giữ nguyên giá trị C và tối ưu giá trị Y, sau đó giữ nguyên giá trị Y và tối ưu giá trị C.

### 4.1. Giữ nguyên giá trị $C$ và tối ưu giá trị $Y$

Khi giữ nguyên giá trị $C$, hàm số $L(Y, C)$ trở thành hàm một biến với biến $Y$.

Việc tối ưu giá trị $Y$ nghĩa là với các tâm cụm đã biết, ta sẽ gán các điểm dữ liệu $X$ vào cụm có tâm là $C$ mà nó gần nhất với tâm cụm đó.

$$ Y^* = \arg \min L(Y) = \arg \min_{j} \sum_{i=1}^{n} \sum_{j=1}^{K} y_{ij} d(x_i, c_j) $$

Nói cách khác, giá trị $y_{ij}$ chỉ được phép bằng $1$ đối với tâm $c_j$ mà điểm dữ liệu $x_i$ gần nhất, còn lại bằng $0$ với các tâm khác.

### 4.2. Giữ nguyên giá trị $Y$ và tối ưu giá trị $C$

Khi giữ nguyên giá trị $Y$, hàm số $L(Y, C)$ trở thành hàm một biến với biến $C$.

Việc tối ưu giá trị C nghĩa là với các cụm của các điểm dữ liệu đã biết, ta sẽ tính toán lại các tâm cụm $C$ sao cho tổng khoảng cách giữa các điểm dữ liệu và tâm của cụm mà chúng thuộc về là nhỏ nhất.

$$ C^* = \arg \min L(C) = \arg \min_{c_j} \sum_{i=1}^{n} \sum_{j=1}^{K} y_{ij} d(x_i, c_j) $$

Nói cách khác, với các điểm dữ liệu đã thuộc cùng một cụm, ta tính toán lại tâm cụm sao cho tổng khoảng cách giữa các điểm dữ liệu trong cụm và tâm cụm là nhỏ nhất.

Xét 1 cụm với $m$ điểm dữ liệu, ta có khoảng cách giữa các điểm dữ liệu trong cụm và tâm cụm đó:

$$ d(X, c) = \sum_{i=1}^{m} d(x_i, c) = \sum_{i=1}^{m} (x_{i} - c)^2 $$

Đây là một hàm số với biến $c$, ta cần tìm $c$ để hàm số này đạt giá trị nhỏ nhất.
Giải phương trình đạo hàm bằng 0:

$$ \frac{\partial}{\partial c} \sum_{i=1}^{m} (x_{i} - c_{j})^2 = 0 $$

$$ \sum_{i=1}^{m} 2(x_{i} - c_{j}) = 0 $$

$$ c_{j} = \frac{1}{m} \sum_{i=1}^{m} x_{i} $$

Từ đó, ta có thể tính toán lại tâm cụm $c_j$ bằng cách lấy trung bình của tất cả các điểm dữ liệu trong cụm đó.

## 5. Ưu điểm và nhược điểm của mô hình

- Ưu điểm:
    - Đơn giản, dễ hiểu và dễ triển khai.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/normal_9.gif" style="width: 400px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/slow_17.gif" style="width: 400px;"/>

- Nhược điểm:
    - Cần phải xác định số cụm K trước khi chạy thuật toán.
    - Kết quả phân cụm có thể khác nhau giữa các lần chạy do việc chọn ngẫu nhiên các tâm cụm ban đầu.
    - Không thể phân cụm các dữ liệu không có hình dạng cầu (circular shape) hoặc không đồng nhất (non-convex shape).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/non_convex_animation.gif" style="width: 400px;"/>

## 6. Phương pháp lựa chọn số lượng cụm

Việc lựa chọn số lượng cụm K là điều kiện tiên quyết để chạy thuật toán K-means Clustering.
Tuy nhiên, việc xác định số lượng cụm K là một vấn đề khó khăn và không có một quy tắc chung nào để xác định số lượng cụm K.

### 6.1. Phương pháp Elbow

Phương pháp Elbow dựa trên việc quan sát đồ thị thể hiện tổng bình phương khoảng cách nội cụm (WCSS – Within-Cluster Sum of Squares) theo số cụm K.

WCSS được tính bằng:

$$ WCSS = \sum_{k=1}^{K} \sum_{i=1}^{n_k} (x_i - c_k)^2 $$

Các bước thực hiện như sau:
- **Bước 1:** Chọn một khoảng giá trị cho K (ví dụ từ 1 đến 10).
- **Bước 2:** Chạy thuật toán K-means với các giá trị K trong khoảng đã chọn.
- **Bước 3:** Tính toán WCSS cho mỗi giá trị K.
- **Bước 4:** Vẽ đồ thị WCSS theo K.
- **Bước 5:** Tìm điểm "khuỷu" (elbow) trên đồ thị, tức là điểm mà sau đó WCSS giảm rất nhẹ.
- **Bước 6:** Chọn giá trị K tại điểm khuỷu này làm số lượng cụm tối ưu.

```python
inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)
```

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/elbow_method.png" style="width: 400px;"/>

### 6.2. Phương pháp Silhouette

Hệ số Silhouette của mỗi điểm đo mức độ gắn kết và tách biệt của nó so với các cụm:

$$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$

trong đó:
- $a(i)$ là khoảng cách trung bình từ điểm $i$ đến tất cả các điểm khác trong cùng cụm của $i$.
- $b(i)$ là khoảng cách trung bình từ điểm $i$ đến tất cả các điểm trong cụm khác gần nhất của $i$.
- $s(i)$ có giá trị từ -1 đến 1.
    - Nếu $s(i)$ gần 1, điểm $i$ được phân cụm tốt.
    - Nếu $s(i)$ gần 0, điểm $i$ nằm ở ranh giới giữa hai cụm.
    - Nếu $s(i)$ gần -1, điểm $i$ có thể đã được phân cụm sai (cụm của $i$ không phải là cụm gần nhất).

```python
from sklearn.metrics import silhouette_score
silhouette_scores = []
K = range(2, 11)  # note: silhouette undefined for k=1
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
```

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/silhouette_method.png" style="width: 400px;"/>

## 7. Các biến thể nâng cấp của mô hình K-means Clustering

### 7.1. K-means++

K-means++ là một biến thể của K-means Clustering nhằm cải thiện việc chọn tâm cụm ban đầu.
K-means++ sử dụng một phương pháp chọn tâm cụm ban đầu thông minh hơn so với việc chọn ngẫu nhiên.

Thay vì chọn ngẫu nhiên K điểm dữ liệu làm tâm cụm ban đầu, K-means++ chọn tâm cụm đầu tiên ngẫu nhiên từ tập dữ liệu, sau đó chọn các tâm cụm tiếp theo dựa trên khoảng cách từ các điểm dữ liệu đến tâm cụm đã chọn.
Điều này giúp đảm bảo rằng các tâm cụm ban đầu được phân bố đều hơn trong không gian dữ liệu, từ đó cải thiện khả năng hội tụ của thuật toán.
K-means++ giúp giảm thiểu khả năng chọn các tâm cụm gần nhau, từ đó giúp thuật toán hội tụ nhanh hơn và đạt được kết quả tốt hơn.

Các bước thực hiện như sau:
- **Bước 1:** Chọn ngẫu nhiên một điểm dữ liệu từ tập dữ liệu làm tâm cụm đầu tiên.
- **Bước 2:** Tính khoảng cách từ tất cả các điểm dữ liệu còn lại đến tâm cụm đầu tiên.
- **Bước 3:** Chọn ngẫu nhiên một điểm dữ liệu từ tập dữ liệu còn lại với xác suất tỷ lệ với bình phương khoảng cách đến tâm cụm đầu tiên.
- **Bước 4:** Lặp lại Bước 2 và Bước 3 cho đến khi chọn đủ K tâm cụm.

### 7.2. Mini-batch K-means

Mini-batch K-means là một biến thể của K-means Clustering nhằm cải thiện hiệu suất và tốc độ của thuật toán khi làm việc với tập dữ liệu lớn.

Thay vì sử dụng toàn bộ tập dữ liệu để cập nhật tâm cụm trong mỗi vòng lặp, Mini-batch K-means sử dụng một tập con ngẫu nhiên (mini-batch) của dữ liệu để cập nhật tâm cụm.
Điều này giúp giảm thiểu thời gian tính toán và bộ nhớ cần thiết cho thuật toán, đồng thời vẫn giữ được chất lượng phân cụm tương đối tốt.

Các bước thực hiện như sau:
- **Bước 1:** Chọn một kích thước mini-batch (ví dụ: 100 điểm dữ liệu).
- **Bước 2:** Chọn ngẫu nhiên một mini-batch từ tập dữ liệu.
- **Bước 3:** Chạy thuật toán K-means trên mini-batch để cập nhật tâm cụm và gán các điểm dữ liệu trong mini-batch vào các cụm tương ứng.
- **Bước 4:** Lặp lại Bước 2 và Bước 3 cho đến khi đạt được số vòng lặp tối đa hoặc hội tụ.

### 7.3. Các biến thể khác

- **Bisecting K-Means:** Tối ưu chất lượng cụm và khắc phục sự phụ thuộc vào k.
Phân chia cụm lớn nhất thành 2 cụm nhỏ, lặp lại đến khi đạt đủ số cụm.
- **Fuzzy C-Means (Soft K-Means):** Mỗi điểm dữ liệu có thể thuộc về nhiều cụm với xác suất khác nhau, không chỉ một cụm duy nhất.
Gán xác suất (mức độ thành viên) cho từng điểm với mỗi cụm.
- **Kernel K-Means:** Xử lý dữ liệu phi tuyến.
Sử dụng kernel function để ánh xạ dữ liệu sang không gian mới.
Tách được cụm có hình dạng phức tạp (phi tuyến).
- **X-Means:** Tự động xác định số lượng cụm K.
Mở rộng K-Means bằng cách kiểm tra chia thêm cụm dựa trên BIC (Bayesian Information Criterion).
