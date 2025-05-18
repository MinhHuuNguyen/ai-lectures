---
time: 08/16/2022
title: Mô hình DBSCAN
description: Khác với K-means Clustering, mô hình phân cụm DBSCAN Clustering không yêu cầu số lượng cụm cần phân chia trước. Trong bài viết này, chúng ta sẽ tìm hiểu về mô hình DBSCAN Clustering, mô hình giúp phân chia dữ liệu thành các cụm dựa trên mật độ của chúng.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/banner.png
tags: [machine-learning]
is_highlight: false
is_published: true
---

| Kết quả với DBSCAN  | Kết quả với K-Means |
|---------------------|---------------------|
| <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/example_blobs_dbscan.gif" width="400"/> | <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/example_blobs_kmeans.gif" width="400"/> |

## 1. Tổng quan

Phân cụm (clustering) là bài toán học máy không giám sát nhằm nhóm các điểm dữ liệu thành những cụm sao cho các điểm trong cùng cụm tương đồng với nhau hơn so với các điểm ở cụm khác.

Khác với K-means Clustering, mô hình phân cụm DBSCAN Clustering không yêu cầu số lượng cụm cần phân chia trước.
Thay vào đó, mô hình này sử dụng mật độ của các điểm dữ liệu để xác định các cụm.

| Kết quả với DBSCAN  | Kết quả với K-Means |
|---------------------|---------------------|
| <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/example_circles_dbscan.gif" width="400"/> | <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/example_circles_kmeans.gif" width="400"/> |

DBSCAN là viết tắt của **Density-Based Spatial Clustering of Applications with Noise** - tạm dịch là Phân cụm không gian dựa trên mật độ với nhiễu.
DBSCAN là một trong những thuật toán phân cụm phổ biến nhất trong học máy không giám sát và được sử dụng nhiều trong các bài toán Định danh khuôn mặt (Face Recognition) hay Nhận diện bất thường (Anomaly Detection).

| Kết quả với DBSCAN  | Kết quả với K-Means |
|---------------------|---------------------|
| <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/example_moons_dbscan.gif" width="400"/> | <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/example_moons_kmeans.gif" width="400"/> |

## 2. Các khái niệm được định nghĩa trong mô hình DBSCAN

Trước khi đi vào các bước hoạt động của thuật toán DBSCAN, chúng ta cần tìm hiểu một số khái niệm được định nghĩa trong mô hình này.
Các khái niệm này sẽ giúp chúng ta hiểu rõ hơn về cách mà thuật toán hoạt động.

Điều kiện tiên quyết để bắt đầu thuật toán là ta cần lựa chọn 2 tham số đầu vào cho mô hình DBSCAN là $eps$ và $minSample$ (hoặc %minPts$).

### 2.1. Epsilon Neighborhood

Cho $D$ là bộ dữ liệu cần được phân cụm, **vùng lận cận epsilon** (Epsilon Neighborhood) của điểm dữ liệu $p$, ký hiệu là $N_{eps}(p)$, được định nghĩa là tập hợp tất cả các điểm dữ liệu $q$ trong $D$ mà khoảng cách giữa $p$ và $q$ nhỏ hơn epsilon.

$$N_{eps}(p) = \{q \in D | dist(p, q) \leq eps\}$$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/epsilon.png" width="400"/>

### 2.2. Core Point

Trong bộ dữ liệu $D$, một điểm dữ liệu $p$ được gọi là **core point** nếu nó có số lượng "người hàng xóm" trong **vùng lận cận epsilon** của nó lớn hơn hoặc bằng giá trị $minSample$ (hoặc $minPts$).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/core_point.png" width="800"/>

### 2.3. Directly Density Reachable

Với hai giá trị $eps$ và $minSample$, một điểm dữ liệu $q$ được gọi là **directly density reachable** từ một điểm dữ liệu $p$ nếu $p$ là một **core point** và $q$ nằm trong vùng lận cận epsilon của $p$.
- $ q \in N_{eps}(p)$
- $ |N_{eps}(p)| \geq min_sample$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/directly_density_reachable.png" width="400"/>

### 2.4. Density Reachable

Với hai giá trị $eps$ và $minSample$, một điểm dữ liệu $q$ được gọi là **density reachable** từ một điểm dữ liệu $p$ nếu tồn tại một chuỗi các điểm dữ liệu $p, p_1, p_2, \ldots, p_n, q$ liên tiếp từ $p$ đến $q$ sao cho mỗi điểm trong chuỗi đều là **directly density reachable** từ điểm trước đó.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/density_reachable_2.png" width="600"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/density_reachable_3.png" width="800"/>

### 2.5. Density Connected

Với hai giá trị $eps$ và $minSample$, hai điểm dữ liệu $p$ và $q$ được gọi là **density connected** nếu tồn tại một điểm dữ liệu $o$ sao cho cả $p$ và $q$ đều là **density reachable** từ $o$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/density_connected.png" width="800"/>

### 2.6. Cluster

Với $D$ là bộ dữ liệu cần phân cụm và hai giá trị $eps$ và $minSample$, một tập hợp không rỗng các điểm dữ liệu $C \subset D$ được gọi là **cluster** nếu thoả mãn các điều kiện sau:
- **Maximality**: $\forall p$ và $q$, nếu $p \in C$ và $q$ là **density reachable** từ $p$, thì $q \in C$.
- **Connectivity**: $\forall p, q \in C$, $p$ và $q$ là **density connected**.

### 2.7. Noise

Với hai giá trị $eps$ và $minSample$, các cụm $C_1, C_2, \ldots, C_n$ là các cụm được tìm thấy trong bộ dữ liệu $D$, các điểm dữ liệu không thuộc về bất kỳ cụm nào trong số đó được gọi là **noise**.

$$ noise = \{p \in D | \forall i: p \notin C_i\} $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/noise.png" width="800"/>

## 3. Các bước của thuật toán

Với các khái niệm đã được định nghĩa ở trên, chúng ta sẽ đi vào các bước hoạt động của thuật toán DBSCAN.

- **Bước 1**: Chọn giá trị cho $eps$ và $minSample$ (hoặc $minPts$).
- Thực hiện vòng lặp thuật toán với mỗi điểm dữ liệu trong bộ dữ liệu, giả sử với điểm dữ liệu đang xét là $p$.
    - **Bước 2**: Khởi tạo một cụm mới $C_i$.
    - **Bước 3**:
        - Nếu điểm dữ liệu $p$ chưa thuộc về bất kỳ cụm nào, tiếp tục lượt lặp với điểm dữ liệu này.
        - Nếu điểm dữ liệu $p$ đã thuộc về một cụm nào đó, đi đến lượt lặp với điểm dữ liệu tiếp theo.
    - **Bước 4**: Lấy hết các điểm dữ liệu trong vùng lận cận epsilon của điểm dữ liệu $p$, ta được tập hợp p_seed_list.
        - Nếu số lượng trong p_seed_list nhỏ hơn $minSample$, đánh dấu điểm dữ liệu này là **noise** và tiếp tục với điểm dữ liệu tiếp theo.
        - Nếu số lượng trong p_seed_list lớn hơn hoặc bằng $minSample$, ta kết luận **p là một core point** và gán tất cả các điểm dữ liệu trong p_seed_list vào cụm $C_i$.
    - Thực hiện vòng lặp với tất cả các điểm dữ liệu trong p_seed_list để mở rộng cụm $C_i$, giả sử với điểm dữ liệu đang xét là $q$.
        - **Bước 5**: Lấy hết các điểm dữ liệu trong vùng lận cận epsilon của điểm dữ liệu $q$, ta được tập hợp q_seed_list.
            - Nếu số lượng trong q_seed_list nhỏ hơn $minSample$, tiếp tục với điểm dữ liệu tiếp theo trong p_seed_list.
            - Nếu số lượng trong q_seed_list lớn hơn hoặc bằng $minSample$, ta kết luận **q là một core point** và gán tất cả các điểm dữ liệu trong q_seed_list vào cụm $C_i$.
        - **Bước 6**: Kết thúc vòng lặp với tất cả các điểm dữ liệu trong q_seed_list.
    - **Bước 7**: Kết thúc vòng lặp với tất cả các điểm dữ liệu trong p_seed_list.
- **Bước 8**: Kết thúc vòng lặp với tất cả các điểm dữ liệu trong bộ dữ liệu $D$.
- **Bước 9**: Trả về các cụm $C_1, C_2, \ldots, C_n$ và các điểm dữ liệu được đánh dấu là **noise**.

## 4. Ưu và nhược điểm của mô hình

- Ưu điểm:
    - Không cần xác định số lượng cụm trước.
    - Có thể phát hiện các cụm có hình dạng bất kỳ.
    - Có thể phát hiện các điểm dữ liệu nhiễu (noise).
- Nhược điểm:
    - Khó khăn trong việc lựa chọn các tham số đầu vào cho mô hình.
    - Không hoạt động tốt với các cụm có mật độ khác nhau.

## 5. Mẹo lựa chọn các tham số đầu vào cho mô hình

Đối với cả tham số $eps$ và $minSample$, việc tăng giảm giá trị của chúng sẽ ảnh hưởng đến việc một điểm dữ liệu nào đó có được coi là **core point** hay không.

||eps rất lớn|eps rất nhỏ|
|---|---|---|
|**minSample rất lớn**|Điều kiện **core point** rất lỏng hoặc rất chặt. Tất cả các điểm dữ liệu đều là **noise** hoặc thuộc chung một cụm|Điều kiện **core point** rất chặt. Tất cả các điểm dữ liệu đều là **noise**|
|**minSample rất nhỏ**|Điều kiện **core point** rất lỏng. Tất cả các điểm dữ liệu sẽ thuộc chung một cụm|Điều kiện **core point** rất lỏng. Mỗi điểm sẽ là một cụm|

Ta sẽ dựa vào những kết quả quan sát trên để tinh chỉnh các tham số đầu vào cho mô hình DBSCAN.

## 6. Các biến thể nâng cấp của mô hình

## 6.1. OPTICS
OPTICS (Ordering Points To Identify the Clustering Structure) là một biến thể, được phát triển để giải quyết một số nhược điểm của DBSCAN.
- Không yêu cầu người dùng phải xác định các tham số $eps$.
- Có thể phát hiện các cụm có mật độ khác nhau.
- Không tạo ra một phân cụm duy nhất, mà xây dựng một biểu diễn có thứ tự của cấu trúc mật độ trong dữ liệu, từ đó có thể trích xuất các cụm sau.

Ví dụ: Xét bộ dữ liệu gồm các điểm dữ liệu và giả sử, ta chọn $minSample = 2$, 

```python
A = (1, 1)
B = (2, 1)
C = (2, 2)
D = (8, 8)
E = (8, 9)
F = (25, 80)
```

- **Bước 1**: Tính khoảng cách giữa từng cặp điểm trong bộ dữ liệu

|   | A     | B     | C    | D     | E     | F     |
| - | ----- | ----- | ---- | ----- | ----- | ----- |
| A | 0     | 1     | 1.41 | 9.9   | 10.63 | 79.06 |
| B | 1     | 0     | 1    | 9.22  | 10    | 79.06 |
| C | 1.41  | 1     | 0    | 8.48  | 9.22  | 78.1  |
| D | 9.9   | 9.22  | 8.48 | 0     | 1     | 72.01 |
| E | 10.63 | 10    | 9.22 | 1     | 0     | 71.42 |
| F | 79.06 | 79.06 | 78.1 | 72.01 | 71.42 | 0     |

- **Bước 2**: Tính Core Distance của từng điểm với $minSample = 2$.
Với $minSample = 2$, Core Distance của một điểm dữ liệu $p$ được tính bằng khoảng cách giữa $p$ và điểm dữ liệu thứ $minSample$ trong vùng lận cận epsilon của nó.

```python
core_distance(A) = 1.41
core_distance(B) = 1
core_distance(C) = 1.41
core_distance(D) = 8.48
core_distance(E) = 9.22
core_distance(F) = 72.01
```

- **Bước 3**: Khởi tạo Reachability Distance cho từng điểm trong bộ dữ liệu bằng $\infty$ (vô cực).

```python
reachability_distance(A) = ∞
reachability_distance(B) = ∞
reachability_distance(C) = ∞
reachability_distance(D) = ∞
reachability_distance(E) = ∞
reachability_distance(F) = ∞
```

- **Bước 4**: Chạy vòng lặp từng điểm trong bộ dữ liệu.
Giả sử, ta bắt đầu với điểm dữ liệu $A$.
Viết các giá trị vào bảng:

| Điểm | Reachability Distance | Core Distance          |
| ---- | --------------------- | ---------------------- |
| A    | ∞                     | 1.41                   |

- **Bước 5**: Tính các Reachability Distance từ $A$ của các điểm hàng xóm của $A$ là $B$ và $C$.
Reachability Distance được tính bằng công thức sau:

$$ reachability_distance(p, q) = max(core_distance(p), dist(p, q)) $$

Thêm điểm $B$ và $C$ vào hàng ưu tiên xử lý.
Viết vào trong bảng theo thứ tự tăng dần của Reachability Distance.

| Điểm | Reachability Distance | Core Distance          |
| ---- | --------------------- | ---------------------- |
| A    | ∞                     | 1.41                   |
| B    | 1.41 (from A)         | 1                      |
| C    | 1.41 (from A)         | 1                      |

- **Bước 6**: Chọn điểm có Reachability Distance nhỏ nhất trong hàng ưu tiên là $B$.
Tính Reachability Distance từ $B$ đến các điểm hàng xóm của $B$ là $C$ (ta không tính với $A$ nữa vì $A$ đã được xử lý).

| Điểm | Reachability Distance | Core Distance          |
| ---- | --------------------- | ---------------------- |
| A    | ∞                     | 1.41                   |
| B    | 1.41 (from A)         | 1                      |
| C    | 1 (from B)            | 1                      |

- **Bước 7**: Chọn điểm có Reachability Distance nhỏ nhất trong hàng ưu tiên là $C$.
Cả hai hàng xóm của $C$ đều đã được xử lý, ta không cần tính Reachability Distance từ $C$ đến các điểm hàng xóm của nó nữa.

- **Bước 8**: Chọn điểm tiếp theo trong bộ dữ liệu chưa được xử lý là $D$.
Viết vào bảng:

| Điểm | Reachability Distance | Core Distance          |
| ---- | --------------------- | ---------------------- |
| A    | ∞ (from init)         | 1.41                   |
| B    | 1.41 (from A)         | 1                      |
| C    | 1 (from B)            | 1                      |
| D    | ∞ (from init)         | 8.48                   |

- **Bước 9**: Tính Reachability Distance từ $D$ đến các điểm hàng xóm của $D$ là $E$ (ta không tính với $C$ nữa vì $C$ đã được xử lý).

Thêm điểm $E$ vào hàng ưu tiên xử lý.
Viết vào trong bảng theo thứ tự tăng dần của Reachability Distance.

| Điểm | Reachability Distance | Core Distance          |
| ---- | --------------------- | ---------------------- |
| A    | ∞ (from init)         | 1.41                   |
| B    | 1.41 (from A)         | 1                      |
| C    | 1 (from B)            | 1                      |
| D    | ∞ (from init)         | 8.48                   |
| E    | 9.22 (from core distance E)| 9.22                   |

- **Bước 10**: Chọn điểm có Reachability Distance nhỏ nhất trong hàng ưu tiên là $E$.
Cả hai hàng xóm của $E$ (là $C$ và $D$) đều đã được xử lý, ta không cần tính Reachability Distance từ $E$ đến các điểm hàng xóm của nó nữa.

- **Bước 11**: Chọn điểm tiếp theo trong bộ dữ liệu chưa được xử lý là $F$.
Viết vào bảng:

| Điểm | Reachability Distance | Core Distance          |
| ---- | --------------------- | ---------------------- |
| A    | ∞ (from init)         | 1.41                   |
| B    | 1.41 (from A)         | 1                      |
| C    | 1 (from B)            | 1                      |
| D    | ∞ (from init)         | 8.48                   |
| E    | 9.22 (from core distance E)| 9.22              |
| F    | ∞ (from init)         | 72.01                  |

- **Bước 12**: Cả hai hàng xóm của $F$ (là $D$ và $E$) đều đã được xử lý, ta không cần tính Reachability Distance từ $F$ đến các điểm hàng xóm của nó nữa.

- **Bước 13**: Kết thúc vòng lặp với tất cả các điểm trong bộ dữ liệu.

- **Bước 14**: Từ cột Reachability Distance, ta có thể trả ra các cụm như sau:

| Điểm | Reachability Distance | Core Distance          | Cụm |
| ---- | --------------------- | ---------------------- | ---- |
| A    | ∞ (from init)         | 1.41                   | 0    |
| B    | 1.41 (from A)         | 1                      | 0    |
| C    | 1 (from B)            | 1                      | 0    |
| D    | ∞ (from init)         | 8.48                   | 1    |
| E    | 9.22 (from core distance E)| 9.22              | 1    |
| F    | ∞ (from init)         | 72.01                  | noise|
