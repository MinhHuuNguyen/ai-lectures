---
time: 01/14/2023
title: Giá trị riêng eigenvalues và vector riêng eigenvectors
description: Trong đại số tuyến tính, eigenvector và eigenvalue là các công cụ cơ bản để phân tích đặc tính của các phép biến đổi tuyến tính. Qua bài viết này, ta sẽ tìm hiểu về bản chất, tính chất, cách tính và ứng dụng phong phú của eigenvector–eigenvalue trong các bài toán ứng dụng.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-vector-matrix/banner.png
tags: [math]
is_highlight: false
is_published: true
---

## 1. Tổng quan

Eigenvalues và eigenvectors là các khái niệm quan trọng trong đại số tuyến tính, thường được sử dụng trong nhiều lĩnh vực như học máy, xử lý tín hiệu, đồ họa máy tính và nhiều lĩnh vực khác.
Chúng giúp phân tích và hiểu rõ hơn về cấu trúc của ma trận và các phép biến đổi tuyến tính.

Một eigenvector của phép biến đổi tuyến tính $T$ (hay ma trận $A$) là một vector không đổi phương (có thể đổi chiều) khi phép biến đổi đó tác động lên nó, và eigenvalue tương ứng là hệ số nhân mà vector bị co giãn hoặc phản chiều.

## 2. Định nghĩa và ý nghĩa

Với phép biến đổi tuyến tính $T$ trên không gian vector $V$, một vector $v \neq 0$ được gọi là eigenvector nếu tồn tại số vô hướng $\lambda$ sao cho
$$ T(v) = \lambda v $$

Trong trường hợp ma trận $A \in \mathbb{R}^{n \times n}$, điều này tương đương với
$$ A v = \lambda v $$

Giá trị $\lambda$ gọi là eigenvalue hay giá trị riêng, biểu diễn mức độ co giãn (nếu $|\lambda| \neq 1$) hoặc lật chiều (nếu $\lambda < 0$) của vector riêng.

Các eigenvalue là các nghiệm mà ta thu được khi giải phương trình đặc trưng.
Để tìm eigenvalue, ta cần giải phương trình đặc trưng:
$$ det(A - \lambda I) = 0 $$
trong đó:
- $A$ là ma trận vuông $n \times n$.
- $I$ là ma trận đơn vị $n \times n$.
- $det$ là định thức của ma trận.
- $det(A - \lambda I)$ là đa thức đặc trưng của ma trận $A$.

Ý nghĩa của **Giá trị riêng (Eigenvalues)** là dùng để đo lường độ lớn của phương sai của ma trận $A$ theo các hướng khác nhau.
Giá trị riêng càng lớn, phương sai theo hướng đó của ma trận $A$ càng lớn, hướng đó của ma trận $A$ càng quan trọng trong việc mô tả ma trận $A$.

Ý nghĩa của **Vector riêng (Eigenvectors)** là dùng để xác định hướng của các thành phần của ma trận $A$.
Kết hợp với giá trị riêng, vector riêng cho phép ta mô tả ma trận $A$.

Ví dụ: Xét ma trận $A$ là ma trận hiệp phương sai của các đặc trưng của bộ dữ liệu $X$.
Cụ thể, ma trận $A$ có kích thước $n \times n$ với $n$ là số đặc trưng của bộ dữ liệu $X$.

Ta có thể sử dụng giá trị riêng của ma trận $A$ để xác định độ quan trọng của các đặc trưng trong bộ dữ liệu $X$.
Đặc trưng nào có giá trị riêng lớn hơn sẽ là đặc trưng quan trọng hơn trong bộ dữ liệu $X$ và ngược lại.

Từ độ lớn của các giá trị riêng, ta có thể lựa chọn các giá trị riêng lớn nhất và các vector riêng tương ứng, tạo ra các thành phần chính (principal components) của ma trận $A$.
Từ thành phần chính này, ta có thể tạo ra ma trận mới $A'$ với kích thước nhỏ hơn ma trận $A$ nhưng vẫn giữ được thông tin quan trọng nhất của ma trận $A$.
