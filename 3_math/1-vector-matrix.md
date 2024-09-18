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

# Kiến thức cơ bản về vector và ma trận

## 1. Giới thiệu chung về vector

### 1.1. Khái niệm vector

Theo cuốn sách MATHEMATICS FOR MACHINE LEARNING của nhóm tác giả Marc Peter Deisenroth, A. Aldo Faisal và Cheng Soon Ong, một cách khái quát, **vector là một đối tượng đặc biệt, trong đó, chúng có thể cộng lại với nhau và nhân với một giá trị để tạo thành một đối tượng mới cùng loại**.

Do đó, bất kỳ đối tượng toán học nào thoả mãn hai tính chất trên sẽ được được xem xét là một vector.

<!-- Ví dụ đầu tiên về vector là vector hình học -->

### 1.2. Phép toán trên vector

#### 1.2.1. Phép cộng vector

$$
a =
\begin{bmatrix}
a_{1} & a_{2} & \dots & a_{n}
\end{bmatrix}
\in R^{n}
,

b =
\begin{bmatrix}
b_{1} & b_{2} & \dots & b_{n}
\end{bmatrix}
\in R^{n}
$$

$$
a + b =
\begin{bmatrix}
a_{1} + b_{1} & a_{2} + b_{2} & \dots & a_{n} + b_{n}
\end{bmatrix}
\in R^{n}
$$

Phép cộng vector có các tính chất tương tự với phép cộng số học:
- Tính chất giao hoán: $a + b = b + a$
- Tính chất kết hợp: $(a + b) + c = a + (b + c)$
- Tính chất cộng với vector 0: $a + 0 = a$
- Tính chất cộng với vector đối: $a + (-a) = 0$


#### 1.2.2. Phép nhân vector với một số vô hướng

$$
a =
\begin{bmatrix}
a_{1} & a_{2} & \dots & a_{n}
\end{bmatrix}
\in R^{n}
,

x \in R
$$

$$
x \cdot a =
\begin{bmatrix}
x \cdot a_{1} & x \cdot a_{2} & \dots & x \cdot a_{n}
\end{bmatrix}
\in R^{n}
$$

Phép nhân vector với một số vô hướng có tính chất giao hoán: $a * x = x * a$

#### 1.2.3. Tích vô hướng của hai vector (Dot product)

Tích vô hướng (dot product) của hai vector là phép nhân giữa hai vector và trả đầu ra kết quả là một giá trị vô hướng.
Tích vô hướng còn có tên gọi khác là scalar product.

$$
a =
\begin{bmatrix}
a_{1} & a_{2} & \dots & a_{n}
\end{bmatrix}
\in R^{n}
,

b =
\begin{bmatrix}
b_{1} & b_{2} & \dots & b_{n}
\end{bmatrix}
\in R^{n}
$$

$$
a \cdot b = a_{1} \cdot b_{1} + a_{2} \cdot b_{2} + \dots + a_{n} \cdot b_{n} \in R
$$

Tích vô hướng là một phép toán rất quan trọng trong đại số tuyến tính vì nó có thể được sử dụng để đánh giá độ tương đồng của hai vector, phép đánh giá này được gọi là **cosine similarity**.

$$sim(a, b) = cos(\theta) = \frac{a \cdot b}{||a|| ||b||}$$

trong đó, $||a||$ là độ dài của vector a, được tính bằng công thức

$$||a|| = \sqrt{\sum_{i=1}^{n} a_{i}^2}$$

Ví dụ về sử dụng tích vô hướng trong hệ thống recommendation:

Ta có thang điểm đánh giá sở thích của một khán giả trên từng thể loại phim khác nhau như sau:
- Khán giả đó thích phim hài: 5 điểm
- Khán giả đó không thích thích phim lãng mạn: 1 điểm
- Khán giả đó trung lập với phim hành động: 3 điểm

Từ đó, tạo thành một vector đại diện cho sở thích của khán giả $a = \begin{bmatrix} 5 & 1 & 3 \end{bmatrix}$.

Ta có hai bộ phim với thang điểm trên từng thể loại như sau:
- Phim 1: nhiều yếu tố hài (4 điểm) và lãng mạn (5 điểm), ít yếu tố hành động (1 điểm): $b_1 = \begin{bmatrix} 4 & 5 & 1 \end{bmatrix}$
- Phim 2: nhiều yếu tố hài (5 điểm) và hành động (5 điểm), ít yếu tố lãng mạn (1 điểm): $b_2 = \begin{bmatrix} 5 & 1 & 5 \end{bmatrix}$

Áp dụng tích vô hướng, ta có thể tính toán được mức độ tương đồng giữa sở thích của khán giả và các tính chất của phim:
- Đối với phim 1: $a \cdot b_1 = 5 * 4 + 1 * 5 + 3 * 1 = 28 $.
- Đối với phim 2: $a \cdot b_2 = 5 * 5 + 1 * 1 + 3 * 5 = 41 $.

Do đó, ta có thể kết luận rằng người khán giả này có khả năng cao hơn sẽ thích phim 2.

## 2. Giới thiệu chung về ma trận

### 2.1. Khái niệm ma trận
Cho hai số $m, n \in N$, ma trận các giá trị thực $A$ là nhóm $m \times n$ giá trị thực $a_{ij}$ (với $i = 1, \dots, m$ và $j = 1, \dots, n$) được sắp xếp theo thứ tự thành hình chữ nhật gồm $m$ hàng và $n$ cột.
Các hàng hoặc cột trong ma trận được gọi là các vector hàng hoặc vector cột.

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\dots & \dots & & \dots \\
a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}
, a_{ij} \in R
$$

Ma trận A ở trên gồm m hàng, n cột và $m*n$ giá trị thực, do đó, ma trận A được ký hiệu $A \in R^{m \times n}$.
Trong một số trường hợp, ta có thể ký hiệu ma trận $A \in R^{m \times n}$ dưới dạng vector $a \in R^{mn}$ bằng cách ghép các cột của ma trận lại tạo ra một vector dài.

$$
a =
\begin{bmatrix}
a_{11} \\
a_{21} \\
\dots \\
a_{m1} \\
a_{12} \\
a_{22} \\
\dots \\
a_{m2} \\
\dots \\
a_{1n} \\
a_{2n} \\
\dots \\
a_{mn} \\
\end{bmatrix}
, a_{ij} \in R
$$

### 2.2. Các dạng ma trận và biến đổi cơ bản

#### 2.2.1. Ma trận bằng nhau (Equality of matrix)

Hai ma trận được gọi là bằng nhau nếu từng giá trị ở từng vị trí của ma trận này bằng giá trị ở từng vị trí tương ứng của ma trận kia.

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\dots & \dots & & \dots \\
a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}
\in R^{m \times n}
,

B =
\begin{bmatrix}
b_{11} & b_{12} & \dots & b_{1n} \\
b_{21} & b_{22} & \dots & b_{2n} \\
\dots & \dots & & \dots \\
b_{m1} & b_{m2} & \dots & b_{mn} \\
\end{bmatrix}
\in R^{m \times n}
$$

$$
A = B \Leftrightarrow
\begin{cases}
  a_{ij} = b_{ij} \\
  \forall i = 1, 2, \dots, m; j = 1, 2, \dots, n
\end{cases}
$$

#### 2.2.2. Ma trận không (Zero matrix)

Ma trận không là ma trận gồm tất cả các giá trị là số 0.

$$
0^{m \times n} =
\begin{bmatrix}
0 & 0 & \dots & 0 \\
0 & 0 & \dots & 0 \\
\dots & \dots & & \dots \\
0 & 0 & \dots & 0 \\
\end{bmatrix}
$$

#### 2.2.3. Ma trận đối (Counter matrix)

Hai ma trận được gọi là đối nhau nếu từng giá trị ở từng vị trí của ma trận này là số đối của giá trị ở từng vị trí tương ứng của ma trận kia.

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\dots & \dots & & \dots \\
a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}
\in R^{m \times n}
,

- A =
\begin{bmatrix}
- a_{11} & - a_{12} & \dots & - a_{1n} \\
- a_{21} & - a_{22} & \dots & - a_{2n} \\
\dots & \dots & & \dots \\
- a_{m1} & - a_{m2} & \dots & - a_{mn} \\
\end{bmatrix}
\in R^{m \times n}
$$

#### 2.2.4. Ma trận vuông (Square matrix)

Ma trận vuông là ma trận có số hàng bằng số cột

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\dots & \dots & & \dots \\
a_{n1} & a_{n2} & \dots & a_{nn} \\
\end{bmatrix}
\in R^{n \times n}
$$

Các phần từ có vị trí hàng bằng vị trí cột tạo nên **đường chéo chính** của ma trận, cụ thể, đường chéo chính gồm các phần tử $a_{11}, a_{22}, a_{33}, \dots, a_{nn}$.

#### 2.2.5. Ma trận tam giác (Triangular matrix)

Xuất phát từ ma trận vuông, ma trận tam giác có toàn bộ các phần tử ở một phía của đường chéo chính bằng 0.

$$
A_1 =
\begin{bmatrix}
a_{11} & 0 & \dots & 0 \\
a_{21} & a_{22} & \dots & 0 \\
\dots & \dots & & \dots \\
a_{n1} & a_{n2} & \dots & a_{nn} \\
\end{bmatrix}
\in R^{n \times n}
$$

$$
A_2 =
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
0 & a_{22} & \dots & a_{2n} \\
\dots & \dots & & \dots \\
0 & 0 & \dots & a_{nn} \\
\end{bmatrix}
\in R^{n \times n}
$$

trong đó, ma trận $A_1$ được gọi là ma trận tam giác dưới, ma trận $A_2$ được gọi là ma trận tam giác trên. 

#### 2.2.6. Ma trận đơn vị (Indentity matrix)

Xuất phát từ ma trận vuông, ma trận đơn vị là ma trận có toàn bộ các phần từ trên đường chéo chính bằng 1, các phần tử khác bằng 0.

$$
A_1 =
\begin{bmatrix}
1 & 0 & \dots & 0 \\
0 & 1 & \dots & 0 \\
\dots & \dots & & \dots \\
0 & 0 & \dots & 1 \\
\end{bmatrix}
\in R^{n \times n}
$$

#### 2.2.7. Ma trận chuyển vị (Transpose of a matrix)

Ma trận chuyển vị là ma trận được biến đổi từ ma trận gốc ban đầu, trong đó, các hàng của ma trận gốc được biến đổi thành các cột của ma trận chuyển vị và các cột của ma trận gốc được biến đổi thành các hàng của ma trận chuyển vị.
Ma trận chuyển vị của ma trận $A$ được ký hiệu là $A^T$

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\dots & \dots & & \dots \\
a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}
\in R^{m \times n}
$$

$$
A^T =
\begin{bmatrix}
a_{11} & a_{21} & \dots & a_{m1} \\
a_{12} & a_{22} & \dots & a_{m2} \\
\dots & \dots & & \dots \\
a_{1n} & a_{2n} & \dots & a_{mn} \\
\end{bmatrix}
\in R^{n \times m}
$$

#### 2.2.8. Ma trận đối xứng
Ma trận đối xứng là ma trận mà sau khi thực hiện phép chuyển vị, ta vẫn thu được ma trận ban đầu.

$A = A^T$ thì $A$ và $A^T$ là hai ma trận đối xứng.

#### 2.2.9. Ma trận nghịch đảo

Ma trận nghịch đảo của một ma trận ban đầu là hai ma trận mà tích của chúng là ma trận đơn vị.
Ma trận nghịch đảo của ma trận $A$ được ký hiệu là $A^{-1}$

$AA^{-1} = I = A^{-1}A$ thì $A$ và $A^{-1}$ là hai ma trận nghịch đảo.

### 2.3. Các phép toán trên ma trận

#### 2.3.1. Phép cộng ma trận

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\dots & \dots & & \dots \\
a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}
\in R^{m \times n}
,

B =
\begin{bmatrix}
b_{11} & b_{12} & \dots & b_{1n} \\
b_{21} & b_{22} & \dots & b_{2n} \\
\dots & \dots & & \dots \\
b_{m1} & b_{m2} & \dots & b_{mn} \\
\end{bmatrix}
\in R^{m \times n}
$$

$$
A + B =
\begin{bmatrix}
a_{11} + b_{11} & a_{12} + b_{12} & \dots & a_{1n} + b_{1n} \\
a_{21} + b_{21} & a_{22} + b_{22} & \dots & a_{2n} + b_{2n} \\
\dots & \dots & & \dots \\
a_{m1} + b_{m1} & a_{m2} + b_{m2} & \dots & a_{mn} + b_{mn} \\
\end{bmatrix}
\in R^{m \times n}
$$

Phép cộng ma trận có các tính chất tương tự với phép cộng số học:
- Tính chất giao hoán: $A + B = B + A$
- Tính chất kết hợp: $(A + B) + C = A + (B + C)$
- Tính chất cộng với ma trận 0: $A + 0 = A$
- Tính chất cộng với ma trận đối: $A + (-A) = 0$

#### 2.3.2. Phép nhân ma trận với một số vô hướng

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\dots & \dots & & \dots \\
a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}
\in R^{m \times n}
,
x \in R
$$

$$
x \cdot A =
\begin{bmatrix}
x \cdot a_{11} & x \cdot a_{12} & \dots & x \cdot a_{1n} \\
x \cdot a_{21} & x \cdot a_{22} & \dots & x \cdot a_{2n} \\
\dots & \dots & & \dots \\
x \cdot a_{m1} & x \cdot a_{m2} & \dots & x \cdot a_{mn} \\
\end{bmatrix}
\in R^{m \times n}
$$

Phép nhân ma trận với một số vô hướng có tính chất giao hoán: $A * x = x * A$

#### 2.3.3. Phép nhân ma trận với ma trận

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\dots & \dots & & \dots \\
a_{m1} & a_{m2} & \dots & a_{mn} \\
\end{bmatrix}
\in R^{m \times n}
,

B =
\begin{bmatrix}
b_{11} & b_{12} & \dots & b_{1k} \\
b_{21} & b_{22} & \dots & b_{2k} \\
\dots & \dots & & \dots \\
b_{n1} & b_{n2} & \dots & b_{nk} \\
\end{bmatrix}
\in R^{n \times k}
$$

$$
AB =
\begin{bmatrix}
c_{11} & c_{12} & \dots & c_{1k} \\
c_{21} & c_{22} & \dots & c_{2k} \\
\dots & \dots & & \dots \\
c_{m1} & c_{m2} & \dots & c_{mk} \\
\end{bmatrix}
\in R^{m \times k}
$$

trong đó,
$c_{ij}$ là kết quả của phép nhân tích vô hướng giữa hàng $i$ của ma trận A và cột $j$ của ma trận B.

Ví dụ: $c_{34} = a_{31} * b_{14} + a_{32} * b_{24} + a_{33} * b_{34}$

Phép nhân tích vô hướng trong vector hay phép nhân ma trận với ma trận là phép toán quan trọng trong linear transformation (phép biến đổi tuyến tính, được sử dụng rất nhiều trong machine learning và cụ thể là mạng nơ ron).


Phép nhân ma trận với ma trận có các tính chất:
- Không có tính chất giao hoán
- Tính chất kết hợp: $(AB)C = A(BC)$
- Tính chất phân phối giữa phép nhân và phép cộng ma trận:
    - A(B + C) = AB + AC
    - (B + C)D = BD + CD
- Nhân với ma trận đơn vị:
    - AI = A
    - BI = B
    - Nếu A là ma trận vuông: AI = IA = A
