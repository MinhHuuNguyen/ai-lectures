---
time: 12/16/2022
title: Đại số tuyến tính Linear Algebra
description: Đại số tuyến tính là nền tảng toán học cốt lõi trong machine learning, cung cấp ngôn ngữ và công cụ để biểu diễn và xử lý dữ liệu dưới dạng vector, ma trận và tensor. Các phép toán như nhân ma trận, chuẩn hóa vector hay phân rã ma trận được sử dụng trong hầu hết các thuật toán học máy, từ hồi quy tuyến tính đến mạng nơ-ron sâu. Nhờ đại số tuyến tính, việc tối ưu hóa mô hình, biểu diễn đặc trưng và xử lý dữ liệu quy mô lớn trở nên hiệu quả và chính xác hơn.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/banner.jpeg
tags: [math]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về Đại số tuyến tính

Đại số tuyến tính là nền tảng toán học cốt lõi trong machine learning, cung cấp ngôn ngữ và công cụ để biểu diễn và xử lý dữ liệu dưới dạng vector, ma trận và tensor.
Các phép toán như nhân ma trận, chuẩn hóa vector hay phân rã ma trận được sử dụng trong hầu hết các thuật toán học máy, từ hồi quy tuyến tính đến mạng nơ-ron sâu.

Nhờ đại số tuyến tính, việc tối ưu hóa mô hình, biểu diễn đặc trưng và xử lý dữ liệu quy mô lớn trở nên hiệu quả và chính xác hơn.

### 1.1. Vector

Theo cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/books/mathematics_for_machine_learning_deisenroth.pdf) của nhóm tác giả Marc Peter Deisenroth, A. Aldo Faisal và Cheng Soon Ong, một cách khái quát, **vector là một đối tượng đặc biệt, trong đó, chúng có thể cộng lại với nhau và nhân với một giá trị để tạo thành một đối tượng mới cùng loại**.

Do đó, bất kỳ đối tượng toán học nào thoả mãn hai tính chất trên sẽ được được xem xét là một vector.

Ví dụ: Trong hình học giải tích, một vector hình học được biểu diễn dưới dạng một mũi tên trong không gian tọa độ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/vector.jpeg" style="width: 400px;"/>

Một cách cụ thể hơn, trong đại số tuyến tính, **vector là một dãy các số thực được sắp xếp theo thứ tự, có thể biểu diễn dưới dạng cột hoặc dạng hàng**.

Cho một số nguyên dương $n \in N$, một vector cột $x$ gồm $n$ phần tử được biểu diễn như sau.

$$
x =
\begin{bmatrix}
x_{1} \\
x_{2} \\
... \\
x_{n}
\end{bmatrix}, x_{i} \in R
$$

Một vector được gọi là vector cột hay vector hàng không quá quan trọng trong đại số tuyến tính, vì ta có thể dễ dàng chuyển đổi giữa hai dạng này thông qua phép chuyển vị (transpose).
Việc biểu diễn vector dưới dạng cột hay hàng thường phụ thuộc vào ngữ cảnh sử dụng trong các phép toán.

### 1.2. Ma trận

Cho hai số nguyên dương $m, n \in N$, một ma trận các giá trị số thực $A$ là nhóm $m \times n$ giá trị thực $a_{ij}$ (với $i = 1, ..., m$ và $j = 1, ..., n$) được sắp xếp theo thứ tự thành hình chữ nhật gồm $m$ hàng và $n$ cột.

Các hàng hoặc cột trong ma trận được gọi là các vector hàng hoặc vector cột.

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\
a_{21} & a_{22} & ... & a_{2n} \\
... & ... & ... & ... \\
a_{m1} & a_{m2} & ... & a_{mn}
\end{bmatrix}, a_{ij} \in R
$$

Ma trận A ở trên gồm m hàng, n cột và $m * n$ giá trị thực, do đó, ma trận A được ký hiệu $A \in R^{m \times n}$.

Trong một số trường hợp, ta có thể ký hiệu ma trận $A \in R^{m \times n}$ dưới dạng vector $a \in R^{mn}$ bằng cách ghép các cột của ma trận lại tạo ra một vector dài.

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/books/mathematics_for_machine_learning_deisenroth.pdf) minh họa cách chuyển đổi từ ma trận sang vector bằng cách ghép các cột của ma trận.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/reshape.jpeg" style="width: 400px;"/>

### 1.3. Tensor

Tensor là một khái niệm tổng quát hoá của vector và ma trận.
Cụ thể, vector là tensor bậc nhất (1D), ma trận là tensor bậc hai (2D), và tensor có thể có bậc cao hơn (3D, 4D, ..., nD).

Hình ảnh dưới đây được lấy từ cuốn sách [Deep Learning with PyTorch](https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/books/deep_learning_pytorch_stevens.pdf) của nhóm tác giả Eli Stevens, Luca Antiga và Thomas Viehmann, minh họa các khái niệm về vector, ma trận và tensor.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/tensor.jpeg" style="width: 800px;"/>

Một tensor bậc $n$ có thể được biểu diễn như một mảng đa chiều với $n$ chỉ số.
Ví dụ, một tensor bậc ba có thể được biểu diễn như một khối lập phương gồm các giá trị thực.

Khi làm việc với những loại dữ liệu phức tạp như hình ảnh màu (có ba kênh RGB) hoặc chuỗi video (có thêm chiều thời gian), tensor trở thành công cụ hữu ích để biểu diễn và xử lý dữ liệu này một cách hiệu quả.
Ngoài ra, khi làm việc với những mô hình deep learning phức tạp, tensor giúp biểu diễn các tham số và trọng số của mô hình một cách trực quan và dễ dàng thao tác.

Tensor là khái niệm được sử dụng rất nhiều trong các thư viện deep learning như TensorFlow, PyTorch ...

### 1.4. Không gian vector - Không gian dữ liệu

Không gian vector (vector space) là một tập hợp các vector, trong đó các phép toán cộng vector và nhân vector với một số vô hướng được định nghĩa và thoả mãn các tính chất cơ bản.

Không gian vector thường được sử dụng để biểu diễn dữ liệu trong machine learning, nơi mỗi vector đại diện cho một điểm dữ liệu trong không gian nhiều chiều, khi đó, không gian vector được gọi là không gian dữ liệu (data space).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/vector_space.jpeg" style="width: 600px;"/>

Trong lĩnh vực xử lý ngôn ngữ tự nhiên, không gian vector được sử dụng để biểu diễn các từ dưới dạng các vector trong không gian nhiều chiều.
Trong đó, các từ có ý nghĩa tương tự sẽ được biểu diễn bởi các vector gần nhau trong không gian này.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/data_space.jpeg" style="width: 600px;"/>

Gần đây, có những mô hình machine learning có khả năng biểu diễn không chỉ từ mà còn cả hình ảnh dưới dạng vector trong cùng một không gian vector, điều này giúp việc tìm kiếm thông tin đa phương tiện trở nên hiệu quả hơn.

## 2. Các khái niệm ma trận đặc biệt

### 2.1. Ma trận bằng nhau (Equality of matrix)

Hai ma trận được gọi là bằng nhau nếu từng giá trị ở từng vị trí của ma trận này bằng giá trị ở từng vị trí tương ứng của ma trận kia.

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\
a_{21} & a_{22} & ... & a_{2n} \\
... & ... & ... & ... \\
a_{m1} & a_{m2} & ... & a_{mn}
\end{bmatrix}
\in R^{m \times n}
,

B =
\begin{bmatrix}
b_{11} & b_{12} & ... & b_{1n} \\
b_{21} & b_{22} & ... & b_{2n} \\
... & ... & ... & ... \\
b_{m1} & b_{m2} & ... & b_{mn}
\end{bmatrix}
\in R^{m \times n}
$$

$$
A = B \Leftrightarrow
\begin{cases}
a_{ij} = b_{ij} \\
\forall i = 1, 2, ..., m; j = 1, 2, ..., n
\end{cases}
$$

### 2.2. Ma trận không (Zero matrix)

Ma trận không là ma trận gồm tất cả các giá trị là số 0.

$$
0^{m \times n} =
\begin{bmatrix}
0 & 0 & ... & 0 \\
0 & 0 & ... & 0 \\
... & ... & ... & ... \\
0 & 0 & ... & 0
\end{bmatrix}
$$

### 2.3. Ma trận đối (Counter matrix)

Hai ma trận được gọi là đối nhau nếu từng giá trị ở từng vị trí của ma trận này là số đối của giá trị ở từng vị trí tương ứng của ma trận kia.

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\
a_{21} & a_{22} & ... & a_{2n} \\
... & ... & ... & ... \\
a_{m1} & a_{m2} & ... & a_{mn}
\end{bmatrix}
\in R^{m \times n}
,

- A =
\begin{bmatrix}
- a_{11} & - a_{12} & ... & - a_{1n} \\
- a_{21} & - a_{22} & ... & - a_{2n} \\
... & ... & ... & ... \\
- a_{m1} & - a_{m2} & ... & - a_{mn}
\end{bmatrix}
\in R^{m \times n}
$$

### 2.4. Ma trận vuông (Square matrix)

Ma trận vuông là ma trận có số hàng bằng số cột

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\
a_{21} & a_{22} & ... & a_{2n} \\
... & ... & ... & ... \\
a_{n1} & a_{n2} & ... & a_{nn}
\end{bmatrix}
\in R^{n \times n}
$$

Các phần từ có vị trí hàng bằng vị trí cột tạo nên **đường chéo chính** của ma trận, cụ thể, đường chéo chính gồm các phần tử $a_{11}, a_{22}, a_{33}, ..., a_{nn}$.

### 2.5. Ma trận tam giác (Triangular matrix)

Xuất phát từ ma trận vuông, ma trận tam giác có toàn bộ các phần tử ở một phía của đường chéo chính bằng 0.

$$
A_1 =
\begin{bmatrix}
a_{11} & 0 & ... & 0 \\
a_{21} & a_{22} & ... & 0 \\
... & ... & ... & ... \\
a_{n1} & a_{n2} & ... & a_{nn}
\end{bmatrix}
\in R^{n \times n}
$$

$$
A_2 =
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\
0 & a_{22} & ... & a_{2n} \\
... & ... & ... & ... \\
0 & 0 & ... & a_{nn}
\end{bmatrix}
\in R^{n \times n}
$$

trong đó, ma trận $A_1$ được gọi là ma trận tam giác dưới, ma trận $A_2$ được gọi là ma trận tam giác trên. 

### 2.6. Ma trận đơn vị (Indentity matrix)

Xuất phát từ ma trận vuông, ma trận đơn vị là ma trận có toàn bộ các phần từ trên đường chéo chính bằng 1, các phần tử khác bằng 0.

$$
A_1 =
\begin{bmatrix}
1 & 0 & ... & 0 \\
0 & 1 & ... & 0 \\
... & ... & ... & ... \\
0 & 0 & ... & 1
\end{bmatrix}
\in R^{n \times n}
$$

### 2.7. Ma trận chuyển vị (Transpose of a matrix)

Ma trận chuyển vị là ma trận được biến đổi từ ma trận gốc ban đầu, trong đó, các hàng của ma trận gốc được biến đổi thành các cột của ma trận chuyển vị và các cột của ma trận gốc được biến đổi thành các hàng của ma trận chuyển vị.
Ma trận chuyển vị của ma trận $A$ được ký hiệu là $A^T$

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\
a_{21} & a_{22} & ... & a_{2n} \\
... & ... & ... & ... \\
a_{m1} & a_{m2} & ... & a_{mn}
\end{bmatrix}
\in R^{m \times n}
$$

$$
A^T =
\begin{bmatrix}
a_{11} & a_{21} & ... & a_{m1} \\
a_{12} & a_{22} & ... & a_{m2} \\
... & ... & ... & ... \\
a_{1n} & a_{2n} & ... & a_{mn}
\end{bmatrix}
\in R^{n \times m}
$$

### 2.8. Ma trận đối xứng
Ma trận đối xứng là ma trận mà sau khi thực hiện phép chuyển vị, ta vẫn thu được ma trận ban đầu.

$A = A^T$ thì $A$ và $A^T$ là hai ma trận đối xứng.

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\
a_{12} & a_{22} & ... & a_{2n} \\
... & ... & ... & ... \\
a_{1n} & a_{2n} & ... & a_{nn}
\end{bmatrix}
\in R^{n \times n}
$$

$$
A^T =
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\
a_{12} & a_{22} & ... & a_{2n} \\
... & ... & ... & ... \\
a_{1n} & a_{2n} & ... & a_{nn}
\end{bmatrix}
\in R^{n \times n}
$$

### 2.9. Ma trận nghịch đảo

Ma trận nghịch đảo của một ma trận ban đầu là hai ma trận mà tích của chúng là ma trận đơn vị.
Ma trận nghịch đảo của ma trận $A$ được ký hiệu là $A^{-1}$

$AA^{-1} = I = A^{-1}A$ thì $A$ và $A^{-1}$ là hai ma trận nghịch đảo.

$$
A =
\begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix}
\in R^{2 \times 2}
$$

$$
A^{-1} = \frac{1}{a_{11}a_{22} - a_{12}a_{21}}
\begin{bmatrix}
a_{22} & - a_{12} \\
- a_{21} & a_{11}
\end{bmatrix}
\in R^{2 \times 2}
$$

## 3. Các phép toán cơ bản trên vector và ma trận

### 3.1. Phép toán trên vector

#### Phép cộng vector

$$
a =
\begin{bmatrix}
a_{1} & a_{2} & ... & a_{n}
\end{bmatrix}
\in R^{n}
,

b =
\begin{bmatrix}
b_{1} & b_{2} & ... & b_{n}
\end{bmatrix}
\in R^{n}
$$

$$
a + b =
\begin{bmatrix}
a_{1} + b_{1} & a_{2} + b_{2} & ... & a_{n} + b_{n}
\end{bmatrix}
\in R^{n}
$$

Phép cộng vector có các tính chất tương tự với phép cộng số học:
- Tính chất giao hoán: $a + b = b + a$
- Tính chất kết hợp: $(a + b) + c = a + (b + c)$
- Tính chất cộng với vector 0: $a + 0 = a$
- Tính chất cộng với vector đối: $a + (-a) = 0$

#### Phép nhân vector với một số vô hướng

$$
a =
\begin{bmatrix}
a_{1} & a_{2} & ... & a_{n}
\end{bmatrix}
\in R^{n}
,

x \in R
$$

$$
x \cdot a =
\begin{bmatrix}
x \cdot a_{1} & x \cdot a_{2} & ... & x \cdot a_{n}
\end{bmatrix}
\in R^{n}
$$

Phép nhân vector với một số vô hướng có tính chất giao hoán: $a * x = x * a$

#### Tích vô hướng của hai vector (Dot product)

Tích vô hướng (dot product) của hai vector là phép nhân giữa hai vector và trả đầu ra kết quả là một giá trị vô hướng.
Tích vô hướng còn có tên gọi khác là scalar product.

$$
a =
\begin{bmatrix}
a_{1} & a_{2} & ... & a_{n}
\end{bmatrix}
\in R^{n}
,

b =
\begin{bmatrix}
b_{1} & b_{2} & ... & b_{n}
\end{bmatrix}
\in R^{n}
$$

$$
a \cdot b = a_{1} \cdot b_{1} + a_{2} \cdot b_{2} + ... + a_{n} \cdot b_{n} \in R
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
- Đối với phim 1: $a \cdot b_1 = 5 * 4 + 1 * 5 + 3 * 1 = 28$.
- Đối với phim 2: $a \cdot b_2 = 5 * 5 + 1 * 1 + 3 * 5 = 41$.

Do đó, ta có thể kết luận rằng người khán giả này có khả năng cao hơn sẽ thích phim 2 hơn so với phim 1 vì điểm số tích vô hướng của phim 2 (41) cao hơn phim 1 (28).

### 3.2. Các phép toán trên ma trận

#### Phép cộng ma trận

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\
a_{21} & a_{22} & ... & a_{2n} \\
... & ... & ... & ... \\
a_{m1} & a_{m2} & ... & a_{mn}
\end{bmatrix}
\in R^{m \times n}
,

B =
\begin{bmatrix}
b_{11} & b_{12} & ... & b_{1n} \\
b_{21} & b_{22} & ... & b_{2n} \\
... & ... & ... & ... \\
b_{m1} & b_{m2} & ... & b_{mn}
\end{bmatrix}
\in R^{m \times n}
$$

$$
A + B =
\begin{bmatrix}
a_{11} + b_{11} & a_{12} + b_{12} & ... & a_{1n} + b_{1n} \\
a_{21} + b_{21} & a_{22} + b_{22} & ... & a_{2n} + b_{2n} \\
... & ... & ... & ... \\
a_{m1} + b_{m1} & a_{m2} + b_{m2} & ... & a_{mn} + b_{mn}
\end{bmatrix}
\in R^{m \times n}
$$

Phép cộng ma trận có các tính chất tương tự với phép cộng số học:
- Tính chất giao hoán: $A + B = B + A$
- Tính chất kết hợp: $(A + B) + C = A + (B + C)$
- Tính chất cộng với ma trận 0: $A + 0 = A$
- Tính chất cộng với ma trận đối: $A + (-A) = 0$

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/books/mathematics_for_machine_learning_deisenroth.pdf) giải thích vì sao phép nhân ma trận không có tính chất giao hoán.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/multiplication.jpeg" style="width: 400px;"/>

#### Phép nhân ma trận với một số vô hướng

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\
a_{21} & a_{22} & ... & a_{2n} \\
... & ... & ... & ... \\
a_{m1} & a_{m2} & ... & a_{mn}
\end{bmatrix}
\in R^{m \times n}
,
x \in R
$$

$$
x \cdot A =
\begin{bmatrix}
x \cdot a_{11} & x \cdot a_{12} & ... & x \cdot a_{1n} \\
x \cdot a_{21} & x \cdot a_{22} & ... & x \cdot a_{2n} \\
... & ... & ... & ... \\
x \cdot a_{m1} & x \cdot a_{m2} & ... & x \cdot a_{mn}
\end{bmatrix}
\in R^{m \times n}
$$

Phép nhân ma trận với một số vô hướng có tính chất giao hoán: $A * x = x * A$

#### Phép nhân ma trận với ma trận

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n} \\
a_{21} & a_{22} & ... & a_{2n} \\
... & ... & ... & ... \\
a_{m1} & a_{m2} & ... & a_{mn}
\end{bmatrix}
\in R^{m \times n}
,

B =
\begin{bmatrix}
b_{11} & b_{12} & ... & b_{1k} \\
b_{21} & b_{22} & ... & b_{2k} \\
... & ... & ... & ... \\
b_{n1} & b_{n2} & ... & b_{nk}
\end{bmatrix}
\in R^{n \times k}
$$

$$
AB =
\begin{bmatrix}
c_{11} & c_{12} & ... & c_{1k} \\
c_{21} & c_{22} & ... & c_{2k} \\
... & ... & ... & ... \\
c_{m1} & c_{m2} & ... & c_{mk}
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

## 4. Một số ứng dụng trong Machine Learning

### 4.1. Bài toán Giảm chiều Dữ liệu (Dimensionality Reduction)

Giảm chiều dữ liệu (Dimensionality reduction) là quá trình giảm số chiều của dữ liệu trong không gian đa chiều.
Mục tiêu của quá trình này là làm cho việc xử lý và phân tích dữ liệu trở nên hiệu quả hơn mà không làm mất đi quá nhiều thông tin quan trọng.

Có hai loại chính của mô hình giảm chiều dữ liệu: **Giảm chiều dữ liệu tuyến tính** và **Giảm chiều dữ liệu phi tuyến tính**.
Ở đây, ta sẽ nói về việc ứng dụng đại số tuyến tính trong giảm chiều dữ liệu tuyến tính, cụ thể là phương pháp **Phân tích thành phần chính (Principal Component Analysis - PCA)**.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/13-pca/idea.jpeg" style="width: 500px;"/>

PCA giúp giảm số chiều của dữ liệu bằng cách tìm kiếm các thành phần chính (principal components) mà giữ lại nhiều phương sai nhất trong dữ liệu.
Điều này giúp bảo toàn sự biến động lớn nhất trong dữ liệu.

Nói cách khác, PCA sẽ đánh giá mức độ quan trọng của từng chiều dữ liệu và loại bỏ những chiều không quan trọng, từ đó giảm số chiều của dữ liệu.

Để làm được điều này PCA sử dụng hai khái niệm quan trọng trong đại số tuyến tính là **Eigenvector** là vector riêng và **Eigenvalue** là giá trị riêng.

Để hiểu rõ hơn về hai khái niệm **Eigenvector** và **Eigenvalue** cũng như là cách áp dụng hai khái niệm này trong mô hình PCA, bạn có thể tham khảo bài viết [Giá trị riêng eigenvalues và vector riêng eigenvectors](/blog/gia-tri-rieng-eigenvalues-va-vector-rieng-eigenvectors) và bài viết [Mô hình PCA](/blog/mo-hinh-pca).

### 4.2. Phép biến đổi tuyến tính và mạng Nơ-ron (Neural Networks)

Phép biến đổi tuyến tính (Linear transformation) là một phép biến đổi giữa hai không gian vector mà giữ nguyên các phép toán cộng vector và nhân vector với một số vô hướng.
Phép biến đổi tuyến tính có thể được biểu diễn dưới dạng **phép nhân ma trận**.

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/books/mathematics_for_machine_learning_deisenroth.pdf) mô tả phép biến đổi tuyến tính từ không gian vector $R^2$ sang không gian vector $R^2$ thông qua phép nhân ma trận lần lượt với các ma trận khác nhau.

$$
A_1 =
\begin{bmatrix}
cos(\pi/4) & -sin(\pi/4) \\
sin(\pi/4) & cos(\pi/4)
\end{bmatrix}
,
A_2 =
\begin{bmatrix}
2 & 0 \\
0 & 1
\end{bmatrix}
,
A_3 =
\frac{1}{2}
\begin{bmatrix}
3 & -1 \\
1 & -1
\end{bmatrix}
$$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/1-neural-network/linear_transformation.jpeg" style="width: 500px;"/>

Trong mạng nơ-ron, mỗi lớp của mạng có thể được xem như một phép biến đổi tuyến tính áp dụng lên đầu vào của lớp đó.
Ý nghĩa của việc sử dụng phép biến đổi tuyến tính trong mạng nơ-ron là giúp mô hình học được các đặc trưng phức tạp từ dữ liệu đầu vào thông qua việc kết hợp các trọng số và bias.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/1-neural-network/dnn.jpeg" style="width: 500px;"/>

Nói cách khác, việc biến đổi tuyến tính từ dữ liệu ban đầu sang không gian đặc trưng mới giúp mạng nơ-ron có khả năng biểu diễn và trích xuất được ra những đặc trưng quan trọng của dữ liệu, từ đó cải thiện hiệu suất của mô hình trong các nhiệm vụ như phân loại, hồi quy hay nhận dạng mẫu.

Để hiểu rõ hơn về sử dụng phép biến đổi tuyến tính (hay phép nhân ma trận) trong mô hình mạng nơ ron, bạn có thể tham khảo bài viết [Mô hình hồi quy tuyến tính Linear Regression](/blog/mo-hinh-hoi-quy-tuyen-tinh-linear-regression) và bài viết [Mô hình mạng nơ ron đơn giản Neural network](/blog/mo-hinh-mang-no-ron-don-gian-neural-network).

<!-- ### 4.3. Chuẩn hoá và khoảng cách -->
