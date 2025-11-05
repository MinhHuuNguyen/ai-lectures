---
time: 01/07/2023
title: Giải tích toán học Calculus
description: Giải tích toán học (Calculus) đóng vai trò trọng yếu trong machine learning, đặc biệt trong việc tối ưu hóa mô hình. Các khái niệm như đạo hàm, vi phân và gradient được sử dụng để cập nhật tham số mô hình nhằm giảm thiểu hàm mất mát. Nhờ giải tích, thuật toán gradient descent và các biến thể của nó có thể tìm ra điểm cực tiểu của hàm mục tiêu, giúp mô hình học được từ dữ liệu. Ngoài ra, giải tích còn hỗ trợ hiểu rõ sự biến thiên của đầu ra theo đầu vào, góp phần vào việc phân tích độ nhạy và ổn định của hệ thống học máy.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/banner.jpeg
tags: [math]
is_highlight: false
is_published: true
---

## 1. Các khái niệm cơ bản

Giải tích toán học (Calculus) đóng vai trò trọng yếu trong machine learning, đặc biệt trong việc tối ưu hóa mô hình.
Các khái niệm như đạo hàm, vi phân và gradient được sử dụng để cập nhật tham số mô hình nhằm giảm thiểu hàm mất mát.

Nhờ giải tích, thuật toán gradient descent và các biến thể của nó có thể tìm ra điểm cực tiểu của hàm mục tiêu, giúp mô hình học được từ dữ liệu.
Ngoài ra, giải tích còn hỗ trợ hiểu rõ sự biến thiên của đầu ra theo đầu vào, góp phần vào việc phân tích độ nhạy và ổn định của hệ thống học máy.

### 1.1. Giới hạn của hàm số (Limits)

Giới hạn mô tả hành vi của hàm số $f(x)$ khi đối số $x$ tiến gần đến một giá trị cụ thể $x_0$.

$$
\lim_{x \to x_0} f(x) = L
$$

Ta có thể diễn giải giới hạn như sau: Khi $x$ tiến gần đến $x_0$ thì giá trị của hàm số $f(x)$ sẽ tiến gần đến giá trị $L$.

Giới hạn là cơ sở để định nghĩa đạo hàm.

Ví dụ: Xét hàm số $f(x) = \frac{x^2 - 1}{x - 1}$.
Khi $x$ tiến gần đến 1, ta có:

$$
\lim_{x \to 1} f(x) = \lim_{x \to 1} \frac{x^2 - 1}{x - 1} = \lim_{x \to 1} \frac{(x-1)(x+1)}{(x-1)} = \lim_{x \to 1} (x+1) = 2
$$

Vậy ta nói rằng giới hạn của hàm số $f(x)$ khi $x$ tiến gần đến 1 là 2, dù $f(1)$ không được xác định.

### 1.2. Tính liên tục của hàm số (Continuity)

Hàm số $f(x)$ được gọi là liên tục tại điểm $x_0$ nếu thỏa mãn ba điều kiện sau:
- Hàm số $f(x)$ được xác định tại điểm $x_0$, tức là $f(x_0)$ tồn tại.
- Giới hạn của hàm số khi $x$ tiến gần đến $x_0$ tồn tại, tức là $\lim_{x \to x_0} f(x)$ tồn tại.
- Giới hạn của hàm số khi $x$ tiến gần đến $x_0$ bằng giá trị của hàm số tại điểm đó, tức là $\lim_{x \to x_0} f(x) = f(x_0)$.

Nếu một trong ba điều kiện trên không thỏa mãn, hàm số $f(x)$ sẽ không liên tục tại điểm $x_0$.

Tính liên tục của hàm số là khái niệm quan trọng, tạo nền tảng cho việc định nghĩa đạo hàm.

Ví dụ: Xét hàm số $f(x) = \frac{x^2 - 1}{x - 1}$.

Khi $x$ tiến gần đến 1, ta đã biết giới hạn của hàm số là 2, tuy nhiên $f(1)$ không được xác định.
Do đó, hàm số $f(x)$ không liên tục tại điểm $x = 1$.

Ví dụ: Xét hàm số $g(x) = x^2$.

Hàm số này được xác định tại mọi điểm trên trục số thực, và giới hạn của hàm số khi $x$ tiến gần đến bất kỳ điểm nào $x_0$ cũng tồn tại và bằng giá trị của hàm số tại điểm đó.
Do đó, hàm số $g(x)$ là liên tục tại mọi điểm trên trục số thực.

### 1.3. Đạo hàm của hàm số (Derivatives) và tính khả vi (Differentiability)

Đạo hàm của hàm số tại một điểm đo lường tốc độ biến thiên tức thời của hàm số tại điểm đó.
Nói cách khác, đạo hàm cho biết hàm số thay đổi như thế nào khi đối số thay đổi một cách rất nhỏ.

Đạo hàm của hàm số $f(x)$ tại điểm $x_0$ được định nghĩa như sau:

$$ f'(x_0) = \lim_{x \to x_0} \frac{f(x)-f(x_0)}{x-x_0} $$

Nếu giới hạn trên tồn tại, ta nói rằng hàm số $f(x)$ khả vi tại điểm $x_0$, và $f'(x_0)$ được gọi là đạo hàm của hàm số tại điểm đó.

Nếu đặt 
$$ \Delta x = x - x_0 $$
$$ \Delta f(x) = f(x)-f(x_0) $$

ta có

$$ f'(x_0) = \lim_{x \to x_0} \frac{\Delta f(x)}{\Delta x} $$

trong đó:
- $\Delta x$ là số gia của đối số tại x0.
- $\Delta f(x)$ được gọi là số gia tương ứng của hàm số.

Hàm số $f(x)$ được gọi là khả vi tại điểm $x_0$ nếu hàm số liên tục tại điểm $x_0$ và đạo hàm $f'(x_0)$ tồn tại.
Nếu hàm số khả vi tại mọi điểm trong một khoảng, ta nói hàm số khả vi trên khoảng đó.

Một hàm số khả vi tại một điểm thì hàm số đó cũng liên tục tại điểm đó nhưng điều ngược lại chưa chắc đúng.

## 2. Các quy tắc và công thức tính đạo hàm

### 2.1. Các quy tắc tính đạo hàm

- Đạo hàm của hằng số bằng 0: $c' = 0$
- Đạo hàm của tổng bằng tổng đạo hàm: $(f(x) + g(x))' = f'(x) + g'(x)$
- Đạo hàm của tích: $(f(x)g(x))' = f'(x)g(x) + f(x)g'(x)$
- Đạo hàm của thương: $(\frac{f(x)}{g(x)})' = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}$
- Đạo hàm của hàm hợp (Chain rule): $f'(g(x)) = f'(g)g'(x)$

### 2.2. Đạo hàm của các hàm sơ cấp

- Đạo hàm của hàm đa thức: $(x^n)' = nx^{n-1}$
- Đạo hàm của hàm hằng số nhân với hàm số: $(cf(x))' = c f'(x)$
- Đạo hàm của hàm số mũ: $(e^x)' = e^x$, $(a^x)' = a^x \ln(a)$
- Đạo hàm của hàm logarit: $(\ln(x))' = \frac{1}{x}$, $(\log_a(x))' = \frac{1}{x \ln(a)}$
- Đạo hàm của hàm lượng giác:
    - Hàm sin: $(\sin(x))' = \cos(x)$
    - Hàm cos: $(\cos(x))' = -\sin(x)$
    - Hàm tan: $(\tan(x))' = \sec^2(x)$
- Đạo hàm của hàm ngược lượng giác:
    - Hàm arcsin: $(\arcsin(x))' = \frac{1}{\sqrt{1-x^2}}$
    - Hàm arccos: $(\arccos(x))' = -\frac{1}{\sqrt{1-x^2}}$
    - Hàm arctan: $(\arctan(x))' = \frac{1}{1+x^2}$

### 2.3. Đạo hàm bậc cao

Đạo hàm bậc hai của hàm số là đạo hàm của đạo hàm bậc nhất, ký hiệu là $f''(x)$ hoặc $\frac{d^2f}{dx^2}$.

Tương tự, đạo hàm bậc ba là đạo hàm của đạo hàm bậc hai, ký hiệu là $f'''(x)$ hoặc $\frac{d^3f}{dx^3}$.

Đạo hàm bậc n được định nghĩa tương tự và ký hiệu là $f^{(n)}(x)$ hoặc $\frac{d^nf}{dx^n}$.

### 2.4. Đạo hàm hàm số nhiều biến - Đạo hàm với vector và ma trận

#### Khái niệm đạo hàm riêng

Khi hàm số có nhiều biến, ta có thể tính đạo hàm riêng của hàm số theo từng biến một.
Đạo hàm riêng của hàm số giúp ta hiểu được mức độ ảnh hưởng của từng biến đến sự thay đổi của hàm số, trong khi giữ các biến khác cố định.

Ký hiệu đạo hàm riêng của hàm số $f(x_1, x_2, ..., x_n)$ theo biến $x_i$ được viết là $\frac{\partial f}{\partial x_i}$.

Ví dụ: Xét hàm số $f(x, y) = x^2y + 3xy^2$.
- Đạo hàm riêng của hàm số theo biến $x$ là: $\frac{\partial f}{\partial x} = 2xy + 3y^2$.
- Đạo hàm riêng của hàm số theo biến $y$ là: $\frac{\partial f}{\partial y} = x^2 + 6xy$.

#### Tính đạo hàm với hàm số nhận đầu vào là một vector, trả đầu ra là một số vô hướng

Xét hàm số $f(\mathbf{x}): \mathbb{R}^n \rightarrow \mathbb{R}$, đạo hàm của hàm $f$ theo vector $x$ được định nghĩa như sau

$$
\nabla_{\mathbf{x}} f(\mathbf{x}) \triangleq 
\left[
\begin{matrix}
\frac{\partial f(\mathbf{x})}{\partial x_1} \ 
\frac{\partial f(\mathbf{x})}{\partial x_2} \ 
... \ 
\frac{\partial f(\mathbf{x})}{\partial x_n}
\end{matrix}
\right] \in \mathbb{R}^n
$$

trong đó
- $\frac{\partial f(\mathbf{x})}{\partial x_i}$ là đạo hàm của hàm $f$ theo thành phần thứ $i$ của vector $x$, và các thành phần còn lại là hằng số.

Đạo hàm của hàm số này là một vector có cùng chiều với vector đang lấy đạo hàm.
Đạo hàm bậc hai của hàm số trên là một ma trận vuông đối xứng được định nghĩa như sau:

$$
\nabla^2 f(\mathbf{x}) \triangleq
\left[
\begin{matrix}
    \frac{\partial^2 f(\mathbf{x})}{\partial x_1^2} & \frac{\partial^2 f(\mathbf{x})}{\partial x_1x_2} & ... & \frac{\partial^2 f(\mathbf{x})}{\partial x_1x_n} \\ 
    \frac{\partial^2 f(\mathbf{x})}{\partial x_2x_1} & \frac{\partial^2 f(\mathbf{x})}{\partial x_2^2} & ... & \frac{\partial^2 f(\mathbf{x})}{\partial x_2x_n} \\ 
    ... & ... & \ddots & ... \\
    \frac{\partial^2 f(\mathbf{x})}{\partial x_nx_1} & \frac{\partial^2 f(\mathbf{x})}{\partial x_nx_2} & ... & \frac{\partial^2 f(\mathbf{x})}{\partial x_n^2} \\ 
\end{matrix}
\right] \in \mathbb{S}^{n}
$$

#### Tính đạo hàm với hàm số nhận đầu vào là một ma trận, trả đầu ra là một số vô hướng

Xét hàm số $f(\mathbf{x}): \mathbb{R}^{n \times m} \rightarrow \mathbb{R}$, đạo hàm của hàm $f$ theo ma trận $x$ được định nghĩa như sau

$$
\nabla_{\mathbf{x}} f(\mathbf{x}) \triangleq 
\left[
\begin{matrix}
    \frac{\partial f(\mathbf{X})}{\partial x_{11}} & \frac{\partial f(\mathbf{X})}{\partial x_{12}} & ... & \frac{\partial f(\mathbf{X})}{\partial x_{1m}} \\
    \frac{\partial f(\mathbf{X})}{\partial x_{21}} & \frac{\partial f(\mathbf{X})}{\partial x_{22}} & ... & \frac{\partial f(\mathbf{X})}{\partial x_{2m}} \\
    ... & ... & \ddots & ... \\
    \frac{\partial f(\mathbf{X})}{\partial x_{n1}} & \frac{\partial f(\mathbf{X})}{\partial x_{n2}} & ... & \frac{\partial f(\mathbf{X})}{\partial x_{nm}} 
\end{matrix}
\right] \in \mathbb{R}^{n \times m}
$$

Đạo hàm của một hàm số theo ma trận là một ma trận có chiều giống với ma trận đó.

Tóm lại, đối với hàm số nhận đầu vào là một vector (hoặc một ma trận), đạo hàm của hàm số đó cũng là một vector (hoặc một ma trận) trong đó các phần tử được tính bằng cách đạo hàm hàm số đó với các phần tử ở vị trí tương ứng là biến, các phần tử còn lại được coi là hằng số.

Ví dụ: Xét hàm số $f: \mathbb{R}^2 \rightarrow \mathbb{R}$, $f(\mathbf{x}) = x_1 ^2 + 2x_1x_2 + \sin(x_1) + 2$.

Đạo hàm bậc nhất của hàm số là

$$
\nabla f(\mathbf{x}) =
\left[
\begin{matrix}
    \frac{\partial f(\mathbf{x})}{\partial x_1} \ 
    \frac{\partial f(\mathbf{x})}{\partial x_2}
\end{matrix}
\right] = \left[
\begin{matrix}
    2x_1 + 2x_2 + \cos(x_1) \ 
    2x_1
\end{matrix}
\right]
$$

Đạo hàm bậc hai của hàm số là

$$
\nabla^2 f(\mathbf{x}) = 
\left[
\begin{matrix}
    \frac{\partial^2 f(\mathbf{x})}{\partial x_1^2} & \frac{\partial f^2(\mathbf{x})}{\partial x_1x_2} \\
    \frac{\partial^2 f(\mathbf{x})}{\partial x_2x_1} & \frac{\partial f^2(\mathbf{x})}{\partial x_2^2}
\end{matrix}
\right] =
\left[
\begin{matrix}
    2 - \sin(x_1) & 2 \\
    2 & 0 
\end{matrix}
\right]
$$

#### Tính đạo hàm với hàm số nhận đầu vào là một số vô hướng, trả đầu ra là một vector

Xét hàm số $f(x): \mathbb{R} \rightarrow \mathbb{R}^n$, cụ thể

$$
v(x) = 
\left[
\begin{matrix}
    v_1(x) \\
    v_2(x) \\
    ... \\
    v_n(x)
\end{matrix}
\right]
$$

Đạo hàm của hàm $f$ theo giá trị $x$ được định nghĩa như sau

$$
\nabla v(x) \triangleq 
\left[
\begin{matrix}
    \frac{\partial v_1(x)}{\partial x} & \frac{\partial v_2(x)}{\partial x} & ... & \frac{\partial v_n(x)}{\partial x}
\end{matrix}
\right]
$$

Và đạo hàm bậc hai của $f$ theo giá trị $x$ có dạng

$$
\nabla^2 v(x) \triangleq 
\left[
\begin{matrix}
    \frac{\partial^2 v_1(x)}{\partial x^2} & \frac{\partial^2 v_2(x)}{\partial x^2} & ... & \frac{\partial^2 v_n(x)}{\partial x^2}
\end{matrix}
\right]
$$

#### Tính đạo hàm với hàm số nhận đầu vào là một vector, trả đầu ra là một vector

Xét hàm số $f(x): \mathbb{R}^k \rightarrow \mathbb{R}^n$, lúc này, đạo hàm bậc nhất của nó là

$$
\begin{eqnarray}
\nabla h(\mathbf{x}) &\triangleq &
\left[
\begin{matrix}
    \frac{\partial h_1(\mathbf{x})}{\partial x_1} & \frac{\partial h_2(\mathbf{x})}{\partial x_1} & ... & \frac{\partial h_n(\mathbf{x})}{\partial x_1} \\ 
    \frac{\partial h_1(\mathbf{x})}{\partial x_2} & \frac{\partial h_2(\mathbf{x})}{\partial x_2} & ... & \frac{\partial h_n(\mathbf{x})}{\partial x_2} \\ 
    ... & ... & \ddots & ... \\
    \frac{\partial h_1(\mathbf{x})}{\partial x_k} & \frac{\partial h_2(\mathbf{x})}{\partial x_k} & ... & \frac{\partial h_n(\mathbf{x})}{\partial x_k}
\end{matrix}
\right] \\ 
& = & 
\left[
\begin{matrix}
    \nabla h_1(\mathbf{x}) & \nabla h_2(\mathbf{x}) & ... & \nabla h_n(\mathbf{x})
\end{matrix}
\right] \in \mathbf{R}^{k\times n}
\end{eqnarray}
$$

## 3. Một số ứng dụng trong Machine Learning

### 3.1. Thuật toán tối ưu Gradient Descent

Gradient Descent là một thuật toán tối ưu hóa phổ biến trong machine learning, sử dụng đạo hàm để tìm cực tiểu của hàm mất mát.

Hàm mất mát $L(\theta)$ đo lường sự khác biệt giữa dự đoán của mô hình và giá trị thực tế.
Ta luôn muốn tìm tham số $\theta$ sao cho hàm mất mát đạt giá trị nhỏ nhất.

Từ đó, ta sử dụng đạo hàm của hàm mất mát để xác định hướng giảm giá trị hàm mất mát, giúp xấp xỉ tìm được giá trị tối ưu của tham số mô hình.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/local_global_minimum.jpeg" style="width: 600px;"/>

Thuật toán Gradient Descent đóng vai trò quan trọng trong nhiều mô hình machine learning, bao gồm hồi quy tuyến tính, hồi quy logistic và mạng nơ-ron sâu (deep neural networks).

Để hiểu rõ hơn về thuật toán Gradient Descent, bạn có thể tham khảo bài viết [Thuật toán tối ưu Gradient Descent](/blog/thuat-toan-toi-uu-gradient-descent) và [Các biến thể nâng cấp của thuật toán tối ưu Gradient descent](/blog/cac-bien-the-nang-cap-cua-thuat-toan-toi-uu-gradient-descent).

### 3.2. Backpropagation trong mạng nơ-ron

Backpropagation - lan truyền ngược là một khái niệm đi kèm khi sử dụng thuật toán Gradient Descent để huấn luyện mạng nơ-ron.

Nó sử dụng đạo hàm để tính toán gradient của hàm mất mát đối với các tham số của mạng nơ-ron, từ đó cập nhật các tham số này nhằm giảm thiểu hàm mất mát.

Quy tắc cốt lõi trong backpropagation là quy tắc chuỗi (chain rule), cho phép tính đạo hàm của hàm hợp.
Từ đó, ta có thể tính toán gradient của hàm mất mát đối với từng tham số trong mạng nơ-ron từ cuối của mạng nơ-ron trở về đầu của mạng (đây là lý do tại sao gọi là lan truyền ngược).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/1-neural-network/dnn.jpeg" style="width: 500px;"/>

Để hiểu rõ hơn về backpropagation trong mô hình mạng nơ ron, bạn có thể tham khảo bài viết [Mô hình mạng nơ ron đơn giản Neural network](/blog/mo-hinh-mang-no-ron-don-gian-neural-network).
