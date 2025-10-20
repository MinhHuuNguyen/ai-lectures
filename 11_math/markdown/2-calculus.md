---
time: 01/07/2023
title: Giải tích toán học Calculus
description: Giải tích toán học (Calculus) đóng vai trò trọng yếu trong machine learning, đặc biệt trong việc tối ưu hóa mô hình. Các khái niệm như đạo hàm, vi phân và gradient được sử dụng để cập nhật tham số mô hình nhằm giảm thiểu hàm mất mát. Nhờ giải tích, thuật toán gradient descent và các biến thể của nó có thể tìm ra điểm cực tiểu của hàm mục tiêu, giúp mô hình học được từ dữ liệu. Ngoài ra, giải tích còn hỗ trợ hiểu rõ sự biến thiên của đầu ra theo đầu vào, góp phần vào việc phân tích độ nhạy và ổn định của hệ thống học máy.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/2-probability/banner.png
tags: [math]
is_highlight: false
is_published: true
---

# Đạo hàm với vector và ma trận

## 1. Đạo hàm cơ bản

### 1.1. Đạo hàm là gì?

Theo toán học giải tích, đạo hàm thực chất là một đại lượng được dùng để mô tả sự biến thiên của hàm số tại một điểm nào đó.

$$
f'(x_0) = \lim_{x \to x_0} \frac{f(x)-f(x_0)}{x-x_0}
$$

Nếu đặt 
$$
\Delta x = x - x_0
\Delta f(x) = f(x)-f(x_0)
$$

ta có

$$
f'(x_0) = \lim_{x \to x_0} \frac{\Delta f(x)}{\Delta x}
$$

trong đó:

$\Delta x$ là số gia của đối số tại x0.
$\Delta f(x)$ được gọi là số gia tương ứng của hàm số.

### 1.2. Các quy tắc tính đạo hàm
- Đạo hàm của hằng số bằng 0:

$$
c' = 0
$$

- Đạo hàm của tổng bằng tổng đạo hàm:

$$
(f(x) + g(x))' = f'(x) + g'(x)
$$

- Đạo hàm của tích (Product rule):

$$
(f(x)g(x))' = f'(x)g(x) + f(x)g'(x)
$$

- Đạo hàm của thương:

$$
(\frac{f(x)}{g(x)})' = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}
$$

- Đạo hàm của hàm hợp (Chain rule):

$$
f'(g(x)) = f'(g)g'(x)
$$


### 1.3. Đạo hàm của các hàm sơ cấp

<img src="https://giasuhanoigioi.edu.vn/wp-content/uploads/2019/03/cong-thuc-dao-ham-3.png" style="height: 600px;"/>


## 2. Đạo hàm với vector và ma trận

## 2.1. Hàm số có đầu ra là một số vô hướng

### 2.1.1. Hàm số nhận đầu vào là một vector

Xét hàm số $f(\mathbf{x}): \mathbb{R}^n \rightarrow \mathbb{R}$, đạo hàm của hàm $f$ theo vector $x$ được định nghĩa như sau

$$
\nabla_{\mathbf{x}} f(\mathbf{x}) \triangleq 
\left[
\begin{matrix}
\frac{\partial f(\mathbf{x})}{\partial x_1} \ 
\frac{\partial f(\mathbf{x})}{\partial x_2} \ 
\vdots \ 
\frac{\partial f(\mathbf{x})}{\partial x_n}
\end{matrix}
\right] \in \mathbb{R}^n
$$

trong đó

$\frac{\partial f(\mathbf{x})}{\partial x_i}$ là đạo hàm của hàm $f$ theo thành phần thứ $i$ của vector $x$, và các thành phần còn lại là hằng số.

Đạo hàm của hàm số này là một vector có cùng chiều với vector đang lấy đạo hàm.

Đạo hàm bậc hai của hàm số trên là một ma trận vuông đối xứng được định nghĩa như sau:

$$
\nabla^2 f(\mathbf{x}) \triangleq
\left[
\begin{matrix}
    \frac{\partial^2 f(\mathbf{x})}{\partial x_1^2} & \frac{\partial^2 f(\mathbf{x})}{\partial x_1x_2} & \dots & \frac{\partial^2 f(\mathbf{x})}{\partial x_1x_n} \\ 
    \frac{\partial^2 f(\mathbf{x})}{\partial x_2x_1} & \frac{\partial^2 f(\mathbf{x})}{\partial x_2^2} & \dots & \frac{\partial^2 f(\mathbf{x})}{\partial x_2x_n} \\ 
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial^2 f(\mathbf{x})}{\partial x_nx_1} & \frac{\partial^2 f(\mathbf{x})}{\partial x_nx_2} & \dots & \frac{\partial^2 f(\mathbf{x})}{\partial x_n^2} \\ 
\end{matrix}
\right] \in \mathbb{S}^{n}
$$

### 2.1.2. Hàm số nhận đầu vào là một ma trận

Xét hàm số $f(\mathbf{x}): \mathbb{R}^{n \times m} \rightarrow \mathbb{R}$, đạo hàm của hàm $f$ theo ma trận $x$ được định nghĩa như sau

$$
\nabla_{\mathbf{x}} f(\mathbf{x}) \triangleq 
\left[
\begin{matrix}
    \frac{\partial f(\mathbf{X})}{\partial x_{11}} & \frac{\partial f(\mathbf{X})}{\partial x_{12}} & \dots & \frac{\partial f(\mathbf{X})}{\partial x_{1m}} \\
    \frac{\partial f(\mathbf{X})}{\partial x_{21}} & \frac{\partial f(\mathbf{X})}{\partial x_{22}} & \dots & \frac{\partial f(\mathbf{X})}{\partial x_{2m}} \\
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial f(\mathbf{X})}{\partial x_{n1}} & \frac{\partial f(\mathbf{X})}{\partial x_{n2}} & \dots & \frac{\partial f(\mathbf{X})}{\partial x_{nm}} 
\end{matrix}
\right] \in \mathbb{R}^{n \times m}
$$

Đạo hàm của một hàm số theo ma trận là một ma trận có chiều giống với ma trận đó.

Tóm lại, đối với hàm số nhận đầu vào là một vector (hoặc một ma trận), đạo hàm của hàm số đó cũng là một vector (hoặc một ma trận) trong đó các phần tử được tính bằng cách đạo hàm hàm số đó với các phần tử ở vị trí tương ứng là biến, các phần tử còn lại được coi là hằng số.

Ví dụ:

Xét hàm số $f: \mathbb{R}^2 \rightarrow \mathbb{R}$, $f(\mathbf{x}) = x_1 ^2 + 2x_1x_2 + \sin(x_1) + 2$.

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

## 2.2. Hàm số có đầu ra là một vector

### 2.2.1. Hàm số nhận đầu vào là một số vô hướng

Xét hàm số $f(x): \mathbb{R} \rightarrow \mathbb{R}^n$, cụ thể

$$
v(x) = 
\left[
\begin{matrix}
    v_1(x) \\
    v_2(x) \\
    \dots \\
    v_n(x)
\end{matrix}
\right]
$$

Đạo hàm của hàm $f$ theo giá trị $x$ được định nghĩa như sau

$$
\nabla v(x) \triangleq 
\left[
\begin{matrix}
    \frac{\partial v_1(x)}{\partial x} & \frac{\partial v_2(x)}{\partial x} & \dots & \frac{\partial v_n(x)}{\partial x}
\end{matrix}
\right]
$$

Và đạo hàm bậc hai của $f$ theo giá trị $x$ có dạng

$$
\nabla^2 v(x) \triangleq 
\left[
\begin{matrix}
    \frac{\partial^2 v_1(x)}{\partial x^2} & \frac{\partial^2 v_2(x)}{\partial x^2} & \dots & \frac{\partial^2 v_n(x)}{\partial x^2}
\end{matrix}
\right]
$$

### 2.2.2. Hàm số nhận đầu vào là một vector

Xét hàm số $f(x): \mathbb{R}^k \rightarrow \mathbb{R}^n$, lúc này, đạo hàm bậc nhất của nó là

$$
\begin{eqnarray}
\nabla h(\mathbf{x}) &\triangleq &
\left[
\begin{matrix}
    \frac{\partial h_1(\mathbf{x})}{\partial x_1} & \frac{\partial h_2(\mathbf{x})}{\partial x_1} & \dots & \frac{\partial h_n(\mathbf{x})}{\partial x_1} \\ 
    \frac{\partial h_1(\mathbf{x})}{\partial x_2} & \frac{\partial h_2(\mathbf{x})}{\partial x_2} & \dots & \frac{\partial h_n(\mathbf{x})}{\partial x_2} \\ 
    \vdots & \vdots & \ddots & \vdots \\
    \frac{\partial h_1(\mathbf{x})}{\partial x_k} & \frac{\partial h_2(\mathbf{x})}{\partial x_k} & \dots & \frac{\partial h_n(\mathbf{x})}{\partial x_k}
\end{matrix}
\right] \\ 
& = & 
\left[
\begin{matrix}
    \nabla h_1(\mathbf{x}) & \nabla h_2(\mathbf{x}) & \dots & \nabla h_n(\mathbf{x})
\end{matrix}
\right] \in \mathbf{R}^{k\times n}
\end{eqnarray}
$$
