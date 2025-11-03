---
time: 07/19/2022
title: Thuật toán tối ưu Gradient Descent
description: Trong Machine Learning, ta thường dùng phép toán đạo hàm để tối ưu hàm loss. Tuy nhiên, trong đa số các trường hợp, việc tính đạo hàm của hàm loss là không thể hoặc rất khó khăn, đặc biệt khi kiến trúc mô hình phức tạp và bộ dữ liệu lớn. Trong bài viết này, chúng ta sẽ tìm hiểu về thuật toán tối ưu Gradient Descent, một trong những phương pháp phổ biến nhất để tối ưu hàm loss trong Machine Learning.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/banner.jpeg
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Nhắc lại về bài toán Khảo sát hàm số

Giả sử, ta có hàm số bậc 2 như sau:

$$f(x) = x^2 + 10x + 4$$

Để tìm giá trị nhỏ nhất của hàm số, ta cần tìm giá trị của $x$ sao cho $f'(x) = 0$. Tức là, ta cần tìm giá trị của $x$ sao cho đạo hàm của hàm số bằng 0.

$$f'(x) = 2x + 10 = 0$$
$$\Rightarrow x = -5$$
$$\Rightarrow  f(-5) = (-5)^2 + 10 \times (-5) + 4 = -21$$

Ta thu được nghiệm tối ưu của bài toán là $x = -5$ và giá trị tối ưu của hàm số là $f(-5) = -21$.
Từ đó, ta vẽ được đồ thị của hàm số như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/example_graph.jpeg" style="width: 600px;"/>

Trong Machine Learning, ta thường phải đi tìm nghiệm tối ưu và giá trị tối ưu của hàm loss (thông thường giá trị tối ưu chính là giá trị cực tiểu của hàm loss).
Ví dụ trong mô hình Linear Regression, ta tìm giá trị tối ưu của hàm loss Mean Squared Error (MSE).

Trong toán học, ta có giá trị cực tiểu địa phương và giá trị cực tiểu toàn cục.
- Giá trị cực tiểu địa phương (local minimum) là giá trị nhỏ nhất của hàm số trong một khoảng lân cận xác định.
- Giá trị cực tiểu toàn cục là (global minimum) giá trị nhỏ nhất của hàm số trên toàn miền xác định.
- Global minimum là một trường hợp đặc biệt của local minimum, là local minimum nhỏ nhất trong toàn bộ miền xác định.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/local_global_minimum.jpeg" style="width: 600px;"/>

Về lý thuyết, ta luôn mong muốn tìm được global minimum của hàm loss.
Tuy nhiên, trong thực tế, việc tìm được global minimum là không dễ dàng, đặc biệt khi hàm loss có dạng phức tạp, bộ dữ liệu lớn, hoặc mô hình phức tạp.

Vì vậy, thay vì việc tính toán trực tiếp ra các giá trị minimim, ta cần một giải pháp để xấp xỉ được các giá trị này.
Và đó là thuật toán Gradient Descent.

Ngoài ra, trong thực tế, việc tìm được nghiệm tối ưu ở local minimum cũng là một kết quả tốt nếu giá trị của hàm loss ở local minimum là đủ nhỏ.
Từ đó, các mô hình Machine Learning thường sử dụng thuật toán Gradient Descent để xấp xỉ local minimum của hàm loss.

## 2. Ý tưởng của thuật toán Gradient Descent

Xét ví dụ về hàm số như trên $f(x) = x^2 + 10x + 4$, ta đã có nghiệm tối ưu của hàm số là $x = -5$ và giá trị tối ưu của hàm số là $f(-5) = -21$.

Ta sẽ sử dụng thuật toán Gradient Descent để xấp xỉ nghiệm tối ưu và giá trị tối ưu này của hàm số.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/fx_vs_gradient_fx.jpeg" style="width: 600px;"/>

Ta ký hiệu nghiệm tối ưu của hàm số là $x^*$ và giá trị tối ưu của hàm số là $f(x^*)$.
Xét giá trị $x^t$ nào đó bất kỳ, ta có

$$f'(x^t) > 0 \leftrightarrow x^t > x^*$$

$$f'(x^t) < 0 \leftrightarrow x^t < x^*$$

giả sử, ta đặt $x^t + \delta = x^*$, ta có

$$f'(x^t) > 0 \leftrightarrow x^t > x^* \leftrightarrow \delta < 0$$

$$f'(x^t) < 0 \leftrightarrow x^t < x^* \leftrightarrow \delta > 0$$

Ta rút ra được $\delta$ và $f'(x^t)$ luôn luôn có mối quan hệ nghịch biến với nhau.
Độ lớn của $|\delta|$ phụ thuộc vào độ lớn của $|f'(x^t)|$.

Từ đó, ta có thể đặt

$$\delta = -\eta \cdot f'(x^t)$$

Lúc này, nếu tính được giá trị của $\delta$ (nghĩa là ta phải tìm được giá trị $\eta$), ta sẽ có thể cập nhật giá trị của $x^t$ để tìm được giá trị $x^*$.

Tuy nhiên, việc ta không có căn cứ nào để tìm được giá trị $\eta$ một cách chính xác, nên ta thường chọn giá trị $\eta$ nhỏ và lặp đi lặp lại quá trình cập nhật giá trị của $x^t$ cho đến khi xấp xỉ được giá trị $x^*$.
Cụ thể, ta sẽ làm như sau:

- Khởi tạo giá trị $x^t = x^0$ bất kỳ.
- Chọn giá trị của $\eta$ là một giá trị nhỏ dương.
- Lặp lại quá trình sau cho đến khi đạt được điều kiện dừng:
    - Tính giá trị của đạo hàm $f'(x^t)$.
    - Cập nhật giá trị của $x^t$ theo công thức $x^{t+1} = x^t - \eta \cdot f'(x^t)$.
    - Lúc này, ta kỳ vọng giá trị của $x^{t+1}$ sẽ gần với giá trị $x^*$ hơn so với giá trị $x^t$.

Và đây chính là ý tưởng của thuật toán Gradient Descent.

Ví dụ trên giúp ta có thể hình dung được ý tưởng của thuật toán Gradient Descent với hàm một biến, và trong thực tế, ta sẽ sử dụng thuật toán Gradient Descent khái quát với hàm nhiều biến.

## 3. Ảnh hưởng của các tham số trong thuật toán Gradient Descent

Hai bước đầu trong thuật toán Gradient Descent (khởi tạo giá trị $x^0$ và chọn giá trị $\eta$) là hai bước quan trọng, ảnh hưởng lớn đến kết quả cuối cùng của thuật toán.

Đối với giá trị khởi tạo, nếu ta chọn giá trị khởi tạo quá xa so với giá trị tối ưu, có thể mất rất nhiều thời gian để thuật toán hội tụ, thuật toán tìm được nghiệm tối ưu.

Trong thực tế, ta thường khởi tạo ngẫu nhiên giá trị $x^0$, và sau này, trong các phiên bản nâng cấp của thuật toán Gradient Descent, ta sẽ cải thiện bằng cách sử dụng các phương pháp khởi tạo giá trị $x^0$ tốt hơn hoặc tinh chỉnh thuật toán để giảm thiểu ảnh hưởng của giá trị khởi tạo.

<p style="float: left;">
    <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/x_0_5_learning_rate_0p1.gif" style="width: 500px;"/>
</p>
<p style="float: right;">
    <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/x_0_-3_learning_rate_0p1.gif" style="width: 500px;"/>
</p>
<br style="clear: both;"/>

Giá trị $\eta$ trong thuật toán Gradient Descent còn được gọi là learning rate, là một tham số quan trọng, ảnh hưởng đến tốc độ hội tụ và khả năng hội tụ của thuật toán.

Trong thực tế, ta bắt buộc phải lựa chọn giá trị learning rate này.
Một số giá trị learning rate phổ biến như 0.1, 0.03, 0.001, 0.0003, ... và ta cần phải thử nghiệm nhiều giá trị learning rate để tìm ra giá trị tốt nhất cho bài toán cụ thể.
Sau này, trong các phiên bản nâng cấp của thuật toán Gradient Descent, ta sẽ cải thiện bằng cách sử dụng các phương pháp tinh chỉnh learning rate tự động.

Khi ta chọn giá trị learning rate quá nhỏ, thuật toán sẽ hội tụ rất chậm, và có thể mất rất nhiều thời gian để thuật toán hội tụ.

<p style="float: left;">
    <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/x_0_5_learning_rate_0p1.gif" style="width: 500px;"/>
</p>
<p style="float: right;">
    <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/x_0_5_learning_rate_0p01.gif" style="width: 500px;"/>
</p>
<br style="clear: both;"/>

Ngược lại, khi ta chọn giá trị learning rate quá lớn, thuật toán có thể không hội tụ được, và giá trị của hàm loss có thể tăng lên sau mỗi lần cập nhật.

<p style="float: left;">
    <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/x_0_5_learning_rate_0p1.gif" style="width: 500px;"/>
</p>
<p style="float: right;">
    <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/x_0_5_learning_rate_0p99.gif" style="width: 500px;"/>
</p>
<br style="clear: both;"/>
