---
time: 08/06/2022
title: Mô hình Support Vector Machine (SVM)
description: SVM là mô hình machine learning dựa vào khoảng cách giữa các điểm dữ liệu và đường phân lớp để tìm ra được đường phân lớp tốt nhất. SVM thường là mô hình phân lớp có độ chính xác cao hơn so với mô hình Logistic Regression (mạng nơ ron với 1 layer).
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/banner.png
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Tổng quan

Support Vector Machine (SVM) là một trong những thuật toán học có giám sát (supervised learning) dựa vào khoảng cách.

Câu hỏi mà SVM đặt ra là: "Trong các đường phân lớp được tạo ra từ các mô hình classification khác nhau, đường nào là đường phân lớp tốt nhất?".

Mô hình SVM đề xuất ra các tiêu chí và phương pháp để tìm ra đường phân lớp tốt nhất trong số các đường phân lớp có thể có.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/comparison.png" style="width: 600px;"/>

## 2. Công thức tính khoảng cách từ 1 điểm

Trong không gian 2 chiều, một đường thẳng là thành phần phân chia mặt phẳng 2 chiều thành 2 nửa mặt phẳng.
Trong không gian 3 chiều, một mặt phẳng là thành phần phân chia không gian 3 chiều thành 2 nửa không gian.
Khái quát hoá trong không gian n chiều, thành phần phân chia không gian n chiều thành 2 nửa không gian được gọi là siêu phẳng.

Ta có, trong không gian 2 chiều, một đường có thể được biểu diễn bằng phương trình:

$$ w_1x_1 + w_2x_2 + b = 0 $$

và công thức tính khoảng cách từ 1 điểm có tọa độ $(x_1, x_2)$ đến đường thẳng là:

$$ d = \frac{|w_1x_1 + w_2x_2 + b|}{\sqrt{w_1^2 + w_2^2}} $$

Trong không gian 3 chiều, một mặt phẳng có thể được biểu diễn bằng phương trình:

$$ w_1x_1 + w_2x_2 + w_3x_3 + b = 0 $$

và công thức tính khoảng cách từ 1 điểm có tọa độ $(x_1, x_2, x_3)$ đến mặt phẳng là:

$$ d = \frac{|w_1x_1 + w_2x_2 + w_3x_3 + b|}{\sqrt{w_1^2 + w_2^2 + w_3^2}} $$

Trong không gian n chiều, một siêu phẳng có thể được biểu diễn bằng phương trình:

$$ w_1x_1 + w_2x_2 + ... + w_nx_n + b = 0 $$

và công thức tính khoảng cách từ 1 điểm có tọa độ $(x_1, x_2, ..., x_n)$ đến siêu phẳng là:

$$ d = \frac{|w_1x_1 + w_2x_2 + ... + w_nx_n + b|}{\sqrt{w_1^2 + w_2^2 + ... + w_n^2}} $$

Và mô hình SVM sử dụng rất nhiều công thức tính khoảng cách nói trên.

## 3. Mô hình SVM - Hard Margin SVM

Mô hình SVM tìm ra đường phân lớp tốt nhất bằng cách tối ưu hoá khoảng cách giữa các điểm dữ liệu và đường phân lớp.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/compare_distance.png" style="width: 600px;"/>

Trong hình trên, khi so sánh khoảng cách giữa điểm dữ liệu gần nhất của mỗi lớp (lớp vàng và lớp tím đậm) với hai đường phân lớp xanh dương và đỏ:
- Khoảng cách giữa điểm dữ liệu gần nhất của lớp vàng đến đường phân lớp đỏ ngắn hơn so với khoảng cách giữa điểm dữ liệu gần nhất của lớp tím đậm đến đường phân lớp đỏ.
- Khoảng cách giữa điểm dữ liệu gần nhất của lớp vàng đến đường phân lớp xanh dương bằng với khoảng cách giữa điểm dữ liệu gần nhất của lớp tím đậm đến đường phân lớp xanh dương.

Vậy trong trường hợp này, đường phân lớp xanh dương là đường phân lớp tốt hơn so với đường phân lớp đỏ.

Mô hình SVM cho rằng đường phân lớp tốt là đường phân lớp công bằng với các lớp dữ liệu, nói cách khác, đường phân lớp tốt là đường phân lớp có khoảng cách bằng nhau với các điểm dữ liệu gần nhất của các lớp dữ liệu.

Ngoài ra, SVM cũng gọi khoảng cách giữa điểm dữ liệu gần nhất của mỗi lớp đến đường phân lớp là **margin**.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/compare_margin.png" style="width: 600px;"/>

Trong hình trên, ta thấy:
- Margin của lớp vàng đến đường phân lớp xanh dương bằng với margin của lớp tím đến đường phân lớp xanh dương.
- Margin của lớp vàng đến đường phân lớp đỏ bằng với margin của lớp tím đậm đến đường phân lớp tím.
- Tuy nhiên, margin của đường phân lớp xanh dương lớn hơn margin của đường phân lớp tím.

Vậy trong trường hợp này, đường phân lớp xanh dương là đường phân lớp tốt hơn so với đường phân lớp tím.

Mô hình SVM tìm ra đường phân lớp tốt nhất, không những margin giữa các lớp là bằng nhau mà còn lớn nhất.

Về lý thuyết, mô hình SVM luôn tìm ra đường phân lớp tốt nhất so sánh với mô hình Logistic Regression (mạng nơ ron với 1 layer).

## 4. Tối ưu mô hình Hard Margin SVM

Xét bộ dữ liệu huấn luyện trong không gian hai chiều $D = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$, trong đó, label $y_i$ có thể nhận giá trị $+1$ hoặc $-1$.

Ta gọi phương trình đường phân lớp trong không gian hai chiều là:
$$ w_1x_1 + w_2x_2 + b = 0 $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/complex_optimization.png" style="width: 600px;"/>

Từ đó, với mỗi điểm dữ liệu $(x_i, y_i)$, ta có thể tính được khoảng cách từ điểm dữ liệu đến đường phân lớp là:

$$ d_i = \frac{|w_1x_{i1} + w_2x_{i2} + b|}{\sqrt{w_1^2 + w_2^2}} $$

Ta lấy giá trị nhỏ nhất trong các khoảng cách $d_i$, ta thu được margin là:

$$ margin = \min_{i=1}^n d_i = \min_{i=1}^n \frac{|w_1x_{i1} + w_2x_{i2} + b|}{\sqrt{w_1^2 + w_2^2}} $$

Ta có hàm mục tiêu để thực hiện bài toán tối ưu mô hình SVM là:

$$ \max_{w, b} (\min_{i=1}^n d_i) = \max_{w, b} (\min_{i=1}^n \frac{|w_1x_{i1} + w_2x_{i2} + b|}{\sqrt{w_1^2 + w_2^2}}) $$

Khi ta cực đại hoá khoảng cách nhỏ nhất giữa các điểm dữ liệu và đường phân lớp, ta sẽ thu được margin lớn nhất có thể và margin này sẽ bằng nhau cho tất cả các lớp dữ liệu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/simple_optimization.png" style="width: 600px;"/>

Tuy nhiên, hàm mục tiêu phía trên rất khó để tối ưu, do đó, ta biến đổi hàm mục tiêu trên bằng việc quy ước hai đường thẳng song song với đường phân lớp là:

$$ w_1x_1 + w_2x_2 + b = 1 $$

$$ w_1x_1 + w_2x_2 + b = -1 $$

Lúc này, ta có thể viết lại hàm mục tiêu trên thành:

$$ \max_{w, b} (\min_{i=1}^n d_i) = \max_{w, b} (\frac{1}{\sqrt{w_1^2 + w_2^2}}) $$

với điều kiện ràng buộc là:

$$ y_i(w_1x_{i1} + w_2x_{i2} + b) \geq 1 $$

Ta tối ưu hàm mục tiêu trên bằng cách sử dụng các thuật toán tối ưu lồi (convex optimization) như phương pháp nhân tử Lagrange (Lagrange multiplier method) hoặc phương pháp gradient descent.

## 5. Mô hình Soft Margin SVM

Mô hình SVM mà ta đã đề cập ở trên là mô hình Hard Margin SVM.
Hard Margin SVM chỉ hoạt động được khi dữ liệu là phân lớp tuyến tính (linearly separable), tức là có thể tìm ra được đường phân lớp mà không có điểm dữ liệu nào bị phân lớp sai.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/linearly_separable.png" style="width: 600px;"/>

Nếu trường hợp dữ liệu có điểm dữ liệu nhiễu (noise), khi đó, mô hình Hard Margin SVM vẫn có thể tìm ra được đường phân lớp tốt nhất, tuy nhiên, margin lúc này sẽ rất nhỏ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/linearly_separable_with_noise.png" style="width: 600px;"/>

Hơn nữa, trong trường hợp bộ dữ liệu phân bố "almost linearly separable", tức là khái quát thì các lớp dữ liệu vẫn phân tách được với nhau, tuy nhiên, có một vài điểm dữ liệu thuộc lớp này nhưng lại nằm rất gần với các điểm dữ liệu thuộc lớp khác.
Trong trường hợp này, mô hình Hard Margin SVM sẽ không thể tìm ra được đường phân lớp, hay nói cách khác, Hard Margin SVM không thể hoạt động được, vô nghiệm.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/almost_linearly_separable.png" style="width: 600px;"/>

Để giải quyết vấn đề này, mô hình SVM đã được mở rộng thành mô hình Soft Margin SVM.
Mô hình Soft Margin SVM cho phép một số điểm dữ liệu bị phân lớp sai hoặc một số điểm dữ liệu nằm trong vùng nguy hiểm, tức là nằm trong khoảng cách margin.

Soft Margin SVM đánh đổi một số điểm dữ liệu bị phân lớp sai hoặc nguy hiểm với việc tăng độ rộng của margin, từ đó, giúp mô hình vẫn có thể hoạt động được trong trường hợp dữ liệu ""almost linearly separable".

Để làm được điều này, mô hình Soft Margin SVM thêm một số hạng mới vào hàm mục tiêu của mô hình Hard Margin SVM.
Hàm mục tiêu của mô hình Soft Margin SVM là:

$$ \max_{w, b} (\frac{1}{\sqrt{w_1^2 + w_2^2}}) - C \sum_{i=1}^n \xi_i $$

trong đó giá trị $\xi_i$ có các trường hợp sau:
- Nếu điểm dữ liệu được phân lớp đúng và nằm ngoài margin, thì $\xi_i = 0$.
- Nếu điểm dữ liệu được phân lớp đúng và nằm trong margin, thì $\xi_i = d_i$ và $d_i < 1$.
- Nếu điểm dữ liệu bị phân lớp sai, thì $\xi_i = d_i$ và $d_i \geq 1$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/c_value_comparison.png" style="width: 900px;"/>

Tham số $C$ là trong hàm mục tiêu trên là tham số điều chỉnh để cân bằng giữa độ rộng của margin và số điểm dữ liệu bị phân lớp sai hoặc nằm trong vùng nguy hiểm.
- $C$ càng lớn thì độ rộng của margin càng nhỏ, số lượng điểm dữ liệu bị phân lớp sai hoặc nằm trong vùng nguy hiểm càng ít.
- $C$ càng nhỏ thì độ rộng của margin càng lớn, số lượng điểm dữ liệu bị phân lớp sai hoặc nằm trong vùng nguy hiểm càng nhiều.
- Nếu tham số $C$ bằng 0, thì mô hình Soft Margin SVM trở thành mô hình Hard Margin SVM.

## 6. Mô hình Kernel SVM

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/kernel_svm_1.png" style="width: 600px;"/>

Mô hình Hard Margin SVM hoạt động tốt với bộ dữ liệu linearly separable.
Mô hình Soft Margin SVM hoạt động tốt với bộ dữ liệu "almost linearly separable".
Tuy nhiên, trong trường hợp bộ dữ liệu không thể phân lớp được bằng đường thẳng (non-linearly separable), thì cả hai mô hình trên đều không hoạt động được.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/kernel_svm_2.png" style="width: 600px;"/>

Lúc này, mô hình SVM đã được mở rộng thành mô hình Kernel SVM.
- Đầu tiên, Kernel SVM sử dụng một hàm kernel để biến đổi không gian dữ liệu đầu vào thành không gian dữ liệu mới, trong đó các lớp dữ liệu có thể phân lớp được bằng đường thẳng.
- Sau đó, mô hình Kernel SVM sẽ tìm ra đường phân lớp tốt nhất trong không gian dữ liệu mới này, tương tự như mô hình Hard Margin SVM hoặc Soft Margin SVM.
- Cuối cùng, mô hình Kernel SVM sẽ biến đổi đường phân lớp trong không gian dữ liệu mới về lại không gian dữ liệu đầu vào.
Đường phân lớp trong không gian dữ liệu đầu vào sẽ là đường cong (non-linear) và có thể phân lớp được các lớp dữ liệu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/8-svm/kernel_svm_3.png" style="width: 600px;"/>
