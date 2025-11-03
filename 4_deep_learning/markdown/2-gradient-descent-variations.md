---
time: 04/05/2022
title: Các biến thể nâng cấp của thuật toán tối ưu Gradient descent
description: Gradient Descent là một thuật toán tối ưu hóa quan trọng trong Machine Learning, nhưng có thể gặp một số vấn đề trong quá trình hội tụ. Trong bài viết này, chúng ta sẽ tìm hiểu về các biến thể nâng cấp của Gradient Descent như Stochastic Gradient Descent (SGD), Mini-batch Gradient Descent, Momentum Gradient Descent, Nesterov Accelerated Gradient (NAG), và các thuật toán tối ưu hóa khác như AdaGrad, RMSProp, Adam, và nhiều biến thể khác.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/3-gradient-descent/banner.jpeg
tags: [deep-learning]
is_highlight: false
is_published: true
---

## 1. Nhắc lại về Gradient Descent

Bài viết giới thiệu cụ thể về thuật toán tối ưu Gradient descent nguyên bản, các bạn có thể xem ở [đây](/blog/thuat-toan-toi-uu-gradient-descent).

Công thức cập nhật trọng số của mô hình trong thuật toán Gradient Descent nguyên bản là:

$$ w^{t+1} = w^t - \eta \cdot L'(w^t) $$

Cụ thể hơn, thuật toán Gradient Descent nguyên bản cần tính toán đạo hàm của hàm mất mát $L$ tại điểm $w^t$ trên toàn bộ bộ dữ liệu huấn luyện.

Do đó, ta có thể viết lại như sau:

$$ w^{t+1} = w^t - \eta \cdot L'(w^t, X, y) $$

Trong bài viết này, chúng ta sẽ tìm hiểu về các biến thể nâng cấp của Gradient Descent để giải quyết một số vấn đề trong quá trình hội tụ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/2-gradient-descent-variations/problems.jpeg" style="width: 600px;"/>

Các biến thể nâng cấp này giúp giải quyết ba vấn đề chính của thuật toán Gradient Descent nguyên bản:
- **Vấn đề 1:** Gradient descent phải tính toán với tất cả các phần tử trong bộ dữ liệu cho mỗi lần cập nhật trọng số của mô hình.
- **Vấn đề 2:** Gradient descent phụ thuộc vào việc khởi tạo giá trị trọng số ban đầu.
- **Vấn đề 3:** Gradient descent phụ thuộc vào việc lựa chọn learning rate.

## 2. Vấn đề: Gradient descent phải tính toán với tất cả các phần tử trong bộ dữ liệu cho mỗi lần cập nhật trọng số của mô hình

Từ thuật toán Gradient Descent, chúng ta có thể nâng cấp lên hai biến thể khác là Stochastic Gradient Descent (SGD) và Mini-batch Gradient Descent giúp giải quyết vấn đề này.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/2-gradient-descent-variations/gd_sgd_mini_batch_gd.jpeg" style="width: 600px;"/>

Trong hai biến thể này, chúng ta sẽ không tính toán gradient dựa trên toàn bộ dữ liệu huấn luyện, từ đó, giảm thiểu thời gian tính toán và tăng tốc độ hội tụ của thuật toán.

### 2.1. Stochastic Gradient Descent (SGD)

Trong Stochastic Gradient Descent (SGD), thay vì tính gradient dựa trên toàn bộ dữ liệu huấn luyện, chúng ta chỉ chọn một điểm dữ liệu ngẫu nhiên từ tập dữ liệu huấn luyện để tính gradient.

SGD thường có tốc độ hội tụ nhanh hơn so với Gradient Descent vì mỗi vòng lặp chỉ tính toán và cập nhật trọng số với một điểm dữ liệu của bộ dữ liệu.

Công thức cập nhật trọng số của mô hình trong thuật toán Stochastic Gradient Descent là:

$$ w^{t+1} = w^t - \eta \cdot L'(w^t, x_i, y_i) $$

Ở đây, thay vì tính đạo hàm trên toàn bộ bộ dữ liệu $L'(w^t, X, y)$, chúng ta chỉ tính đạo hàm tại một điểm dữ liệu ngẫu nhiên $(x_i, y_i)$ trong bộ dữ liệu huấn luyện $L'(w^t, x_i, y_i)$.

Bằng cách áp dụng gradient tính từ một điểm dữ liệu ngẫu nhiên, SGD thường hội tụ nhanh hơn so với Gradient Descent, nhưng cũng có thể tạo ra sự dao động trong quá trình hội tụ.
Điều này xảy ra do trong bộ dữ liệu có thể chứa những điểm dữ liệu ngoại lai (outlier) hoặc những điểm dữ liệu nhiễu (noise) và nó có thể khiến cho quá trình huấn luyện mô hình trở nên không ổn định.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/2-gradient-descent-variations/gd_sgd_mini_batch_gd.gif" style="width: 600px;"/>

### 2.2. Mini-batch Gradient descent

Mini-batch Gradient Descent là thuật toán tối ưu nằm ở giữa của Gradient Descent nguyên bản và Stochastic Gradient Descent (SGD).
Mini-batch Gradient Descent:
- Không yêu cầu tính toán đạo hàm trên cả bộ dữ liệu như Gradient Descent nguyên bản.
Từ đó, tăng tốc thời gian hội tụ của quá trình huấn luyện mô hình.
- Không tính toán đạo hàm trên duy nhất một điểm dữ liệu riêng lẻ như SGD.
Từ đó, độ ổn định của quá trình huấn luyện mô hình được cải thiện.

Trong Mini-batch Gradient Descent, chúng ta chia dữ liệu huấn luyện thành các mini-batch nhỏ gồm một số lượng điểm dữ liệu nhất định như 32, 64, 128, v.v.
Mỗi vòng lặp, chúng ta tính gradient dựa trên một mini-batch và cập nhật trọng số của mô hình.

Công thức cập nhật trọng số của mô hình trong thuật toán Mini-batch Gradient Descent là:

$$ w^{t+1} = w^t - \eta \cdot L'(w^t, X_{batch}, y_{batch}) $$

Ở đây, thay vì tính đạo hàm trên toàn bộ bộ dữ liệu $L'(w^t, X, y)$ hay $L'(w^t, x_i, y_i)$, chúng ta tính đạo hàm với một batch các điểm dữ liệu ngẫu nhiên $(X_{batch}, y_{batch})$ trong bộ dữ liệu huấn luyện $L'(w^t, X_{batch}, y_{batch})$.

Kích thước của mini-batch là một siêu tham số (hyper-parameter) quan trọng, và lựa chọn này có thể ảnh hưởng đến tốc độ hội tụ và tính ổn định của thuật toán.
Về lý thuyết, kích thước mini-batch càng lớn thì quá trình hội tụ càng ổn định, nhưng khối lượng tính toán sẽ tăng lên.

## 3. Vấn đề: Gradient descent phụ thuộc vào việc khởi tạo giá trị trọng số ban đầu

### 3.1. Momentum Gradient Descent

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/2-gradient-descent-variations/momentum.jpeg" style="width: 600px;"/>

Momentum Gradient Descent thêm một yếu tố "momentum" để lưu trữ thông tin về hướng và độ lớn của các bước di chuyển trước đó của các bước cập nhật.
Điều này giúp thuật toán "nhớ" hướng di chuyển trước đó và giúp cho quá trình cập nhật trọng số của mô hình mượt mà hơn.

Công thức cập nhật trọng số của mô hình trong thuật toán Momentum Gradient Descent là:

$$ w^{t+1} = w^t - \eta \cdot v^t $$

trong đó, vector "momentum" $v^t$ được cập nhật dựa vào $v^{t-1}$ theo công thức:

$$ v^{t} = \beta \cdot v^{t-1} + (1 - \beta) \cdot L'(w^t) $$

Trong công thức trên, $\beta$ là hệ số momentum, thường được đặt trong khoảng từ 0.8 đến 0.99.

Momentum Gradient Descent giúp làm giảm bớt sự dao động trong quá trình cập nhật trọng số, đồng thời tăng tốc độ hội tụ. Nó đặc biệt hiệu quả khi đối mặt với các bề mặt hàm lồi gần như ngang hoặc khi có sự biến đổi lớn trong các gradient cục bộ.

Lựa chọn hệ số momentum và learning rate là quan trọng để đạt được hiệu suất tốt nhất cho Momentum Gradient Descent.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/2-gradient-descent-variations/gd_momentum_gd_nag.gif" style="width: 600px;"/>

### 3.2. Nesterov Accelerated Gradient (NAG)

Nesterov Accelerated Gradient (NAG) cải tiến Momentum Gradient Descent bằng cách tính gradient tại vị trí **ước tính** trọng số tiếp theo, trước khi **thực sự** cập nhật trọng số của mô hình bằng công thức của Momentum.
Điều này giúp tạo ra một dự đoán tốt hơn về hướng di chuyển tiếp theo và làm giảm sự dao động.

Công thức cập nhật trọng số của mô hình trong thuật toán Nesterov Accelerated Gradient là:

$$ w^{t+1} = w^t - \eta \cdot v^t $$

Để tính được $v^t$ trong công thức trên, ta cần **ước tính** trọng số tiếp theo bằng:

$$ w^{t+1}_{\text{estimated}} = w^t - \beta \cdot v^{t-1} $$

Sau đó, vector "momentum" $v^t$ của NAG được cập nhật theo công thức:

$$ v^{t} = \beta \cdot v^{t-1} + (1 - \beta) \cdot L'(w^{t+1}_{\text{estimated}}) $$

Trong công thức trên, $\beta$ là hệ số momentum, thường được đặt trong khoảng từ 0.8 đến 0.99.

NAG giúp tăng tốc quá trình hội tụ và giảm sự dao động so với Momentum Gradient Descent. 
Lựa chọn hệ số momentum và learning rate vẫn quan trọng để đạt được hiệu suất tốt cho Nesterov Accelerated Gradient.

## 4. Vấn đề: Gradient descent phụ thuộc vào việc lựa chọn learning rate

Trong các biến thể cơ bản của Gradient Descent, ta sử dụng cùng một learning rate cho tất cả các trọng số của mô hình và trong thực tế, điều này gây ra vấn đề về tốc độ hội tụ và sự ổn định trong quá trình hội tụ.

Nói cách khác, mỗi trọng số của mô hình có vai trò khác nhau trong quá trình mô hình tính toán ra giá trị dự đoán.
Do đó, việc sử dụng cùng một learning rate cho tất cả các trọng số là không hợp lý.

Để giải quyết vấn đề này, chúng ta có thể sử dụng các thuật toán tối ưu **adaptive learning rate** để mỗi trọng số của mô hình có thể có một learning rate khác nhau.

### 4.1. Adaptive Gradient Algorithm (AdaGrad)

Adaptive Gradient Algorithm (AdaGrad) là một biến thể Gradient Descent, được cải tiến để tự động điều chỉnh learning rate cho từng trọng số dựa trên tần suất xuất hiện của gradient của trọng số đó trong quá trình tối ưu hóa.

Ý tưởng chính của AdaGrad là giảm learning rate cho các trọng số có gradient lớn và tăng learning rate cho các trọng số có gradient nhỏ.
Điều này giúp cải thiện hiệu suất tối ưu hóa trên các trọng số có sự biến đổi khác nhau.

Trong AdaGrad, chúng ta lưu trữ tổng bình phương của gradient tính từ trước trong một ma trận đường chéo $G$.

$$ G_{t} = G_{t-1} + (L'(w_{t}))^2 $$

Công thức cập nhật trọng số của mô hình trong thuật toán Adaptive Gradient Algorithm là:

$$ w_{t+1} = w_{t} - \frac{\eta}{\sqrt{G_{t}} + \epsilon} \cdot L'(w_{t}) $$

trong công thức trên, giá trị learning rate thật sự được sử dụng cho mỗi trọng số được tính toán lại dựa trên giá trị learning rate khởi tạo ban đầu theo ma trận đường chéo $G$.

AdaGrad tự động điều chỉnh learning rate dựa trên lịch sử gradient của từng trọng số, giúp tối ưu hóa hiệu quả hơn trên các trọng có biến đổi khác biệt.
Điều này cũng giúp việc lựa chọn learning rate trở nên dễ dàng hơn.

Tuy nhiên, một vấn đề của AdaGrad là sau một thời gian dài, learning rate có thể giảm quá mức do việc tích luỹ tổng bình phương gradient, dẫn đến quá trình hội tụ chậm lại.
Điều này có thể xảy ra khi các trọng số không còn thay đổi nhiều nữa, và learning rate trở nên quá nhỏ để tiếp tục cập nhật trọng số.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/2-gradient-descent-variations/gd_ada_grad_rms_prop.jpeg" style="width: 600px;"/>

### 4.2. Root Mean Square Propagation (RMSProp)

Trong một số trường hợp, AdaGrad có thể dẫn đến việc giảm learning rate quá mức sau một thời gian, khiến quá trình hội tụ chậm lại.
RMSProp giải quyết vấn đề này bằng cách thay đổi cách tính tổng bình phương gradient trong công thức cập nhật.

Thay vì tích luỹ tổng bình phương gradient qua tất cả các vòng lặp như AdaGrad, RMSProp sử dụng một hệ số giảm dần để giảm dần độ quan trọng của gradient cũ trong quá trình tính toán.
Điều này giúp làm giảm tốc độ giảm learning rate sau thời gian dài, tạo ra một tốc độ hội tụ ổn định hơn.

$$ E[L'(w_{t})^2] = \beta \cdot E[L'(w_{t-1})^2] + (1 - \beta) \cdot (L'(w_{t}))^2 $$

Công thức cập nhật trọng số của mô hình trong thuật toán Root Mean Square Propagation là:

$$ w_{t+1} = w_{t} - \frac{\eta}{\sqrt{E[L'(w_{t})^2]} + \epsilon} \cdot L'(w_{t}) $$

RMSProp giúp kiểm soát sự biến đổi của gradient và learning rate, giúp tối ưu hóa ổn định hơn trong quá trình huấn luyện.
Điều này đặc biệt hữu ích khi tối ưu hóa các mô hình phức tạp và khi gradient có biến đổi lớn.

## 5. Adaptive Moment Estimation (Adam)

Adam là một biến thể rất phổ biến và mạnh mẽ của thuật toán tối ưu hóa Gradient Descent, được thiết kế để kết hợp những điểm mạnh của cả Momentum và RMSProp.
Đây là một thuật toán phổ biến và mạnh mẽ được sử dụng trong học máy để tối ưu hóa các mô hình.

Adam tận dụng thông tin từ gradient và moment của các bước cập nhật trước đó để điều chỉnh learning rate cho từng tham số.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/2-gradient-descent-variations/adam.jpeg" style="width: 600px;"/>

Ý tưởng chính là kết hợp cả yếu tố "momentum" (theo dõi hướng di chuyển trước) và yếu tố "adaptive learning rate" (tự động điều chỉnh learning rate) để cải thiện tốc độ hội tụ và ổn định của quá trình tối ưu hóa.

### 5.1. Thành phần "momentum"

Adam giữ lại phần tử "momentum" từ thuật toán Gradient Descent với Moment, trong đó vector $v$ lưu trữ thông tin về hướng di chuyển trước đó.
Điều này giúp Adam "nhớ" sự di chuyển trước đó và giảm sự dao động trong quá trình cập nhật.

$$ v_{t} = \beta_1 \cdot v_{t-1} + (1 - \beta_1) \cdot L'(w_{t}) $$

### 5.2. Thành phần "adaptive learning rate"

Tương tự như RMSProp, Adam cũng sử dụng một vector $E_{t}$ để theo dõi biến đổi của gradient.

$$ E_{t} = \beta_2 \cdot E_{t-1} + (1 - \beta_2) \cdot (L'(w_{t}))^2 $$

### 5.3. Thành phần "bias correction"

Adam thêm bước điều chỉnh để loại bỏ sự bias của $v$ và $E$ tại các vòng lặp ban đầu.

Điều này được thực hiện bằng cách chia $v$ và $E$ cho một lũy thừa của hệ số $\beta_1$ và $\beta_2$ tương ứng. Quá trình này đảm bảo rằng trong giai đoạn đầu của huấn luyện, các giá trị này không bị ảnh hưởng bởi việc khởi tạo ban đầu.

$$ \hat{v}_{t} = \frac{v_{t}}{1 - \beta_1^t} $$

$$ \hat{E}_{t} = \frac{E_{t}}{1 - \beta_2^t} $$

### 5.4. Công thức cập nhật trọng số

Công thức cập nhật trọng số của mô hình trong thuật toán Adam là:

$$ w_{t+1} = w_{t} - \frac{\eta}{\sqrt{\hat{E}_{t}} + \epsilon} \cdot \hat{v}_{t} $$

Ta thường sử dụng các giá trị $\beta_1 = 0.9$ và $\beta_2 = 0.999$ cho các tham số của Adam, cùng với một learning rate khởi tạo thường là $0.001$.

Kết hợp cả hai yếu tố "momentum" và "adaptive learning rate" giúp Adam tự động điều chỉnh learning rate cho từng tham số dựa trên lịch sử gradient và moment.

Điều này giúp tối ưu hóa hiệu quả hơn trong các ngữ cảnh phức tạp, giảm tình trạng hội tụ chậm và sự dao động.

Một rule-of-thumb trong việc huấn luyện mô hình deep learning, **Nếu bạn không chắc chắn về thuật toán tối ưu nào nên sử dụng, hãy bắt đầu với Adam.**

## 6. Một số biến thể khác của Gradient descent

- **Nadam (Nesterov-accelerated Adaptive Moment Estimation):** Kết hợp cả Nesterov Accelerated Gradient và Adam.
- **Adadelta:** Một biến thể của AdaGrad, không cần lưu trữ tổng bình phương gradient, mà sử dụng một cửa sổ trượt để tính toán.
- **RAdam (Rectified Adam):** Một biến thể của Adam với cách cập nhật tốt hơn cho trạng thái không ổn định trong quá trình đầu tối ưu hóa.
