---
time: 10/01/2022
title: Các biến thể nâng cấp của thuật toán tối ưu Gradient descent
description:
banner_url:
tags: [machine-learning]
is_highlight: false
is_published: true
---
# Các biến thể và nâng cấp của thuật toán Gradient Descent

## 1. Nhắc lại về Gradient Descent

Cho hàm loss \( L \) với các trọng số \( \mathbf{w} \):

**Bước 1. Gradient Calculation (Tính toán gradient):**
- Tính gradient của hàm loss \( L \) theo các trọng số \( \mathbf{w} \).
- Gradient này cho biết hướng tăng lên nhanh nhất của hàm loss.

   \[ \nabla L(\mathbf{w}) = \left( \frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, ..., \frac{\partial L}{\partial w_n} \right) \]

**Bước 2. Parameter Update (Cập nhật trọng số):**
- Cập nhật các trọng số \( \mathbf{w} \) bằng cách điều chỉnh chúng theo hướng ngược lại với gradient, nhằm giảm thiểu hàm loss.
- Khoảng cách di chuyển theo gradient được điều chỉnh bởi một trọng số gọi là learning rate.

   \[ \mathbf{w_{t+1}} = \mathbf{w_{t}} - \alpha \cdot \nabla L(\mathbf{w_{t}}) \]

   Trong đó:
   - \( \mathbf{w} \) là vector các trọng số cần cập nhật.
   - \( \alpha \) là learning rate, là một số dương nhỏ thể hiện bước di chuyển trong mỗi vòng lặp. Giá trị learning rate quyết định độ lớn của bước cập nhật.

Gradient Descent là một phương pháp quan trọng để điều chỉnh trọng số của mô hình dựa trên đạo hàm của hàm mất mát.

Tuy nhiên, việc sử dụng chỉ Gradient Descent có thể gặp một số vấn đề, và đó là lý do tại sao chúng ta cần các thuật toán tối ưu hóa khác để cải thiện quá trình học.

## 2. Stochastic Gradient Descent (SGD) và Mini-batch Gradient descent

Từ thuật toán Gradient Descent, chúng ta có thể nâng cấp lên hai biến thể khác là Stochastic Gradient Descent (SGD) và Mini-batch Gradient Descent.

Cả hai biến thể này được thiết kế để tối ưu hóa quá trình cập nhật tham số trong quá trình huấn luyện mô hình.

### 2.1. Stochastic Gradient Descent (SGD)

Trong SGD, thay vì tính gradient dựa trên toàn bộ dữ liệu huấn luyện, chúng ta chỉ chọn một điểm dữ liệu ngẫu nhiên từ tập huấn luyện để tính gradient.

SGD thường có tốc độ hội tụ nhanh hơn so với Gradient Descent vì mỗi vòng lặp chỉ tính toán và cập nhật một phần nhỏ của dữ liệu.

**Parameter Update với Stochastic Gradient Descent:**

\[ \mathbf{w_{t+1}} = \mathbf{w_{t}} - \alpha \cdot \nabla L(\mathbf{w_{t}}; x_i, y_i) \]

Trong đó:
- \( \mathbf{w} \) là vector các tham số cần cập nhật.
- \( \alpha \) là learning rate.
- \( \nabla L(\mathbf{w}; x_i, y_i) \) là gradient của hàm loss \( L \) tính từ điểm dữ liệu \( x_i \) và giá trị thực tế \( y_i \).

Bằng cách áp dụng gradient tính từ một điểm dữ liệu ngẫu nhiên, SGD thường hội tụ nhanh hơn so với Gradient Descent, nhưng cũng có thể tạo ra sự dao động trong quá trình hội tụ.

Điều này có thể được kiểm soát bằng cách điều chỉnh learning rate hoặc sử dụng các phương pháp tối ưu hóa khác.

### 2.2. Mini-batch Gradient descent

Mini-batch Gradient Descent là sự kết hợp giữa Gradient Descent và SGD.
Nó có thể tận dụng hiệu suất tính toán của các thư viện và đồng thời giảm tối đa sự không ổn định của SGD.

Trong biến thể này, chúng ta chia dữ liệu huấn luyện thành các mini-batch nhỏ.
Mỗi vòng lặp, chúng ta tính gradient dựa trên một mini-batch và cập nhật tham số.

**Parameter Update với Mini-batch Gradient Descent:**

\[ \mathbf{w_{t+1}} = \mathbf{w_{t}} - \alpha \cdot \nabla L(\mathbf{w_{t}}; X_{\text{batch}}, y_{\text{batch}}) \]

Trong đó:
- \( \mathbf{w} \) là vector các tham số cần cập nhật.
- \( \alpha \) là learning rate.
- \( \nabla L(\mathbf{w}; X_{\text{batch}}, y_{\text{batch}}) \) là gradient của hàm loss \( L \) tính từ một mini-batch \( X_{\text{batch}} \) các điểm dữ liệu và tương ứng \( y_{\text{batch}} \) các giá trị thực tế.

Kích thước của mini-batch là một tham số quan trọng, và lựa chọn này có thể ảnh hưởng đến tốc độ hội tụ và tính ổn định của thuật toán.

## 3. Momentum Gradient Descent

Momentum Gradient Descent thêm một yếu tố "momentum" để lưu trữ thông tin về các hướng di chuyển trước đó của các bước cập nhật. Điều này giúp thuật toán "nhớ" hướng di chuyển trước đó và giúp cho quá trình cập nhật tham số mượt mà hơn.

**Parameter Update với Momentum:**

\[ \mathbf{v_{t}} = \beta \cdot \mathbf{v_{t-1}} + (1 - \beta) \cdot \nabla L(\mathbf{w_{t}}) \]
\[ \mathbf{w_{t+1}} = \mathbf{w_{t}} - \alpha \cdot \mathbf{v_{t}} \]

Trong đó:
- \( \mathbf{v} \) là vector "momentum", lưu trữ thông tin về các hướng di chuyển trước đó.
- \( \beta \) là hệ số momentum, thường được đặt trong khoảng từ 0.8 đến 0.99.
- \( \alpha \) là learning rate.

Momentum Gradient Descent giúp làm giảm bớt sự dao động trong quá trình cập nhật tham số, đồng thời tăng tốc độ hội tụ. Nó đặc biệt hiệu quả khi đối mặt với các bề mặt hàm lồi gần như ngang hoặc khi có sự biến đổi lớn trong các gradient cục bộ.

Lựa chọn hệ số momentum và learning rate là quan trọng để đạt được hiệu suất tốt nhất cho Momentum Gradient Descent.


## 4. Nesterov Accelerated Gradient (NAG)

NAG cải tiến Momentum Gradient Descent bằng cách tính gradient tại vị trí ước tính tham số tiếp theo, trước khi thực sự cập nhật tham số bằng công thức của Momentum. Điều này giúp tạo ra một dự đoán tốt hơn về hướng di chuyển tiếp theo và làm giảm sự dao động.

**Parameter Update với NAG:**

\[ \mathbf{v_{t}} = \beta \cdot \mathbf{v_{t-1}} + (1 - \beta) \cdot \nabla L(\mathbf{w_{t}} - \beta \cdot \mathbf{v_{t-1}}) \]
\[ \mathbf{w_{t+1}} = \mathbf{w_{t}} - \alpha \cdot \mathbf{v_{t}} \]

Trong đó:
- \( \mathbf{v} \) là vector "momentum" được tính dựa trên dự đoán gradient ở vị trí tiếp theo.
- \( \beta \) là hệ số momentum, thường được đặt trong khoảng từ 0.8 đến 0.99.
- \( \alpha \) là learning rate.

NAG giúp tăng tốc quá trình hội tụ và giảm sự dao động so với Momentum Gradient Descent. 

Lựa chọn hệ số momentum và learning rate vẫn quan trọng để đạt được hiệu suất tốt cho Nesterov Accelerated Gradient.

## 5. Adaptive Gradient Algorithm (AdaGrad)

Trong các biến thể cơ bản của Gradient Descent, việc sử dụng cùng một learning rate cho tất cả các tham số có thể gây ra vấn đề về tốc độ hội tụ và ổn định.

AdaGrad là một biến thể của thuật toán tối ưu hóa Gradient Descent, được cải tiến để tự động điều chỉnh learning rate cho từng tham số dựa trên tần suất xuất hiện của gradient của tham số đó trong quá trình tối ưu hóa.

Ý tưởng chính của AdaGrad là giảm learning rate cho các tham số có gradient lớn và tăng learning rate cho các tham số có gradient nhỏ. Điều này giúp cải thiện hiệu suất tối ưu hóa trên các tham số có sự biến đổi khác nhau.

**Parameter Update với AdaGrad:**

\[ \mathbf{G_{t}} = \mathbf{G_{t-1}} + (\nabla L(\mathbf{w_{t}}))^2 \]
\[ \mathbf{w_{t+1}} = \mathbf{w_{t}} - \frac{\alpha}{\sqrt{\mathbf{G_{t}} + \epsilon}} \cdot \nabla L(\mathbf{w_{t}}) \]

Trong đó:
- \( \mathbf{G} \) là ma trận đường chéo lưu trữ tổng bình phương các gradient tính từ trước.
- \( \nabla L(\mathbf{w}) \) là gradient của hàm chi phí \( L \).
- \( \alpha \) là learning rate,.
- \( \epsilon \) là một số nhỏ được thêm vào để tránh chia cho 0.

AdaGrad tự động điều chỉnh learning rate dựa trên lịch sử gradient, giúp tối ưu hóa hiệu quả hơn trên các tham số có biến đổi khác biệt.
Tuy nhiên, cần lưu ý về vấn đề giảm learning rate quá mức sau một thời gian dài.

## 6. Root Mean Square Propagation (RMSProp)

Trong một số trường hợp, AdaGrad có thể dẫn đến việc giảm learning rate quá mức sau một thời gian, khiến quá trình hội tụ chậm lại.
RMSProp giải quyết vấn đề này bằng cách thay đổi cách tính tổng bình phương gradient trong công thức cập nhật.

Thay vì tích luỹ tổng bình phương gradient qua tất cả các vòng lặp như AdaGrad, RMSProp sử dụng một hệ số giảm dần để giảm dần độ quan trọng của gradient cũ trong quá trình tính toán.
Điều này giúp làm giảm tốc độ giảm learning rate sau thời gian dài, tạo ra một tốc độ hội tụ ổn định hơn.

**Parameter Update với RMSProp:**

\[ \mathbf{E}[\nabla L(\mathbf{w_{t}})^2] = \beta \cdot \mathbf{E}[\nabla L(\mathbf{w_{t-1}})^2] + (1 - \beta) \cdot (\nabla L(\mathbf{w_{t}}))^2 \]
\[ \mathbf{w_{t+1}} = \mathbf{w_{t}} - \frac{\alpha}{\sqrt{\mathbf{E}[\nabla L(\mathbf{w_{t}})^2] + \epsilon}} \cdot \nabla L(\mathbf{w_{t}}) \]

Trong đó:
- \( \mathbf{E}[\nabla L(\mathbf{w})^2] \) là giá trị kỳ vọng của bình phương gradient, được tính dựa trên một hệ số giảm dần \( \beta \).
- \( \nabla L(\mathbf{w}) \) là gradient của hàm chi phí \( L \).
- \( \alpha \) là learning rate.
- \( \epsilon \) là một số nhỏ để tránh chia cho 0.

RMSProp giúp kiểm soát sự biến đổi của gradient và learning rate, giúp tối ưu hóa ổn định hơn trong quá trình huấn luyện.
Điều này đặc biệt hữu ích khi tối ưu hóa các mô hình phức tạp và khi gradient có biến đổi lớn.

## 7. Adaptive Moment Estimation (Adam)

Adam là một biến thể của thuật toán tối ưu hóa Gradient Descent, được thiết kế để kết hợp cả Momentum và RMSProp.
Đây là một thuật toán phổ biến và mạnh mẽ được sử dụng trong học máy để tối ưu hóa các mô hình.

Adam tận dụng thông tin từ gradient và moment của các bước cập nhật trước đó để điều chỉnh learning rate cho từng tham số.
Ý tưởng chính là kết hợp cả yếu tố "momentum" (theo dõi hướng di chuyển trước) và yếu tố "adaptive learning rate" (tự động điều chỉnh learning rate) để cải thiện tốc độ hội tụ và ổn định của quá trình tối ưu hóa.

**Parameter Update với Adam:**

\[ \mathbf{v_{t}} = \beta_1 \cdot \mathbf{v_{t-1}} + (1 - \beta_1) \cdot \nabla L(\mathbf{w_{t}}) \]
\[ \mathbf{E_{t}} = \beta_2 \cdot \mathbf{E_{t-1}} + (1 - \beta_2) \cdot (\nabla L(\mathbf{w_{t}}))^2 \]
\[ \hat{\mathbf{v}}_{t} = \frac{\mathbf{v}_{t}}{1 - \beta_1^t} \]
\[ \hat{\mathbf{E}}_{t} = \frac{\mathbf{E}_{t}}{1 - \beta_2^t} \]
\[ \mathbf{w_{t+1}} = \mathbf{w_{t}} - \frac{\alpha}{\sqrt{\hat{\mathbf{E}}_{t} + \epsilon}} \cdot \hat{\mathbf{v}}_{t}\]

Trong đó:
- \( \mathbf{v} \) là vector moment, lưu trữ thông tin về hướng di chuyển trước.
- \( \mathbf{E} \) là vector ước tính của bình phương gradient, giúp điều chỉnh learning rate dựa trên biến đổi của gradient.
- \( \beta_1 \) và \( \beta_2 \) là các hệ số moment, thường trong khoảng từ 0.8 đến 0.999.
- \( \alpha \) là learning rate, tỷ lệ học.
- \( \epsilon \) là một số nhỏ để tránh chia cho 0.
- \( t \) là số vòng lặp.

**Moment (Momentum) Component:**
Adam giữ lại phần tử "momentum" từ thuật toán Gradient Descent với Moment, trong đó vector \( \mathbf{v} \) lưu trữ thông tin về hướng di chuyển trước đó. Điều này giúp Adam "nhớ" sự di chuyển trước đó và giảm sự dao động trong quá trình cập nhật.

**Adaptive Learning Rate Component:**
Tương tự như RMSProp, Adam cũng sử dụng một vector \( \mathbf{E} \) để theo dõi biến đổi của gradient. Tuy nhiên, thay vì sử dụng tổng bình phương gradient như trong RMSProp, Adam sử dụng giá trị kỳ vọng của bình phương gradient.

**Bias Correction:**
Adam thêm bước điều chỉnh để loại bỏ sự bias của \( \mathbf{v} \) và \( \mathbf{E} \) tại các vòng lặp ban đầu. Điều này được thực hiện bằng cách chia \( \mathbf{v} \) và \( \mathbf{E} \) cho một lũy thừa của hệ số \( \beta_1 \) và \( \beta_2 \) tương ứng. Quá trình này đảm bảo rằng trong giai đoạn đầu của huấn luyện, khi \( t \) còn nhỏ, các giá trị này không bị ảnh hưởng bởi việc khởi tạo ban đầu.

Kết hợp cả hai yếu tố "momentum" và "adaptive learning rate" giúp Adam tự động điều chỉnh learning rate cho từng tham số dựa trên lịch sử gradient và moment. Điều này giúp tối ưu hóa hiệu quả hơn trong các ngữ cảnh phức tạp, giảm tình trạng hội tụ chậm và sự dao động.

## 8. Một số biến thể khác của Gradient descent

### 8.1. Nadam (Nesterov-accelerated Adaptive Moment Estimation):

Kết hợp cả Nesterov Accelerated Gradient và Adam. Nadam kết hợp yếu tố moment và adaptive learning rate để cải thiện tốc độ hội tụ.

### 8.2. Adadelta:

Tương tự như RMSProp, nhưng sử dụng tỷ lệ chia của các bước cập nhật trước đó để điều chỉnh learning rate.

### 8.3. AMSGrad (Adaptive Moment Estimation for AMSGrad):

Một biến thể của Adam nhằm khắc phục vấn đề của RMSProp và Adam trong việc ổn định learning rate.

### 8.4. RAdam (Rectified Adam):

Một biến thể của Adam với cách cập nhật tốt hơn cho trạng thái không ổn định trong quá trình đầu tối ưu hóa.

### 8.5. FTRL-Proximal (Follow The Regularized Leader):

Sử dụng kỹ thuật Follow The Regularized Leader để tối ưu hóa, thường được sử dụng trong các bài toán tối ưu hóa lớn với dữ liệu thưa.

### 8.6. Yogi:

Một biến thể của Adam nhằm cải thiện tốc độ hội tụ trong trường hợp có nhiễu trong dữ liệu.

### 8.7. Lookahead:

Kết hợp cả Adam và Nesterov Accelerated Gradient để cải thiện tốc độ hội tụ và khắc phục vấn đề của Adam khi gặp các vùng hẹp.
