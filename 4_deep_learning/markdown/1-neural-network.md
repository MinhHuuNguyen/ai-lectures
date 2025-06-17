---
time: 08/30/2022
title: Mô hình mạng nơ ron đơn giản Neural network
description: Mô hình mạng nơ ron đơn giản Neural network là một mô hình tính toán lấy cảm hứng từ cấu trúc và hoạt động của bộ não con người. Mô hình mạng nơ ron đơn giản là nền tảng cho sự phát triển của các mô hình mạng nơ ron phức tạp hơn được sử dụng trong các mô hình Trí tuệ nhân tạo nổi tiếng hiện nay.
banner_url:
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Tổng quan

Mạng nơ ron nhân tạo (Artificial Neural Network – ANN) là một mô hình tính toán lấy cảm hứng từ cấu trúc và hoạt động của bộ não con người.
Mỗi mạng nơ ron gồm nhiều "nơ ron nhân tạo" (artificial neurons) được kết nối với nhau theo một trật tự nhất định, giúp mô hình học được các mối quan hệ phi tuyến tính giữa dữ liệu đầu vào và đầu ra.

<img src="" style="width: 1200px;"/>

Mô hình mạng nơ ron đơn giản là phiên bản phức tạp hơn của các mô hình machine learning như Linear Regression hay Logistic Regression nhưng nó cũng là nền tảng cho sự phát triển của các mô hình mạng nơ ron phức tạp hơn được sử dụng trong các mô hình Trí tuệ nhân tạo nổi tiếng hiện nay.

## 2. Mối quan hệ giữa hàm XOR và mạng nơ ron

Xét hàm XOR với đầu vào là 2 biến nhị phân $x_1$ và $x_2$ và đầu ra là 1 biến nhị phân $y$.

| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 0     | 0     | 0   |
| 0     | 1     | 1   |
| 1     | 0     | 1   |
| 1     | 1     | 0   |

Ta có thể phát biểu hàm XOR như sau: "Nếu $x_1$ và $x_2$ khác nhau thì $y$ bằng 1, nếu giống nhau thì $y$ bằng 0".

<img src="" style="width: 1200px;"/>

Hàm XOR là một ví dụ kinh điển cho thấy giới hạn của các mô hình tuyến tính, không thể dùng một mô hình tuyến tính để học và mô phỏng được hàm XOR.
Điều này là nền tảng cho sự ra đời của các mô hình mạng nơ ron phức tạp hơn.

## 3. Kiến trúc và các lớp trong mạng nơ ron

Các nơ ron trong neural network được kết nối với nhau tạo thành các lớp (layer).
Ta có lớp nhận đầu vào được gọi là input layer, các lớp tính toán ở giữa được gọi là hidden layer, lớp trả đầu ra được gọi là output layer.

<img src="" style="width: 1200px;"/>

Với kiến trúc này, neural network cho phép sử dụng rất nhiều các loại layer khác nhau, có các chức năng khác nhau.

Ngoài ra, với kiến trúc gồm các layer, neural network cho phép người dùng tăng / giảm kích thước của mô hình dễ dàng thông qua việc tăng / giảm số lượng layer và tăng / giảm kích thước của từng layer.

### 3.1. Linear layer

Linear layer là layer đơn giản nhất nhưng nền tảng để xây dựng neural network.
Cụ thể, linear layer thực hiện phép biến đổi tuyến tính (linear transformation) thông qua phép toán nhân ma trận.

$$ y = WX $$

trong đó:
- X là ma trận đầu vào của linear layer
- W là trọng số của neural network tại linear layer đó
- y là ma trận đầu ra của linear layer, kết quả của phép biến đổi tuyến tính

<img src="" style="width: 1200px;"/>

Bản thân Linear layer cũng đã là một mô hình machine learning, cụ thể là một mô hình hồi quy tuyến tính (linear regression).

### 3.2. Activation layer

Chúng ta đã cùng nhau nghiên cứu về một số các logistic activation layer như Sigmoid, Softmax, Tanh.
Bên cạnh các layer đó, còn rất nhiều các activation layer khác với các vai trò khác nhau.

#### 3.2.1. ReLU

Khác với Sigmoid hay Softmax thường dược đặt ở layer cuối cùng trong neural network, ReLU (Rectified-Linear Unit) là một activation layer thường được đặt giữa các linear layer nằm giữa của neural network.

$$ y = max(0, x) $$

trong đó:
- x là giá trị đầu vào của hàm ReLU
- y là giá trị đầu ra của hàm ReLU

Khi sử dụng hàm ReLU cho một vector hoặc ma trận, ta sử dụng hàm ReLU cho từng phần tử trên vector hay ma trận.

<img src="" style="width: 1200px;"/>

Các activation layer, đặc biệt là ReLU được đặt xen kẽ giữa các linear layer với vai trò giúp cho các linear layer có nghĩa.
Điều này đồng nghĩa với việc, nếu không có các activation layer đặt giữa các linear layer thì nhiều các linear layer đặt chồng lên nhau cũng không khác gì so với việc chỉ có một linear layer.

Ví dụ:

<img src="" style="width: 1200px;"/>

#### 3.2.2. Một số biến thể của ReLU

- **Leaky ReLU**: Giải quyết hiện tượng vanishing gradient tại những vị trí có giá trị nhỏ hơn hoặc bằng 0 của ReLU.

$$ y = max(\gamma x, x) $$

trong đó:
- x là giá trị đầu vào của hàm ReLU
- $\gamma$ là giá trị rất nhỏ, thường được lựa chọn là 0.1
- y là giá trị đầu ra của hàm ReLU

### 3.3. Normalization layer

Đối với các neural network các phức tạp và có kích thước mô hình lớn, quá trình huấn luyện càng khó khăn và bất ổn định.
Sự bất ổn định có thể dẫn đến việc neural network có kết quả huấn luyện rất kém.

Sự ra đời của các Normalization layer đã giúp cải thiện đáng kể tình trạng này.
Các normalization layer nói chung giúp chuẩn hoá đầu ra của mỗi layer trong neural network từ đó giúp ổn định hoá quá trình huấn luyện.

<img src="" style="width: 1200px;"/>

#### 3.3.1. Batch normalization layer

#### 3.3.2. Các normalization layer khác

- **Layer normalization layer**: 

- **Instance normalization layer**: 

- **Group normalization layer**: 

### 3.4. Dropout layer

Chúng ta đã bàn luận về hiện tượng overfit trong các mô hình machine learning, trong neural network, hiện tượng overfit cũng là một vấn đề nan giải.
Ta có Dropout layer là một lớp giúp phần nào đó giảm bớt hiện tượng overfit.

Cụ thể, tại mỗi vòng lặp trong suốt quá trình huấn luyện mô hình, dropout layer sẽ ngẫu nhiên lựa chọn một số các nơ ron trong neural network, chính xác hơn là lựa chọn ngẫu nhiên một số các trọng số của mô hình và gán giá trị bằng 0.
Từ đó, ta sẽ giảm được độ phức tạp của mô hình mạng neural network, từ đó giảm được hiện tượng overfit.

<img src="" style="width: 1200px;"/>

Một kỹ thuật hiện đại hơn được xây dựng với ý tưởng tương tự với Dropout layer là Regularization.

Bài viết giải thích chi tiết về Regularization, các bạn có thể xem ở [đây](/blog/hien-tuong-overfit-va-underfit)

## 4. Huấn luyện mạng nơ ron

Quá trình huấn luyện mạng nơ ron, bản chất, không quá khác biệt so với quá trình huấn luyện những mô hình như Logistic Regression, sử dụng thuật toán Gradient descent.

<img src="" style="width: 1200px;"/>

### 4.1. Đưa dữ liệu qua mô hình - Model feedforward

Feedforward là quá trình neural network nhận đầu vào input layer, thực hiện các phép tính toán qua các linear layer, activation layer, normalization layer ... để trả đầu ra output layer.

<img src="" style="width: 1200px;"/>

### 4.2. Tính toán giá trị loss sử dụng các Loss function

Hàm loss được sinh ra để đo đạc độ lệch (hay độ sai khác) giữa lời dự đoán của mô hình và giá trị thực tế đúng trong bộ dữ liệu.
Từ giá trị độ lệch này, mô hình sẽ được huấn luyện và tối ưu giúp độ chính xác của mô hình cải thiện dần theo thời gian.

<img src="" style="width: 1200px;"/>

Chúng ta đã nghiên cứu về khá loại hàm loss khác nhau, đối với hai bài toán cơ bản trong Machine Learning là Regression và Classification, ta có:
- Đối với bài toán Regression:
    - **Mean absolute error**:
    $$ MAE(\hat{y}, y) = \frac{1}{N}\sum_{i=1}^{N} |\hat{y}_i - y_i| $$
    - **Mean squared error**:
    $$ MSE(\hat{y}, y) = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2 $$
- Đối với bài toán Classification:
    - **Binary cross entropy**:
    $$BCE(\hat{y}, y) = - \sum_{i=1}^{N} (y^i \log \hat{y}^i + (1 - y^i) \log(1 - \hat{y}^i)) $$
    - **Categorical cross entropy** hay **Cross entropy**:
    $$ CE(\hat{y}, y) = - \sum_{i=1}^{N} \sum_{j=1}^{K} (y^{ij} \log \hat{y}^{ij})$$

### 4.3. Tính toán đạo hàm và cập nhật trọng số của mô hình - Model Backpropagation

Ngược lại với quá trình Model feedforward, sau khi đưa kết quả dự đoán của neural network vào hàm loss và tính toán giá trị loss, ta thực hiện quá trình Model backpropagation.

<img src="" style="width: 1200px;"/>

Quá trình Backpropagation là quá trình tính toán đạo hàm tương ứng với mỗi trọng số của mô hình nhằm phục vụ cho quá trình tối ưu mô hình bằng thuật toán Gradient descent.
Trong thực tế, ta sẽ không sử dụng thuật toán Gradient descent nguyên bản mà ta sẽ sử dụng các biến thể nâng cấp của Gradient descent nhằm cải thiện hiệu suất và giúp tối ưu mô hình đến giá trị tối ưu tốt hơn.
Bài viết giới thiệu cụ thể về Các biến thể nâng cấp của Gradient descent, các bạn có thể xem ở [đây](/blog/
