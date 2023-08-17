---
layout: "post"
title:  "Neural network"
author: "Nguyễn Hữu Minh"
permalink: "/deep-learning/neural-network"
parent: "Deep learning"
nav_order: 1
---

# Deep learning và neural network

## 1. Kiến trúc mạng nơ ron - Neural network

Được lấy cảm hứng từ mạng nơ ron sinh học trong bộ não của con người, neural network là kiến trúc mạng gồm rất nhiều các nơ ron kết nối với nhau theo một trật tự nhất định.

<img src="https://askabiologist.asu.edu/sites/default/files/resources/articles/neuron_anatomy.jpg" style="width: 600px;"/>


Các nơ ron trong neural network được kết nối với nhau tạo thành các lớp (layer). Ta có lớp nhận đầu vào được gọi là input layer, các lớp tính toán ở giữa được gọi là hidden layer, lớp trả đầu ra được gọi là output layer.

<img src="https://tikz.net/wp-content/uploads/2021/12/neural_networks-001.png" style="width: 600px;"/>

Trong neural network có rất nhiều các loại layer khác nhau, có các chức năng khác nhau.
Trong nội dung bài này, chúng ta sẽ nghiên cứu về một số các loại layer cơ bản nhất.

### 1.1. Linear layer

Linear layer là layer đơn giản nhất nhưng nền tảng để xây dựng neural network.
Cụ thể, linear layer thực hiện phép biến đổi tuyến tính (linear transformation) thông qua phép toán nhân ma trận.

$$
y = WX
$$

trong đó:
- X là ma trận đầu vào của linear layer
- W là trọng số của neural network tại linear layer đó
- y là ma trận đầu ra của linear layer, kết quả của phép biến đổi tuyến tính

### 1.2. Activation layer

Chúng ta đã cùng nhau nghiên cứu về một số các logistic activation layer như Sigmoid, Softmax, Tanh.
Bên cạnh các layer đó, còn rất nhiều các activation layer khác.
Các activation layer được đặt xen kẽ giữa các linear layer với vai trò giúp cho các linear layer có nghĩa.
Điều này đồng nghĩa với việc, nếu không có các activation layer đặt giữa các linear layer thì nhiều các linear layer đặt chồng lên nhau cũng không khác gì so với việc chỉ có một linear layer.

#### 1.2.1. ReLU

Khác với Sigmoid hay Softmax thường dược đặt ở layer cuối cùng trong neural network, ReLU (Rectified-Linear Unit) là một activation layer thường được đặt giữa các linear layer nằm giữa của neural network.

$$
y = max(0, x)
$$

trong đó:
- x là giá trị đầu vào của hàm ReLU
- y là giá trị đầu ra của hàm ReLU

Khi sử dụng hàm ReLU cho một vector hoặc ma trận, ta sử dụng hàm ReLU cho từng phần tử trên vector hay ma trận.

<img src="https://www.researchgate.net/profile/Yingzhi-Zhang-4/publication/351263278/figure/fig6/AS:1018784072601600@1619908459755/ReLU-operation-on-single-depth-slice.ppm" style="width: 600px;"/>

#### 1.2.2. Leaky ReLU

Một vấn đề khi sử dụng hàm ReLU là vanishing gradient tại những vị trí có giá trị nhỏ hơn hoặc bằng 0.
Do đó, Leaky ReLU giúp cải thiện vấn đề này.

$$
y = max(\gamma x, x)
$$

trong đó:
- x là giá trị đầu vào của hàm ReLU
- $\gamma$ là giá trị rất nhỏ, thường được lựa chọn là 0.1
- y là giá trị đầu ra của hàm ReLU

<img src="https://miro.medium.com/v2/resize:fit:1400/1*ypsvQH7kvtI2BhzR2eT_Sw.png" style="width: 600px;"/>

### 1.3. Normalization layer

Đối với các neural network các phức tạp và có kích thước mô hình lớn, quá trình huấn luyện càng khó khăn và bất ổn định.
Sự bất ổn định có thể dẫn đến việc neural network có kết quả huấn luyện rất kém.

Sự ra đời của các Normalization layer, và cụ thể là Batch normalization layer đã giúp cải thiện đáng kể tình trạng này.
Các normalization layer nói chung giúp chuẩn hoá đầu ra của mỗi layer trong neural network từ đó giúp ổn định hoá quá trình huấn luyện.

<img src="https://kharshit.github.io/img/batch_normalization.png" style="width: 400px;"/>

Bên cạnh Batch normalization layer, ta còn một số các normalization layer khác như Layer normalization layer, Instance normalization layer, Group normalization layer ...

### 1.4. Dropout layer

Chúng ta đã bàn luận về hiện tượng overfit trong các mô hình machine learning, trong neural network, hiện tượng overfit cũng là một vấn đề nan giải.
Ta có Dropout layer là một lớp giúp phần nào đó giảm bớt vấn đề overfit.

Cụ thể, dropout layer sẽ ngẫu nhiên lựa chọn một số các nơ ron trong neural network, chính xác hơn là lựa chọn ngẫu nhiên một số các trọng số của mô hình và gán giá trị bằng 0.
Từ đó, ta sẽ giảm được độ phức tạp của neural network, từ đó giảm được hiện tượng overfit.

<img src="https://www.baeldung.com/wp-content/uploads/sites/4/2020/05/2-1-2048x745-1.jpg" style="width: 1000px;"/>

## 2. Loss function và Optimization

### 2.1. Loss function

Chúng ta đã cùng nhau nghiên cứu về khá nhiều các loại hàm loss khác nhau và cụ thể cho từng bài toán khác nhau.

Đối với bài toán Regression ta có một số hàm loss như
- Mean absolute error

$$
MAE(\hat{y}, y) = \frac{1}{N}\sum_{i=1}^{N} |\hat{y}_i - y_i|
$$

- Mean squared error

$$
MSE(\hat{y}, y) = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

Đối với bài toán Classification ta có một số hàm loss như
- Binary cross entropy

$$
BCE(\hat{y}, y) = - \sum_{i=1}^{N} (y^i \log \hat{y}^i + (1 - y^i) \log(1 - \hat{y}^i))
$$

- Categorical cross entropy hay cross entropy

$$
CE(\hat{y}, y) = - \sum_{i=1}^{N} \sum_{j=1}^{K} (y^{ij} \log \hat{y}^{ij})
$$

### 2.2. Optimization

Chúng ta đã cùng nhau nghiên cứu về khá nhiều các thuật toán tối ưu khác nhau biến thể từ Gradient descent như: Stochastic gradient descent (SGD), Momentum, Nesterov accelerated gradient (NAG) ...

## 3. Model feedforward và backpropagation

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/7-Figure1-1-1.png" style="width: 500px;"/>

Feedforward là quá trình neural network nhận đầu vào input layer, thực hiện các phép tính toán qua các linear layer, activation layer, normalization layer ... để trả đầu ra output layer.

Ngược lại với feedforward, sau khi đưa kết quả dự đoán của neural network vào hàm loss và tính toán giá trị loss, Backpropagation là quá trình tính toán giá trị đạo hàm của hàm loss theo từng trọng số của neural network.
Sau khi tính được giá trị gradient tương ứng với mỗi trọng số, neural network thực hiện cập nhật lại các trọng số này theo thuật toán tối ưu dựa trên gradient descent.
