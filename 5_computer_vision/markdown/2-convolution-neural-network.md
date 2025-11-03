---
time: 11/04/2022
title: Mạng nơ ron tích chập Convolutional Neural Network
description: Mạng nơ ron tích chập Convolutional Neural Network (CNN) là một trong những kiến trúc mạng nơ ron phổ biến nhất trong lĩnh vực Computer Vision. CNN được xây dựng dựa trên phép nhân tích chập convolution, giúp mô hình học được các đặc tính không gian của ảnh đầu vào, giúp trích xuất các đặc trưng quan trọng của ảnh đầu vào.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/banner.jpeg
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Công thức của phép nhân tích chập Convolution

Phép nhân tích chập convolution là một kỹ thuật quan trọng xử lý ảnh số (digital image processing).
Nó xuất hiện trong các thuật toán xử lý ảnh như làm mờ (blur), làm nét (sharpen), làm rõ đường nét (edge detection).

Ngoài ra, với sự phát triển của deep learning, các mô hình xử lý ảnh computer vision cũng sử dụng rất nhiều các mạng nơ ron được xây dựng từ phép convolution.
Convolution layer là một tầng biến đổi ma trận đầu vào để làm rõ và tách ra các đặc tính của hình ảnh mà vẫn bảo toàn tính tương quan không gian giữa đầu ra và đầu vào.

[Đây](https://setosa.io/ev/image-kernels/) là một bài blog khá hay về trực quan hoá phép convolution trên ảnh.

### 1.1. Đối với ma trận hai chiều

Công thức của phép convolution được tính như sau:

$$ y_{11} = (x_{11}w_{11} + x_{12}w_{12} + x_{13}w_{13}) + (x_{21}w_{21} + x_{22}w_{22} + x_{23}w_{23}) + (x_{31}w_{31} + x_{32}w_{32} + x_{33}w_{33}) $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/convolution.gif" style="width: 1000px;"/>

### 1.2. Đối với ma trận ba chiều (ảnh RGB)

Đối với ảnh RGB, thay vì được đại diện bởi 1 con số, mỗi pixel được đại diện bởi 3 con số đại diện cho giá trị tương ứng lần lượt của màu đỏ (Red), xanh lá (Green) và xanh dương (Blue).

Do đó, phép convolution cũng phức tạp hơn, công thức của phép convolution đối với ảnh RGB được tính như sau:

$$ y^{R} = (x^{R}_{11}w^{R}_{11} + x^{R}_{12}w^{R}_{12} + x^{R}_{13}w^{R}_{13}) + (x^{R}_{21}w^{R}_{21} + x^{R}_{22}w^{R}_{22} + x^{R}_{23}w^{R}_{23}) + (x^{R}_{31}w^{R}_{31} + x^{R}_{32}w^{R}_{32} + x^{R}_{33}w^{R}_{33}) $$
$$ y^{G} = (x^{G}_{11}w^{G}_{11} + x^{G}_{12}w^{G}_{12} + x^{G}_{13}w^{G}_{13}) + (x^{G}_{21}w^{G}_{21} + x^{G}_{22}w^{G}_{22} + x^{G}_{23}w^{G}_{23}) + (x^{G}_{31}w^{G}_{31} + x^{G}_{32}w^{G}_{32} + x^{G}_{33}w^{G}_{33}) $$
$$ y^{B} = (x^{B}_{11}w^{B}_{11} + x^{B}_{12}w^{B}_{12} + x^{B}_{13}w^{B}_{13}) + (x^{B}_{21}w^{B}_{21} + x^{B}_{22}w^{B}_{22} + x^{B}_{23}w^{B}_{23}) + (x^{B}_{31}w^{B}_{31} + x^{B}_{32}w^{B}_{32} + x^{B}_{33}w^{B}_{33}) $$
$$ y = y^{R} + y^{G} + y^{B} + b $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/convolution_rgb.gif" style="width: 1000px;"/>

### 1.3. Một số kernel đặc biệt trong convolution

Trước khi được sử dụng rộng rãi trong mạng nơ ron, tạo ra mang nơ ron tích chập CNN, phép convolution là một phép toán được sử dụng nhiều trong xử lý ảnh.

Ta có thể sử dụng các kernel đặc biệt để thực hiện các phép toán khác nhau giúp biến đổi ảnh đầu vào, tạo ra các ảnh đầu ra với các đặc điểm mong muốn.
Phép convolution có thể được sử dụng để làm mờ ảnh, làm nét ảnh, phát hiện cạnh, phát hiện góc, phát hiện đường thẳng, phát hiện hình tròn, phát hiện hình chữ nhật ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/special_kernels.jpeg" style="width: 1000px;"/>

## 2. Các tham số quan trọng của phép convolution

### 2.1. Kernel size

Kernel size là kích thước của ma trận được sử dụng làm kernel trong phép convolution.
Kích thước của kernel sẽ ảnh hưởng đến kích thước của ma trận đầu ra sau khi thực hiện phép convolution.

Nếu kích thước của ma trận đầu vào là $n \times n$ và kích thước của kernel là $k \times k$, thì kích thước của ma trận đầu ra sẽ là:

$$ \left( n - k + 1 \right) \times \left( n - k + 1 \right) $$

Thông thường kernel là ma trận vuông, có kích thước là $3 \times 3$, $5 \times 5$, $7 \times 7$ ...
Trong một số trường hợp đặc biệt, kernel có thể là ma trận chữ nhật.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/kernel_size.gif" style="width: 600px;"/>

### 2.2. Stride

Stride có thể được gọi là bước nhảy trong phép tính convolution giữa các pixel của ma trận đầu vào.
Stride là một tham số quan trọng trong phép convolution, nó xác định khoảng cách giữa các lần tính toán của kernel trên ma trận đầu vào.

Với stride bằng 1, ta dịch chuyển kernel đến pixel ngay tiếp theo để tiếp tục tính toán, trong khi với stride bằng 2, ta dịch chuyển 2 bước.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/stride.gif" style="width: 600px;"/>

Nếu kích thước của ma trận đầu vào là $n \times n$, kích thước của kernel là $k \times k$ và stride là $s$, thì kích thước của ma trận đầu ra sẽ là:

$$ \left( \frac{n - k}{s} + 1 \right) \times \left( \frac{n - k}{s} + 1 \right) $$

### 2.3. Padding

Padding là thao tác mà ta bổ sung thêm một số pixel vào xung quanh của ma trận đầu vào trước khi tính toán convolution.
Padding giúp cho kích thước của ma trận đầu ra giống với kích thước của ma trận đầu vào.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/padding.gif" style="width: 600px;"/>

Kiểu padding phổ biến nhất là zero padding (hay black padding) tức là ta sẽ thêm các pixel có giá trị bằng 0 vào xung quanh ma trận đầu vào.
Một số kiểu padding khác như:
- Replication padding: lấy các pixel ở biên của ma trận đầu vào và nhân bản chúng vào các pixel padding lân cận.
- Reflection padding: lấy các pixel đối xứng qua pixel ở biên của ma trận đầu vào để điền vào các pixel padding.
- Circular padding: lấy các pixel ở biên của ma trận đầu vào và nối chúng lại với nhau theo chiều vòng tròn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/other_padding.jpeg" style="width: 1000px;"/>

Nếu kích thước của ma trận đầu vào là $n \times n$, kích thước của kernel là $k \times k$, stride là $s$ và padding là $p$, thì kích thước của ma trận đầu ra sẽ là:

$$ \left( \frac{n - k + 2p}{s} + 1 \right) \times \left( \frac{n - k + 2p}{s} + 1 \right) $$

## 3. Kiến trúc mạng nơ ron tích chập Convolution Neural Network CNN

Kiến trúc khái quát của một convolution neural network CNN được mô tả thông qua hình ảnh dưới đây.
Hình ảnh này được lấy từ bài báo [An Introduction to Convolutional Neural Networks](https://arxiv.org/abs/1511.08458).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/cnn.jpeg" style="width: 1000px;"/>

### 3.1. Convolution layer

Convolution layer là layer cơ bản nhất trong CNN, giúp thực hiện phép toán convolution trên ma trận đầu vào.

Điểm khác biệt của convolution layer so với một phép convolution đơn giản là convolution layer có thể chứa rất nhiều các kernel khác nhau và thực hiện nhiều phép convolution khác nhau trên cùng một ma trận đầu vào.
Điều này giúp cho convolution layer có thể học được nhiều đặc tính khác nhau của ảnh đầu vào, từ đó tạo ra các ma trận đầu ra (hay còn gọi là feature maps) với các đặc tính khác nhau.

Trong convolution layer trong CNN, ta có một tham số mới về số lượng kernel được sử dụng trong layer convolution đó.
Trong một số thư viện deep learning như PyTorch, tham số này được gọi là out_channel.

### 3.2. Pooling layer

Pooling layer là layer giúp giảm kích thước của feature maps trong CNN, giúp giảm đáng kể lượng tài nguyên tính toán cần thiết để phục vụ mô hình.

Ngoài ra, pooling layer phần nào cũng giúp làm giảm hiện tượng overfitting trong mô hình CNN.

Có nhiều loại pooling layer khác nhau, tuy nhiên, cơ chế hoạt động chính của pooling layer vẫn là chia feature maps thành các ô và thực hiện phép tính toán trên từng ô.

#### 3.2.1. Max pooling và Average pooling

Max pooling và Average pooling là hai kỹ thuật pooling nền tảng, góp phần xây dựng nên nhiều các kỹ thuật pooling khác nhau.

Max pooling và Average pooling đều thực hiện phép chia feature maps thành các ô có kích thước là một tham số được xác định trước, sau đó thực hiện phép tính toán trên từng ô.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/pooling.jpeg" style="width: 600px;"/>

Nếu kích thước đầu vào của feature maps là $n \times n$ và kích thước của ô là $k \times k$, thì kích thước của ma trận đầu ra sẽ là:

$$ \left( \frac{n}{k} \right) \times \left( \frac{n}{k} \right) $$

Max pooling lựa chọn giá trị max của mỗi ô làm giá trị đầu ra thì Average pooling tính trung bình các giá trị của mỗi ô làm giá trị đầu ra.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/pooling_comparison.jpeg" style="width: 1000px;"/>

Điểm mạnh của max pooling là giúp làm rõ hơn các chi tiết sáng, tuy nhiên điều này kéo theo việc max pooling làm mất đi các chi tiết ít sáng hơn.
Trong khi đó, average pooling gần như sao chép y hệt hình ảnh input ra output nhưng với kích thước nhỏ hơn.

#### 3.2.2. Adaptive pooling - RoI pooling

Với cách chia feature maps của max pooling và average pooling cơ bản ở trên, ta thu được output có tỷ lệ kích thước tương đối giống với tỷ lệ kích thước của input.
Trong khi đó, adaptive pooling hay RoI pooling tiếp cận việc chia feature maps theo một cách khác.

Adaptive pooling hay RoI pooling xác định trước kích thước đầu ra, sau đó chia đều feature maps input theo tỷ lệ kích thước của output.
Điều này giúp cho ta luôn đảm bảo được chính xác kích thước của output cho dù input có kích thước bất kỳ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/adaptive_pooling.jpeg" style="width: 600px;"/>

Sau khi thực hiện chia feature maps thành các ô, adaptive pooling cũng sẽ thực hiện max hoặc average pooling trên từng ô để tạo ra output cuối cùng.
Do đó, hiệu ứng của adaptive max pooling cũng tương tự như max pooling, trong khi adaptive average pooling cũng tương tự như average pooling.

#### 3.2.3. Global pooling

Global pooling là một kỹ thuật pooling đặc biệt, nó không chia feature maps thành các ô mà thực hiện phép tính toán trên toàn bộ feature maps.
Nói cách khác, global pooling sẽ biến đổi một features maps trực tiếp thành một vector.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/global_pooling.jpeg" style="width: 600px;"/>

Global max pooling sẽ lấy giá trị lớn nhất của toàn bộ feature maps làm giá trị đầu ra, trong khi global average pooling sẽ tính trung bình các giá trị của toàn bộ feature maps làm giá trị đầu ra.
Tuy nhiên, do biến đổi toàn bộ feature maps thành vector, hiệu ứng của global max pooling và global average pooling sẽ khó nhận biết so với max pooling và average pooling.

### 3.3. Flatten layer

Trong CNN, ta thường xuyên làm việc với các ma trận input và output, tuy nhiên, để đưa ra giá trị dự đoán cuối cùng, CNN cần có một layer giúp biến đổi các ma trận nhiều chiều thành vector.
Từ vector này, ta sẽ sử dụng các linear layer để đưa ra kết quả dự đoán cuối cùng phục vụ các bài toán classification, regression ...

Flatten layer là một layer đặc biệt trong CNN, nó không thực hiện phép toán gì cả mà chỉ đơn giản là biến đổi ma trận đầu ra của các layer trước đó thành vector một chiều.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/flatten.jpeg" style="width: 1000px;"/>

Flatten layer là layer giúp biến đổi các ma trận đầu ra của các layer trước đó thành vector một chiều.
Flatten layer sẽ nối các giá trị của ma trận đầu ra thành một vector dài.
Ví dụ, nếu ma trận đầu ra có kích thước là $n \times m$, thì Flatten layer sẽ biến đổi ma trận này thành một vector có số phần tử là $n \times m$.

### 3.4. Convolution transpose layer

Convolution transpose layer là một layer đặc biệt trong CNN, nó thực hiện phép toán ngược lại với phép toán convolution.
Nếu như đối với phép toán convolution, ma trận đầu vào được biến đổi thành ma trận đầu ra nhỏ hơn, thì đối với phép toán convolution transpose, ma trận đầu vào sẽ được biến đổi thành ma trận đầu ra lớn hơn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/convolution_transpose.gif" style="width: 1000px;"/>

Convolution transpose layer thường được sử dụng trong các mô hình giải quyết bài toán phân đoạn ảnh (image segmentation) như U-Net hay những mô hình sinh ra dữ liệu ảnh như GANs hay Diffusion models.

## 4. Các mô hình CNN nổi tiếng

### 4.1. VGG

VGG viết tắt của Visual Geometry Group, tổ chức đã nghiên cứu và công bố mô hình VGG vào năm 2014.
Bài báo công bố mô hình VGG là [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).
Các hình ảnh trong phần này được lấy từ bài báo trên.

Có hai phiên bản nổi tiếng của VGG là VGG-16 và VGG-19.
Trong đó, 16 và 19 ở đây đại diện cho số lượng các layer convolution trong mô hình.

VGG là một trong những mô hình CNN đầu tiên được xây dựng với kiến trúc rất đơn giản, chỉ sử dụng các layer convolution và max pooling.
- Mô hình VGG sử dụng các kernel có kích thước là $3 \times 3$ và stride là 1, do đó, kích thước của ma trận đầu ra sẽ gần như giống với kích thước của ma trận đầu vào.
- Mô hình VGG sử dụng các layer max pooling với kích thước kernel là $2 \times 2$ và stride là 2, do đó, kích thước của ma trận đầu ra sẽ giảm đi một nửa so với kích thước của ma trận đầu vào.
- Mô hình VGG sử dụng các layer fully connected để đưa ra kết quả dự đoán cuối cùng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/vgg.jpeg" style="width: 800px;"/>

Mô hình VGG, ở thời điểm nó ra đời, đã đạt được kết quả rất tốt trên bộ dữ liệu ImageNet và trở thành một trong những mô hình CNN phổ biến nhất trong lĩnh vực computer vision.

Sau này, với sự phát triển của các bộ dữ liệu lớn, VGG đã gặp phải một số vấn đề khi tăng kích thước của mô hình, cụ thể là vấn đề vanishing gradient khi mô hình quá sâu.
Do đó, các mô hình CNN sau này đã được cải tiến và phát triển dựa trên VGG.

### 4.2. Inception (GoogleNet)

Inception là một kiến trúc mô hình CNN được phát triển bởi Google vào năm 2014.
Bài báo công bố mô hình Inception là [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842).
Các hình ảnh trong phần này được lấy từ bài báo trên.

Inception được xây dựng dựa trên ý tưởng về việc sử dụng nhiều kernel với kích thước khác nhau trong cùng một layer convolution, ví dụ như $1 \times 1$, $3 \times 3$, $5 \times 5$.
Điều này giúp mô hình học được các đặc tính khác nhau của ảnh đầu vào và cải thiện độ chính xác của mô hình.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/inception.jpeg" style="width: 1000px;"/>

Nhược điểm của Inception là mô hình có kích thước lớn, do đó, cần nhiều tài nguyên tính toán để huấn luyện và triển khai mô hình.
Một số phiên bản nâng cấp của Inception:
- Inception-v2: cải tiến Inceptionv1 ở việc sử dụng batch normalization, giảm kích thước của mô hình.
- Inception-v3: cải tiến Inceptionv2 ở việc sử dụng factorization để giảm kích thước của kernel, sử dụng auxiliary classifier để cải thiện độ chính xác của mô hình.
- Inception-v4: cải tiến Inceptionv3 ở việc sử dụng các residual connection để cải thiện độ chính xác của mô hình.

### 4.3. ResNet

ResNet viết tắt của Residual Network, là một kiến trúc mô hình CNN được xây dựng bởi Microsoft vào năm 2015.
Bài báo công bố mô hình ResNet là [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
Các hình ảnh trong phần này được lấy từ bài báo trên.

ResNet được xây dựng dựa trên ý tưởng về việc sử dụng các residual connection để giải quyết vấn đề vanishing gradient trong các mô hình CNN sâu và ResNet là một ý tưởng cực kỳ đột phá ở thời điểm nó ra đời.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/residual_connection.jpeg" style="width: 400px;"/>

Residual connection là một kết nối giữa đầu vào và đầu ra của một layer convolution, giúp cho gradient có thể được lan truyền thông qua hai đường, một đường đi qua layer convolution và một đường đi thẳng từ đầu vào đến đầu ra.
Điều này giúp cho gradient có thể được lan truyền qua các layer sâu mà không bị mất đi, giúp mô hình học được các đặc tính của ảnh đầu vào.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/residual_connection_example.jpeg" style="width: 800px;"/>

ResNet hướng đến việc xây dựng mô hình CNN rất sâu, rất nhiều layer, rất nhiều trọng số giúp ResNet trở thành một mô hình CNN trích xuất đặc trưng của hình ảnh rất tốt, học được trên những bộ dữ liệu lớn và phức tạp.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/resnet.jpeg" style="width: 800px;"/>

ResNet có một số phiên bản nổi tiếng là ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, ResNet-200 được sử dụng với các bộ dữ liệu có kích thước khác nhau.
ResNet đã đạt được kết quả rất tốt trên bộ dữ liệu ImageNet và trở thành một trong những mô hình CNN pretrained phổ biến nhất trong lĩnh vực computer vision.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/resnet_params.jpeg" style="width: 1000px;"/>

Một số phiên bản mô hình nâng cấp hơn của ResNet:
- ResNeXt: cải tiến ResNet ở việc sử dụng group convolution để giảm kích thước của mô hình.
- ResNest: cải tiến ResNeXt ở việc sử dụng split attention để cải thiện độ chính xác của mô hình.

### 4.4. MobileNet

MobileNet là một kiến trúc mô hình CNN được phát triển bởi Google vào năm 2017.
Bài báo công bố mô hình MobileNet là [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861).
Các hình ảnh trong phần này được lấy từ bài báo trên.

MobileNet, như tên gọi của nó, là một mô hình CNN siêu gọn nhẹ, có thể chạy được trên các thiết bị di động và nhúng với tài nguyên tính toán hạn chế.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/mobile_net.jpeg" style="width: 1000px;"/>

MobileNet có số lượng trọng số của mô hình rất ít (nếu so sánh với VGG hay ResNet) nhờ sử dụng một layer convolution đặc biệt gọi là depthwise separable convolution gồm hai bước:
- Depthwise convolution: thực hiện phép convolution với kernel kích thước $k \times k$ trên từng kênh của ảnh đầu vào, tức là mỗi kênh sẽ được xử lý riêng biệt.
- Pointwise convolution: thực hiện phép convolution với kernel kích thước $1 \times 1$ trên toàn bộ ảnh đầu vào, tức là kết hợp các kênh của ảnh đầu vào lại với nhau.
Điều này giúp giảm số lượng trọng số của mô hình rất nhiều, từ đó giảm kích thước của mô hình và tăng tốc độ tính toán.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/depthwise_separable_convolution.jpeg" style="width: 600px;"/>

Tất nhiên là với kích thước mô hình nhỏ hơn nhiều, trong một số bài toán và bộ dữ liệu cụ thể, MobileNet không thể cạnh tranh được với ResNet hay VGG hay các kiến trúc CNN khác về độ chính xác, tuy nhiên, với lợi thế về tốc độ tính toán rất nhanh, MobileNet vẫn thường được sử dụng trong một số thiết bị di động hoặc trong một số trường hợp cần mô hình nhỏ trong thực tế.
