---
time: 11/04/2022
title: Mạng nơ ron tích chập Convolutional Neural Network
description: Mạng nơ ron tích chập Convolutional Neural Network (CNN) là một trong những kiến trúc mạng nơ ron phổ biến nhất trong lĩnh vực Computer Vision. CNN được xây dựng dựa trên phép nhân tích chập convolution, giúp mô hình học được các đặc tính không gian của ảnh đầu vào, giúp trích xuất các đặc trưng quan trọng của ảnh đầu vào.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/banner.png
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

$$y_{11} = (x_{11} ∗ w_{11} + x_{12} ∗ w_{12} + x_{13} ∗ w_{13}) + (x_{21} ∗ w_{21} + x_{22} ∗w_{22} + x_{23} ∗ w_{23}) + (x_{31} ∗ w_{31} + x_{32} ∗ w_{32} + x_{33} ∗ w_{33})$$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/convolution.gif" style="width: 1000px;"/>

### 1.2. Đối với ma trận ba chiều (ảnh RGB)

Đối với ảnh RGB, thay vì được đại diện bởi 1 con số, mỗi pixel được đại diện bởi 3 con số đại diện cho giá trị tương ứng lần lượt của màu đỏ (Red), xanh lá (Green) và xanh dương (Blue).

Do đó, phép convolution cũng phức tạp hơn, công thức của phép convolution đối với ảnh RGB được tính như sau:

$$
y_{R} = (x_{R11} ∗ w_{R11} + x_{R12} ∗ w_{R12} + x_{R13} ∗ w_{R13}) + (x_{R21} ∗ w_{R21} + x_{R22} ∗ w_{R22} + x_{R23} ∗ w_{R23}) + (x_{R31} ∗ w_{R31} + x_{R32} ∗ w_{R32} + x_{R33} ∗ w_{R33})
y_{G} = (x_{G11} ∗ w_{G11} + x_{G12} ∗ w_{G12} + x_{G13} ∗ w_{G13}) + (x_{G21} ∗ w_{G21} + x_{G22} ∗ w_{G22} + x_{G23} ∗ w_{G23}) + (x_{G31} ∗ w_{G31} + x_{G32} ∗ w_{G32} + x_{G33} ∗ w_{G33})
y_{B} = (x_{B11} ∗ w_{B11} + x_{B12} ∗ w_{B12} + x_{B13} ∗ w_{B13}) + (x_{B21} ∗ w_{B21} + x_{B22} ∗ w_{B22} + x_{B23} ∗ w_{B23}) + (x_{B31} ∗ w_{B31} + x_{B32} ∗ w_{B32} + x_{B33} ∗ w_{B33})

y = y_{R} + y_{G} + y_{B} + b
$$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/convolution_rgb.gif" style="width: 1000px;"/>

### 1.3. Một số kernel đặc biệt trong convolution

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/special_kernels.png" style="width: 1000px;"/>

## 2. Các tham số quan trọng của phép convolution

### 2.1. Kernel size

Kernel size là kích thước của ma trận được sử dụng làm kernel.

Thông thường kernel là ma trận vuông, có kích thước là $3 \times 3$, $5 \times 5$, $7 \times 7$ ...
Trong một số trường hợp đặc biệt, kernel có thể là ma trận chữ nhật.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/kernel_size.gif" style="width: 1000px;"/>

### 2.2. Stride

Stride có thể được gọi là bước nhảy trong phép tính convolution.
Với stride bằng 1, ta dịch chuyển kernel đến pixel ngay tiếp theo để tiếp tục tính toán, trong khi với stride bằng 2, ta dịch chuyển 2 bước.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/stride.gif" style="width: 1000px;"/>

### 2.3. Padding

Padding là thao tác mà ta bổ sung thêm một số pixel vào xung quanh của ma trận đầu vào trước khi tính toán convolution.
Padding giúp cho kích thước của ma trận đầu ra giống với kích thước của ma trận đầu vào.
Có một số kiểu padding như black padding, reflect padding ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/padding.gif" style="width: 1000px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/2-convolution-neural-network/other_padding.gif" style="width: 1000px;"/>

## 3. Mạng nơ ron tích chập Convolution Neural Network CNN

Kiến trúc khái quát của một convolution neural network CNN được mô tả như sau:

<img src="https://editor.analyticsvidhya.com/uploads/67201cnn.jpeg" style="width: 1200px;"/>

### 2.1. Convolution layer

Convolution layer là layer cơ bản nhất trong CNN, giúp thực hiện phép convolution.

Điểm khác biệt của convolution layer so với một phép convolution đơn giản là convolution layer có thể chứa rất nhiều các kernel khác nhau và thực hiện nhiều phép convolution khác nhau.
Do đó, trong convolution layer, ta có thể một tham số về số lượng kernel được sử dụng trong layer convolution đó (trong một số thư viện deep learning, tham số này được gọi là out_channel).

### 2.2. Pooling layer

Pooling layer là layer giúp giảm kích thước của feature maps trong CNN.

Pooling layer chia feature maps thành các ô và thực hiện phép tính toán trên từng ô.

#### 2.2.1. Max pooling và Average pooling

Max pooling và Average pooling chia feature maps thành các ô có kích thước là một tham số được xác định trước.

<img src="https://www.researchgate.net/publication/333593451/figure/fig2/AS:765890261966848@1559613876098/Illustration-of-Max-Pooling-and-Average-Pooling-Figure-2-above-shows-an-example-of-max.png" style="width: 1200px;"/>

Max pooling lựa chọn giá trị max của mỗi ô làm giá trị đầu ra thì Average pooling tính trung bình các giá trị của mỗi ô làm giá trị đầu ra.

<img src="https://mriquestions.com/uploads/3/4/5/7/34572113/example-max-and-avg-pooling_orig.png" style="width: 1200px;"/>

Điểm mạnh của max pooling là giúp làm rõ hơn các chi tiết sáng, tuy nhiên điều này kéo theo việc max pooling làm mất đi các chi tiết ít sáng hơn.

Trong khi đó, average pooling gần như sao chép y hệt hình ảnh input ra output nhưng với kích thước nhỏ hơn.

#### 2.2.3. Adaptive pooling - RoI pooling

Với cách chia feature maps của max pooling và average pooling cơ bản ở trên, ta thu được output có tỷ lệ kích thước tương đối giống với tỷ lệ kích thước của input.

Adaptive pooling tiếp cận việc chia feature maps theo một cách khác.
Adaptive pooling xác định trước kích thước đầu ra, sau đó chia đều feature maps input theo tỷ lệ kích thước của output.
Điều này giúp cho ta luôn đảm bảo được chính xác kích thước của output với input có kích thước bất kỳ.

<img src="https://arthurdouillard.com/figures/roi_pooling2.svg" style="width: 1200px;"/>

Sau khi thực hiện chia feature maps thành các ô, Adaptive pooling cũng sẽ thực hiện max hoặc average pooling trên từng ô.

### 2.3. Flatten layer

Trong CNN, ta thường xuyên làm việc với các ma trận input và output, tuy nhiên, để đưa ra giá trị dự đoán cuối cùng, CNN cần có một layer giúp biến đổi các ma trận nhiều chiều thành vector.

Flatten layer lần lượt nối các giá trị của ma trận output thành một vector dài nhằm đưa các giá trị này qua các linear layer và cho ra kết quả dự đoán cuối cùng của mô hình CNN.

<!-- ### 2.4. Convolution transpose layer -->


## 4. Các mô hình CNN nổi tiếng

### 3.1. VGG

VGG viết tắt của Visual Geometry Group, tổ chức đã nghiên cứu và công bố mô hình VGG.

Có hai phiên bản nổi tiếng của VGG là VGG-16 và VGG-19.
16 và 19 ở đây đại diện cho số lượng các layer convolution trong mô hình.

<img src="https://viso.ai/wp-content/uploads/2021/10/vgg-neural-network-architecture.png" style="width: 1200px;"/>

VGG là một mô hình có kiến trúc đơn giản, phù hợp với một số các bộ dữ liệu nhỏ hoặc đơn giản.

### 3.2. ResNet

ResNet viết tắt của Residual Network, là một kiến trúc mô hình CNN đột phá ở thời điểm nó ra đời.
ResNet hướng đến việc xây dựng mô hình CNN rất sâu, rất nhiều layer, rất nhiều trọng số giúp giải quyết được nhiều bài toán, nhiều bộ dữ liệu phức tạp.
ResNet có một số phiên bản nổi tiếng là ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, ResNet-200 ...

<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-09-25_at_10.26.40_AM_SAB79fQ.png" style="width: 1200px;"/>

Tuy nhiên, khi xây dựng các mô hình quá sâu, trong quá trình training thường gặp phải vấn đề vanishing gradient.
Do đó, nhóm tác giả của ResNet đã đề xuất ý tưởng về residual connection và residual block, giúp giải quyết vấn đề này đối với các mạng CNN có kích thước lớn.

<img src="https://d2l.ai/_images/residual-block.svg" style="width: 1200px;"/>

Một số phiên bản mô hình nâng cấp hơn của ResNet là ResNext, ResNest ...

### 3.3. MobileNet

MobileNet, giống với cái tên của nó, là một mô hình CNN siêu gọn nhẹ có thể chạy được trên các thiết bị di động.
MobileNet có số lượng trọng số của mô hình rất ít (nếu so sánh với VGG hay ResNet) nhờ sử dụng một layer convolution đặc biệt gọi là Depthwise Separable Convolution.

<img src="https://images.viblo.asia/0e04f1cd-6a0f-4170-a2ab-c91341a2ef9e.png" style="width: 1200px;"/>

Tất nhiên là với kích thước mô hình nhỏ hơn nhiều, trong một số bài toán và bộ dữ liệu cụ thể, MobileNet không thể cạnh tranh được với ResNet hay VGG hay các kiến trúc CNN khác về độ chính xác, tuy nhiên, với lợi thế về tốc độ tính toán rất nhanh, MobileNet vẫn thường được sử dụng trong một số thiết bị di động hoặc trong một số trường hợp cần mô hình nhỏ trong thực tế.


<!-- ### 3.1. VGG
### 3.1. VGG -->

<!-- ## 4. Các ứng dụng khác của kiến trúc CNN trong các bài toán computer vision -->

