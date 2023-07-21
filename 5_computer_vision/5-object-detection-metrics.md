---
layout: "post"
title:  "Object detection metrics"
author: "Nguyễn Hữu Minh"
permalink: "/deep-learning/computer-vision/object-detection-metrics"
parent: "Computer vision"
grand_parent: "Deep learning"
nav_order: 6
---

# Metrics trong bài toán object detection

## 1. Một số khái niệm quan trọng

### 1.1. Intersection Over Union (IoU)

Intersection Over Union (gọi tắt là IoU) là chỉ số được tính toán dựa trên Jaccard Index nhằm đánh giá độ trùng nhau giữa 2 bbox trong bài toán object detection (bbox groundtruth và bbox prediction).

IoU được tính bằng thương của diện tích giao của hai bbox và diện tích hợp của hai bbox.

<img src="https://raw.githubusercontent.com/rafaelpadilla/Object-Detection-Metrics/master/aux_images/iou.png" style="width: 700px;"/>

### 1.2. TP, FP, FN và TN

Từ khái niệm về IoU, ta có các khái niệm:
- True Positive (TP): Bbox prediction được gọi là TP nếu IoU giữa bbox groundtruth và bbox prediction $\geq$ ngưỡng IoU (IoU threshold).
- False Positive (FP): Bbox prediction được gọi là FP nếu IoU giữa bbox groundtruth và bbox prediction $\lt$ ngưỡng IoU (IoU threshold).
- False Negative (FN): Là các bbox groundtruth không được dự đoán
- True Negative (TN): Không được sử dụng trong bài toán object detection

Đối với các bài toán object detection có nhiều class hơn 2 (class object và class background), bbox prediction được gọi là TP không những phải thoả mãn yêu cầu về IoU mà còn phải thoả mãn yêu cầu về class dự đoán phải đúng.

<img src="https://pyimagesearch.com/wp-content/uploads/2016/09/iou_examples.png" style="width: 700px;"/>

Trong quá trình đánh giá mô hình object detection, ta thường sử dụng ba ngưỡng là 0.5, 0.75 và 0.9.

### 1.3. Precision và Recall

Tương tự như việc đánh giá mô hình classification, precision và recall cũng được sử dụng để đánh giá mô hình object detection.

$$
\text{precision} = \frac{\text{TP}}{\text{TP + FP}}
$$

$$
\text{recall} = \frac{\text{TP}}{\text{TP + FN}}
$$

## 2. Các metrics phổ biến trong bài toán object detection

### 2.1. Precision x Recall curve

Precision x Recall curve là một biểu đồ giúp đánh giá tương quan về đánh đổi giữa precision và recall.

<img src="https://www.statology.org/wp-content/uploads/2021/09/precisionRecall2.png" style="width: 700px;"/>

### 2.2. Average Precision (AP) và Mean Average Precision (mAP)

Average Precision (gọi tắt là AP) là metrics phổ biến nhất dùng đánh giá mô hình object detection.

AP được tính bằng chỉ số Area under Curve (AuC) của Precision x Recall curve.
Thay vì quan sát hai Precision x Recall curve xem đường nào tốt hơn thì ta tính toán ra một chỉ số để dễ dàng đánh giá được.

Trong thực tế, tên gọi Average Precision có nghĩa là Precision được tính Average trên khoảng Recall từ 0 đến 1.

Ngoài ra, đối với những mô hình object detection với nhiều class object khác nhau, ta có chỉ số Mean Average Precision (mAP) được tính bằng trung bình chỉ số AP trên tất cả các class object.

### 2.3. Ví dụ minh hoạ việc tính toán AP

Ta có các bbox prediction màu đỏ (được ký hiệu các ký tự A, B, C ... Y) với các chỉ số confidence bên cạnh, các bbox groundtruth màu xanh lá.

<img src="https://raw.githubusercontent.com/rafaelpadilla/Object-Detection-Metrics/master/aux_images/samples_1_v2.png" style="width: 1000px;"/>

Với việc lấy IoU $\geq$ 0.3, ta có từng bbox prediction là các bbox TP hay FN như bảng dưới

<img src="https://raw.githubusercontent.com/rafaelpadilla/Object-Detection-Metrics/master/aux_images/table_1_v2.png" style="width: 500px;"/>

Sau khi xác định các bbox prediction là TP hay FN, ta sắp xếp chúng theo thứ tự về confidence để tính toán Precision và Recall.

<img src="https://raw.githubusercontent.com/rafaelpadilla/Object-Detection-Metrics/master/aux_images/table_2_v2.png" style="width: 900px;"/>

Với mỗi cặp giá trị Precision và Recall vừa tính toán được, ta thu được Precision x Recall curve

<img src="https://raw.githubusercontent.com/rafaelpadilla/Object-Detection-Metrics/master/aux_images/precision_recall_example_1_v2.png" style="width: 800px;"/>

Tiếp theo, ta xấp xỉ đường Precision trên bằng một đường gọi là Interpolated Precision.
Đến đây, ta có hai cách để tính đường interpolated precision.

#### 11-point interpolation

Cách đầu tiên được gọi là 11-point interpolation, ta chia trục recall của Precision x Recall curve thành 10 phần với 11 mốc recall (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1).
Với mỗi mốc recall, ta lấy chỉ số precision cao nhất mà nhận chỉ số recall lớn hơn hoặc bằng mốc recall đang xét.

<img src="https://raw.githubusercontent.com/rafaelpadilla/Object-Detection-Metrics/master/aux_images/11-pointInterpolation.png" style="width: 800px;"/>

Cuối cùng, ta tính AP bằng cách lấy trung bình cộng của các giá trị precision lấy được ở bước trên.

$$
\text{AP} = \frac{1}{11} \sum_{r \in \{0, 0.1, ... 0.9, 1 \}} \text{p}_{\text{interp}}(\text{r})
$$

$$
\text{AP} = \frac{1}{11} (1 + 0.6666 + 0.4285 + 0.4285 + 0.4285 + 0 + 0 + 0 + 0 + 0 + 0 + 0) = 0.268372 \approx 26.84\%
$$

#### All-point interpolation

Cách thứ hai được gọi là all-point interpolation, ta tính interpolated precision với chính xác các mốc recall của Precision x Recall curve.

<img src="https://raw.githubusercontent.com/rafaelpadilla/Object-Detection-Metrics/master/aux_images/interpolated_precision_v2.png" style="width: 750px;"/>

Cuối cùng, ta tính toán AuC của đường interpolated precision để thu được chỉ số AP cần tính

<img src="https://raw.githubusercontent.com/rafaelpadilla/Object-Detection-Metrics/master/aux_images/interpolated_precision-AUC_v2.png" style="width: 750px;"/>

$$
\text{AP} = \text{A1} + \text{A2} + \text{A3} + \text{A4}
$$

$$
\text{A1} = (0.0666 - 0) \times 1 = 0.0666
$$

$$
\text{A1} = (0.1333 - 0.0666) \times 0.0666 = 0.04446222
$$

$$
\text{A1} = (0.4 - 0.1333) \times 0.4285 = 0.11428095
$$

$$
\text{A1} = (0.4666 - 0.4) \times 0.3043 = 0.02026638
$$

$$
\text{AP} = 0.0666 + 0.04446222 + 0.11428095 + 0.02026638 = 0.24560955 \approx 24.56\%
$$

Hai cách tính chỉ số AP khác nhau là 11-point interpolation và All-point interpolation cho ta hai giá trị AP khác nhau một chút là 26.84% và 24.56%.
