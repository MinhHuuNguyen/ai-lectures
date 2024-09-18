---
slug: machine-learning
time: "11/09/2024"
title: "Machine learning"
description: "Machine Learning (ML) là một phần của trí tuệ nhân tạo (AI) mà chúng ta dùng để xây dựng các mô hình hoặc chương trình máy tính có khả năng tự học từ dữ liệu."
author: "Nguyễn Hữu Minh"
banner_url: "https://tenten.vn/tin-tuc/wp-content/uploads/2023/08/1cG6U1qstYDijh9bPL42e-Q.jpg"
tags:
is_highlight: false
---

# Mô hình single-stage object detection

## 1. Kiến trúc Feature Pyramid Networks

Feature Pyramid Networks (gọi tắt là FPN) được giới thiệu như một kiến trúc backbone trong các mô hình object detection nhằm giải quyết vấn đề chênh lệch về kích thước giữa các đối tượng trong ảnh.

### 1.1. So sánh các kiến trúc pyramid khác nhau

<img src="https://production-media.paperswithcode.com/methods/new_teaser_TMZlD2J.jpg" style="width: 800px;"/>

Ý tưởng về việc xây dựng và sử dụng các đặc trưng của ảnh với nhiều kích thước khác nhau không mới, tuy nhiên, các giải pháp đã có vào th đó đều vướng phải một số vấn đề:
- Featurized image pyramid: Sử dụng nhiều kích thước ảnh khác nhau để tạo ra feature maps đạt được hiệu quả cao về độ chính xác, nhưng phương pháp này khiến cho mô hình trở nên cồng kềnh và tốn rất nhiều thời gian để xử lý và gần như bất khả thi để có thể train được.
- Single feature map: Việc sử dụng chỉ một kích thước đặc trưng duy nhất giúp xử lý nhanh hơn nhưng lại khiến cho mô hình khó có thể học được những đặc trưng giữa các đối tượng có kích thước chênh lệch trong ảnh.
- Pyramidal feature hierarchy: Việc sử dụng nhiều feature maps có kích thước khác nhau cùng đưa ra kết quả được sử dụng trong mô hình object detection khá nổi tiếng là SSD.
- Feature Pyramid Network: Dựa trên vấn đề trên từ SSD, nhóm tác giả đề xuất FPN tận dụng tối đa các feature maps trích xuất được từ backbone nhằm tạo ra bộ feature map mới gồm nhiều kích thước khác nhau và chứa rất nhiều thông tin về nội dung của ảnh đầu vào. 
Để đạt được điều này, nhóm tác giả thiết kế kiến trúc kết hợp những feature maps có kích thước lớn và những feature maps có kích thước nhỏ bằng top-down pathway và lateral connections.

### 1.2. Kiến trúc mô hình Feature Pyramid Networks

Ý tưởng về việc sử dụng kiến trúc theo dạng top-down không phải là mới và đã được nhắc đến trong một số nghiên cứu. 
Tuy nhiên, điểm giống nhau của các nghiên cứu có thiết kế theo kiểu top-down đó là mô hình chỉ sử dụng một feature map cuối cùng, sau khi đã tổng hợp các thông tin trong suốt quá trình top-down, để đưa ra quyết định dự đoán cuối cùng.

Trong khi đó, đối với FPN, nhóm tác giả đưa ra quyết định dự đoán trên từng feature maps trong suốt quá trình top-down. 
Từ đó, đặc biệt nâng cao chất lượng của mô hình object detection khi có thể vừa trích xuất được thông tin của các đối tượng có kích thước lớn từ các feature map có kích thước nhỏ vừa trích xuất được thông tin của các đối tượng có kích thước nhỏ từ các feature map có kích thước lớn.

<img src="https://media.arxiv-vanity.com/render-output/6504128/x2.png" style="width: 500px;"/>

Kiến trúc FPN có thể được áp dụng với nhiều backbone Conv khác nhau như AlexNet, VGG hay ResNet, cụ thể trong nghiên cứu, nhóm tác giả lựa chọn ResNet làm backbone. 

<img src="https://pic2.zhimg.com/80/v2-dacf2d16e42d6bb90596f947ec0044f9_1440w.webp" style="width: 500px;"/>

Kiến trúc FPN có thể được chia làm hai phần:
- Bottom-up pathway: là quá trình mà ta đưa ảnh đầu vào qua backbone như ResNet và thu được các feature map.
    - Trong các backbone Conv, sẽ có một nhóm các lớp Conv tạo ra các feature map có kích thước giống nhau, và nhóm các lớp Conv này được gọi là một khối Conv.
    Đối với FPN, nhóm tác giả lựa chọn các feature map được sinh ra từ các lớp Conv cuối cùng trong mỗi khối Conv để sử dụng cho nhánh Top-down pathway. 
    - Đối với backbone ResNet, nhóm tác giả sử dụng các feature maps được sinh ra từ residual block cuối cùng của mỗi khối Conv (trừ khối Conv đầu tiên do kích thước của feature maps này lớn và gây ra vấn đề về bộ nhớ), ký hiệu là C2, C3, C4, C5.
    Các feature maps này có kích thước lần lượt bằng 1/4, 1/8, 1/16 và 1/32 so với kích thước của ảnh đầu vào.
- Top-down pathway và lateral connections là quá trình mà FPN sinh ra thêm các feature maps mới từ các feature maps của Bottom-up pathway và kết hợp chúng lại thông qua lateral connections.
    - Các feature maps của Bottom-up pathway được đưa qua các lớp Conv có kích thước 1x1, stride bằng 1 nhằm giữ nguyên kích thước chiều dài, chiều rộng và chỉ thay đổi kích thước chiều channel của feature maps.
    - Các feature maps ở vị trí cao hơn (có kích thước nhỏ hơn) được upsample thông qua thuật toán nearest neighbor và cộng ma trận với feature maps đầu ra từ lớp Conv 1x1 nói trên.
    - Cuối cùng, các feature maps đầu ra từ phép cộng ma trận nói trên được đi qua một lớp Conv 3x3 có cùng số đầu ra channel của feature maps nhằm giảm bớt hiệu ứng của thuật toán nearest neighbor và tạo ra các feature maps đầu ra cuối cùng có cùng số channel với nhau.
    - Tập hợp feature maps này được gọi là P2,P3,P4,P5 tương ứng với các feature maps có cùng kích thước C2,C3,C4,C5.

### 1.3. Vấn đề tồn đọng của kiến trúc mô hình FPN

Kiến trúc FPN ra đời đã tạo ra một trong số những kiến trúc backbone kinh điển trong các bài toán computer vision nói chung và bài toán object detection nói riêng. 
Kiến trúc FPN đã giúp cho nhiều mô hình đạt độ chính xác cao hơn và trong khi thời gian xử lý không bị tăng một cách đáng kể. 

Tuy nhiên, đối với cụ thể bài toán object detection, việc kết hợp kiến trúc FPN vào Faster R-CNN mới chỉ cải thiện về mặt độ chính xác mà chưa giúp tăng tốc.
Vẫn còn một câu hỏi cần phải được giải quyết đó là làm sao để duy trì được độ chính xác mà FPN mang lại những mô hình object detection vẫn có để đạt tốc độ nhanh hơn nữa.

## 2. Tổng quan về các mô hình single-stage object detection

Các mô hình single-stage object detection ở thời điểm ban đầu đa phần chỉ sử dụng một backbone kết hợp thêm với các lớp Conv và lớp fully connected để đưa ra dự đoán về lớp của đối tượng trong ảnh và độ lệch của bbox so với groundtruth.

<img src="https://i.stack.imgur.com/xA4qz.png" style="width: 1200px;"/>

Việc loại bỏ Region proposals module khiến các mô hình single-stage object detection cần phải xây dựng một phương pháp riêng nhằm đề xuất ra các anchor chứa đối tượng.
Hai mô hình single-stage object detection nổi tiếng vào thời điểm đó là YOLO và SSD có các cách đề xuất ra anchor tương tự với nhau.

<img src="https://leimao.github.io/images/blog/2019-04-15-YOLOs/yolo_v1_diagram.png" style="width: 800px;"/>

YOLO đề xuất ra các anchor thông qua việc chia ảnh đầu vào thành dạng grid có kích thước SxS và với mỗi grid sẽ trả đầu ra dự đoán có kích thước SxS(Bx5+C).
Nếu tâm của một bbox nằm trong ô nào trên grid, ô đó sẽ cần phải được dự đoán là chứa đối tượng. 
Mỗi ô trên grid sẽ được mô hình dự đoán (Bx5+C) giá trị, trong đó:
- B là số lượng bbox dự đoán.
- 5 là các giá trị trong đó có 4 giá trị x, y, w, h đại diện cho bbox được dự đoán và 1 giá trị confidence. 
Thay vì được học là 1 nếu anchor có IoU cao với groundtruth bbox và ngược lại là 0 nếu anchor có IoU thấp với groundtruth bbox, điểm đặc biệt về giá trị confidence mà nhóm tác giả thiết kế trong mô hình YOLO là nó bằng chính giá trị IoU so với groundtruth.
- C là số lượng lớp đối tượng trong bài toán object detection. Mỗi giá trị dự đoán trong C là giá trị xác suất điều kiện nếu ô trên grid chứa đối tượng thì đó là đối tượng nào.

Trong nghiên cứu, nhóm tác giả của YOLO sử dụng S = 7,B = 2,C = 20.

<img src="https://www.researchgate.net/profile/Dumitru-Erhan/publication/286513835/figure/fig1/AS:613509750616127@1523283531509/SSD-framework-a-SSD-only-needs-an-input-image-and-ground-truth-boxes-for-each-object.png" style="width: 800px;"/>

SSD cũng sử dụng feature maps như là các dạng grid của ảnh đầu vào nhưng thay vì sử dụng một grid như YOLO thì SSD sử dụng nhiều grid từ nhiều feature maps có cách kích thước khác nhau.
Với mỗi grid tạo bởi một feature maps có kích thước mxn, SSD trả đầu ra dự đoán có kích thước mxn(k(c+4)). 
Nếu tâm của một bbox nằm trong ô nào trên grid, ô đó sẽ cần phải được dự đoán là chứa đối tượng. 
Mỗi ô trên grid sẽ được mô hình dự đoán k(c+4) giá trị, trong đó:
- k là số lượng bbox dự đoán.
- 4 là 4 giá trị x, y, w, h đại diện cho bbox được dự đoán.
- c là số lượng lớp đối tượng trong bài toán object detection. Mỗi giá trị dự đoán trong c là giá trị xác suất anchor đó là đối tượng nào.

## 3. Mô hình RetinaNet

RetinaNet là một mô hình single-stage object detection cân bằng giữa độ chính xác của các mô hình two-stage và tốc độ của các mô hình singlestage ở thời điểm đó.

Với ý tưởng khởi tạo anchor như YOLOv1 và SSD, nhóm tác giả của RetinaNet đã chỉ ra vấn đề nghiêm trọng về mất cân bằng dữ liệu trong quá trình train mô hình.
Vấn đề này xảy ra chủ yếu do sự chênh lệch giữa foreground và background, hay nói cách khác là phần chứa đối tượng và phần không chứa đối tượng.

Các mô hình two-stage object detection không thật sự gặp phải vấn đề mất cân bằng dữ liệu này bởi vì trong quá trình đưa các region proposal từ Region proposals module sang Feature extraction module thường đã có một bước lọc và lựa chọn. 
Cụ thể hơn, với số lượng lớn các khu vực không chứa đối tượng được đề xuất bởi Region proposals module, chỉ có một số ít trong đó được lựa chọn để làm đầu vào cho Feature extraction module và lúc này, tỷ lệ giữa các khu vực chứa và không chứa đối tượng thường là 1:3 - một tỷ lệ mất cân bằng không quá nghiêm trọng và không ảnh hưởng tới việc train mô hình object detection.

### 5.2. Hàm Focal loss

Để giải quyết vấn đề mất cân bằng dữ liệu nói trên, nhóm tác giả của RetinaNet đã đề xuất hàm Focal loss dựa trên nền tảng của hàm binary cross entropy loss giải quyết vấn đề mất cân bằng dữ liệu nghiêm trọng.
Nhóm tác giả chú thích rằng hàm Focal loss hiệu quả đối với cả bài toán phân lớp với nhiều hơn hai lớp nhưng để đơn giản hoá, nhóm tác giả sử dụng hàm binary cross entropy loss.

<img src="https://velog.velcdn.com/images/xuio/post/3013e178-bd81-4f89-9f6b-fa194d10f9e5/image.png" style="width: 500px;"/>

### 5.3. Kiến trúc mô hình RetinaNet

RetinaNet là mô hình single-stage object detection gồm có các thành phần:
- Phần backbone Feature Pyramid Networks được sử dụng nhằm trích xuất đặc trưng của ảnh đầu vào với nhiều kích thước đặc trưng khác nhau.
- Phần trích xuất anchor được thực hiện tương tự với cách trích xuất của mô hình RPN.

<img src="https://developers.arcgis.com/python/guide/images/retinanet.png" style="width: 800px;"/>

- Phần Classification Subnet được chia sẻ giữa tất cả các feature maps của backbone FPN, gồm các lớp Conv 3x3xC và lớp Conv cuối cùng 3x3xKA.
Trong đó, 
K là số lượng lớp đối tượng trong bài toán object detection, 
A là số lượng anchor tại vị trí trên mỗi feature maps của backbone FPN (tác giả chọn A = 9), 
C là số lượng channel của lớp Conv (tác giả chọn C = 256).

- Phần Box Regression Subnet được thiết kế khác với cách thiết kế trong mô hình Faster R-CNN khi không dùng chung các lớp Conv với Classification Subnet. 
Box Regression Subnet cũng gồm các lớp Conv 3x3xC và lớp Conv cuối cùng 3x3x4A. 
Trong đó, 
A là số lượng anchor tại vị trí trên mỗi feature maps của backbone FPN (tác giả chọn A = 9), 
4 là 4 độ lệch trong toạ độ của bbox dự đoán so với groundtruth, 
C là số lượng channel của lớp Conv (tác giả chọn C = 256).

### 5.4. Kết luận về mô hình RetinaNet

Mô hình RetinaNet ra đời là một bước tiến lớn đối với việc giải quyết bài toán object detection khi nó giải quyết vấn đề mất cân bằng dữ liệu của các mô hình single-stage giúp tăng độ chính xác của mô hình ngang bằng với các mô hình two-stage nhưng vẫn duy trì được một tốc độ nhanh và có thể sử dụng trong thời gian thực.

Mô hình RetinaNet cho đến nay vẫn là một mô hình tốt để giải quyết các bài toán con của object detection, cụ thể là face detection. 
Trong các phần tiếp theo của luận văn, ta sẽ bàn luận về các mô hình kế thừa RetinaNet giải quyết rất tốt bài toán face detection.


