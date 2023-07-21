---
layout: "post"
title:  "Two-stage object detection"
author: "Nguyễn Hữu Minh"
permalink: "/deep-learning/computer-vision/two-stage-object-detection"
parent: "Computer vision"
grand_parent: "Deep learning"
nav_order: 4
---

# Object detection và các mô hình two-stage object detection

## 1. Giới thiệu chung bài toán object detection
Bài toán object detection là một bài toán rất phổ biến trong computer vision và được coi là một trong số các bài toán machine learning kinh điển.

Tính ứng dụng của bài toán object detection trong thực tiễn là rất lớn trong nhiều ngành nghề khác nhau.
Object detection được sử dụng trong y tế giúp xác định vị trí bị bệnh trong cơ thể, trong bảo mật giúp định vị vị trí của con người trong những khu vực cấm, trong nông nghiệp giúp xác định số lượng nông sản, trong hệ thống xe tự hành ...

<img src="https://i.imgur.com/zsla1eq.png" style="width: 700px;"/>

Bài toán object detection là sự tổng hợp của hai bài toán con: object localization và image classification. 
- Object localization là bài toán định vị vị trí của object trong ảnh: nhận đầu vào là một ảnh và trả đầu ra là một hoặc nhiều bbox của từng đối tượng.
- Image classification là bài toán phân lớp ảnh: nhận đầu vào là một ảnh và trả đầu ra là lớp của đối tượng đó.
Bài toán object detection kết hợp cả hai bài toán trên, yêu cầu mô hình vừa định vị vị trí của một hoặc nhiều đối tượng trong ảnh vừa xác định lớp của từng đối tượng đó.

## 2. Khái quát các mô hình giải quyết bài toán object detection

### 2.1. Nhóm các mô hình two-stage

Các mô hình thuộc nhóm two-stage ra đời khá sớm từ năm 2014 đến 2017. Nhóm này có đặc điểm chung về kiến trúc gồm hai phần:
- Region proposals module: module nhận đầu vào là ảnh ban đầu và trả đầu ra là các khu vực trên ảnh mà có khả năng chứa đối tượng.
- Feature extraction module: module nhận đầu vào là các region từ Region proposals module giúp xác định chính xác đối tượng trong khu vực đó là đối tượng nào và tinh chỉnh toạ độ của khu vực chính xác hơn.

<img src="https://www.researchgate.net/publication/353284602/figure/fig3/AS:1046072046673927@1626414419841/Two-stage-vs-one-stage-object-detection-models.ppm" style="width: 700px;"/>

### 2.2. Nhóm các mô hình single-stage

Các mô hình thuộc nhóm single-stage ra đời muộn hơn từ năm 2016 đến nay, tuy nhiên lại đang nhận được sự quan tâm rất lớn của giới nghiên cứu trong thời gian trở lại đây vì tính ứng dụng trong thực tiễn cao của chúng.

Các mô hình single-stage đều dựa vào động lực trong việc loại bỏ Region proposals module nhằm giảm khối lượng tính toán, qua đó tăng tốc độ và đưa mô hình đến gần hơn với khả năng chạy real-time.

## 3. Nhóm các mô hình R-CNN, Fast R-CNN và Faster RCNN

### 3.1. Mô hình R-CNN

Một trong các mô hình đầu tiên ứng dụng deep learning giải quyết bài toán object detection là Regions with CNN features (gọi tắt là R-CNN).

Tuy nhiên, ở thời điểm mà R-CNN ra đời, do các deep learning chưa thật sự phát triển nên R-CNN không hoàn toàn sử dụng deep learning mà vẫn dựa trên kết quả của thuật toán xử lý ảnh như Graph-Based Image Segmentation và Selective Search.

#### 3.1.1. Kiến trúc mô hình R-CNN

Là một trong số các mô hình two-stage, R-CNN bao gồm hai thành phần:
- Region proposals module của R-CNN là thuật toán Selective Search: nhận đầu vào là ảnh, Selective Search trả đầu ra là khoảng 2000 khu vực có khả năng có chứa đối tượng.
- Feature extraction module của R-CNN là một mô hình phân lớp ảnh, cụ thể theo nghiên cứu là AlexNet: nhận đầu vào là ảnh, AlexNet đánh giá xem ảnh đó chứa đối tượng hay không và nếu có thì khu vực đó chứa đối tượng nào.

<img src="https://i.imgur.com/npdeMCI.png" style="width: 700px;"/>

#### 3.1.2. Vấn đề tồn đọng của mô hình R-CNN

Vấn đề lớn nhất của R-CNN là thời gian cần cho quá trình train và quá trình test là rất lớn. 
Trong quá trình test, R-CNN mất tới 47 giây để hoàn thành việc xử lý một ảnh. Kết quả này khiến cho R-CNN gần như không có giá trị ứng dụng thực tiễn.

### 3.2. Mô hình Fast R-CNN

Fast R-CNN là một phiên bản nâng cấp hơn so với R-CNN giúp phần nào giải quyết được một phần điểm yếu về tốc độ.

#### 3.2.1. Kiến trúc mô hình Fast R-CNN

Là một phiên bản nâng cấp của R-CNN, nên Fast R-CNN cũng bao gồm hai thành phần:
- Region proposals module của Fast R-CNN vẫn là thuật toán Selective Search tương tự như R-CNN.
- Feature extraction module của Fast R-CNN là một mô hình phân lớp ảnh, cụ thể là VGG16.
Các thành phần của Fast R-CNN không có thay đổi gì quá nổi bật so với R-CNN, tuy nhiên, điểm khác biệt mang lại giá trị của Fast R-CNN nằm ở cách mà nó kết hợp hai thành phần trên.

<img src="https://uniduc.com/vi/public/uploads/img/images/pon/fast-RCNN.png" style="width: 800px;"/>

Khác với R-CNN, Fast R-CNN đưa toàn bộ ảnh ban đầu qua các lớp conv và pooling của Feature extraction module để tạo ra được đặc trưng của toàn bộ ảnh.

Tiếp theo, với mỗi khu vực mà thuật toán Selective Search đề xuất (regions of interest hay RoIs), Fast R-CNN crop từ đặc trưng của toàn bộ ảnh ra đặc trưng đại diện cho region proposal đó.

Cuối cùng, mỗi đặc trưng đại diện cho mỗi region proposal được đưa qua các lớp fully-connected và trả hai đầu ra gồm giá trị xác suất khu vực đó là đối tượng nào và giá trị độ lệch của bbox.

Tuy nhiên, mỗi region proposal từ thuật toán Selective Search có kích thước khác nhau, nên kích thước của đặc trưng đại diện cho mỗi region proposal cũng khác nhau.
Tuy nhiên, ta lại cần các đặc trưng này có cùng kích thước để có thể đưa vào cùng chung các lớp fully-connected.
Đây là lý do ra đời của lớp RoI pooling

#### 3.2.2. Lớp RoI pooling

Có hai phương pháp pooling phổ biến là maxpooling và average pooling.

<img src="https://www.researchgate.net/profile/Nura-Aljaafari/publication/332092821/figure/fig4/AS:779719519764482@1562911028330/Example-of-max-pooling-and-average-pooling-operations-In-this-example-a-4x4-image-is.jpg" style="width: 400px;"/>


Trong khi đó, RoIs pooling được giới thiệu không hoạt động giống như Max Pooling hay Average Pooling thông thường.

Thay vì yêu cầu ta phải định nghĩa kernel và stride của lớp pooling, RoI pooling yêu cầu ta phải định nghĩa kích thước của đặc trưng đầu ra, từ đó, RoI pooling sẽ tính toán và chia đặc trưng đầu vào thành các vùng trước khi thực hiện phép max pooling.

<img src="https://drive.google.com/uc?id=1VcpGjBDBkqJvXA6h8bQdy0xTShGk_rV9" style="width: 700px;"/>

#### 3.2.3 Vấn đề tồn đọng của mô hình Fast R-CNN

Những kết quả vượt bậc về mặt tốc độ của mô hình Fast R-CNN đã giải quyết được vấn đề tồn đọng của R-CNN trong khi vẫn duy trì được độ chính xác cao.

Tuy nhiên, kiến trúc của mô hình Fast R-CNN vẫn phụ thuộc vào một thuật toán Selective Search và điều này tạo động lực để xây dựng mô hình deep learning thay thế cho các thuật toán này.

### 3.3. Mô hình Faster R-CNN

Mô hình Faster R-CNN được phát triển với trung tâm là kiến trúc mô hình Region Proposal Network (gọi tắt là RPN).

Mô hình RPN được kỳ vọng sẽ thay thế hoàn toàn các thuật toán như Selective Search trong thành phần Region proposals module của các mô hình two-stage.
Việc thay thế các thuật toán bằng một kiến trúc deep learning hướng đến việc cải thiện không chỉ tốc độ của mô hình mà còn cải thiện về độ chính xác.

#### 3.3.1. Kiến trúc mô hình RPN và khái niệm Anchor

Mô hình RPN nhận đầu vào là ảnh với kích thước bất kỳ và trả đầu ra là toạ độ của các khu vực và xác suất khu vực đó là đối tượng nào trong các lớp đối tượng. 
Nhằm tiết kiệm chi phí tính toán, mô hình RPN dùng chung phần Feature extraction module với Fast R-CNN.

<img src="https://uniduc.com/vi/public/uploads/img/images/pon/faster-RCNN.png" style="width: 900px;"/>

Mô hình RPN nhận đầu vào là feature maps từ Feature extraction module và trả đầu ra là các region proposal gọi là các anchor.
Cụ thể:
- RPN đưa feature maps qua một lớp Conv và thu được feature maps mới có kích thước WxH. 
- Với mỗi pixel trên feature maps kích thước WxH, tác giả lấy ra 9 khu vực gọi là 9 anchor.
Từ đó, ta có tổng cộng (WxHx9) anchor.
- Các feature maps đại diện cho các anchor này được tiếp tục đưa qua các lớp Conv để biến đổi về các feature maps mới
    - có dạng (WxHx9)x1 đại diện cho xác suất anchor đó là object
    - có dạng (W xHx9)x4 đại diện cho 4 toạ độ x của góc trái trên, y của góc trái trên, chiều dài và chiều rộng của bbox.

Một điểm mạnh của RPN so với các mô hình object detection thời bấy giờ đó chính là khả năng dự đoán được các object có kích thước khác nhau và tỷ lệ giữa chiều dài và chiều rộng khác nhau nhờ vào cách cấu hình của anchor.

#### 3.3.2. Hàm loss và cách train mô hình RPN

Để train được RPN, nhóm tác giả gán cho mỗi anchor một lớp groundtruth và thiết lập hàm loss đối với từng anchor.
Nhóm tác giả gán lớp groundtruth positive cho anchor dựa theo hai cách sau:
- Những anchor có chỉ số IoU lớn nhất đối với một groundtruth bbox được gán là anchor positive.
- Những anchor có chỉ số IoU lớn hơn 0.7 đối với một groundtruth bbox được gán là anchor positive.

Với hai cách như trên, một groundtruth bbox có thể gán được cho nhiều anchor khác nhau.

Ngoài ra, nhóm tác giả cũng gán lớp groundtruth negative cho các anchor không phải là positive và có chỉ số IoU nhỏ hơn 0.3 đối với một groundtruth bbox.

RPN được thiết kế để có thể train cùng với quá trình train object detection từ đó giúp kết quả đề xuất khu vực trở nên chính xác hơn.


Tuy nhiên, RPN sẽ đề xuất ra nhiều các anchor negative hơn rất nhiều so với số anchor positive từ đó gây mất cân bằng dữ liệu.
Ngoài ra, việc train mô hình với toàn bộ số anchor được đề xuất ra cũng sẽ khiến cho khối lượng tính toán lớn và thời gian kéo dài quá trình train.

Từ đó, nhóm tác giả đề xuất việc lựa chọn ngẫu nhiên 256 anchor trên mỗi ảnh để thực hiện việc tính loss. 
Việc lựa chọn này giúp tỷ lệ anchor positive và negative trở nên cân bằng hơn và giảm thiểu bởi những phần khối lượng tính toán dư thừa.

#### 3.3.3. Sự kết hợp giữa Region Proposal Network và Fast R-CNN

Nhóm tác giả cho rằng, việc train mô hình RPN và Fast R-CNN cần phải diễn ra đồng thời, vì từ đó, việc chia sẻ chung thành phần backbone Conv mới trở nên hiệu quả.

<img src="https://www.researchgate.net/publication/341871095/figure/fig1/AS:902168047521793@1592105032301/Network-structure-diagram-of-Faster-R-CNN-Faster-R-CNN-is-mainly-divided-into-the.ppm" style="width: 500px;"/>

Nhóm tác giả nêu ra ba phương án để train RPN kết hợp với Fast R-CNN:
- Cách 1: Alternating training:
    - Nhóm tác giả train RPN trước sử dụng những hàm loss của RPN nói trên.
    - Sau khi train xong RPN, tác giả sử dụng những khu vực được đề xuất bởi RPN để train Fast R-CNN.
    - Backbone sau khi được train bởi Fast R-CNN tiếp tục được sử dụng để train RPN mới và vòng lặp này tiếp tục diễn ra cho đến khi kết quả hội tụ.

- Cách 2: Approximate joint training:
    - Phương pháp này kết hợp RPN và Fast R-CNN thành một mô hình duy nhất trong quá trình train.
    - Các khu vực được đề xuất bởi RPN được coi như là tất định đối với nhánh Fast RCNN và khiến cho phương pháp train này được gọi là approximate bởi vì những thông tin từ nhánh Fast R-CNN sẽ không được cập nhật cho nhánh RPN.
    - Quá trình backprop được thực hiện độc lập giữa RPN và Fast R-CNN, riêng phần backbone chung được cập nhật theo giá trị hàm loss chung.
    - Phương pháp này đạt hiệu quả thấp hơn chút so với Alternating training tuy nhiên thời gian train được giảm 25 - 50%.

- Cách 3: Non-approximate joint training:
    - Phương pháp này cải thiện được vấn đề tồn đọng của Approximate joint training.
    - Tuy nhiên, để làm được điều này, nhóm tác giả cần tinh chỉnh lại lớp RoI pooling trong Fast R-CNN để có thể update cho cả các thành phần của Fast RCNN và RPN. 

Tóm lại, nhóm tác giả dựa vào phương pháp Alternating training và thực hiện quá trình train gồm 4 bước như sau:
- Bước 1: Nhóm tác giả khởi tạo RPN với pretrained ImageNet và train RPN.
- Bước 2: Nhóm tác giả khởi tạo Fast R-CNN với pretrained ImageNet và train Fast R-CNN với các khu vực được đề xuất bởi RPN.
- Bước 3: Nhóm tác giả khởi tạo lại RPN nhưng sử dụng phần backbone đã được train từ bước 2. Nhóm tác giả chỉ train những lớp riêng của RPN và không cập nhật cho phần backbone.
- Bước 4: Nhóm tác giả finetune lại những lớp riêng của Fast RCNN với các khu vực được đề xuất bởi RPN và thu được Faster R-CNN cuối cùng.

Nhóm tác giả cũng đã lặp lại 4 bước trên vài lần nhưng kết quả không thay đổi quá nhiều.

#### 3.3.4. Vấn đề tồn đọng của mô hình Faster R-CNN

Kết quả của Faster R-CNN và tâm điểm là kiến trúc RPN giúp thay thế thuật toán Selective Search đã giúp cho Faster R-CNN đạt độ chính xác cao hơn so với Fast R-CNN sử dụng Selective Search.

Hơn nữa, RPN giúp cho Faster R-CNN nhanh hơn tới 10 lần so với cấu hình tương tự Fast R-CNN sử dụng Selective Search.

Điều này giúp cho Faster R-CNN cho đến nay vẫn là một mô hình tốt để giải quyết bài toán object detection, vừa đạt độ chính xác cao, vừa có tốc độ tương đối tốt. 

<!-- ## 4. Kiến trúc Feature Pyramid Networks

Feature Pyramid Networks (gọi tắt là FPN) được giới thiệu như một kiến trúc backbone trong các mô hình object detection nhằm giải quyết vấn đề chênh lệch về kích thước giữa các đối tượng trong ảnh.

### 4.1. So sánh các kiến trúc pyramid khác nhau

HÌNH ẢNH 

Ý tưởng về việc xây dựng và sử dụng các đặc trưng của ảnh với nhiều kích thước khác nhau không mới, tuy nhiên, các giải pháp đã có vào th đó đều vướng phải một số vấn đề:
- Featurized image pyramid: Việc sử dụng nhiều kích thước ảnh khác nhau để tạo ra nhiều đặc trưng có kích thước khác nhau một cách độc lập là ý tưởng cơ bản nhất. 
Mặc dù đạt được hiệu quả cao về độ chính xác khi khai thác ảnh đầu vào với nhiều kích thước khác nhau, nhưng phương pháp này khiến cho mô hình giải bài toán object detection trở nên cồng kềnh và tốn rất nhiều thời gian để xử lý và gần như bất khả thi để có thể train được mô hình.

- Single feature map: Việc sử dụng chỉ một kích thước đặc trưng duy nhất giúp cho mô hình xử lý nhanh hơn nhưng lại khiến cho mô hình khó có thể học được những đặc trưng giữa các đối tượng có kích thước chênh lệch trong ảnh. 
Đặc biệt, việc đưa ảnh đầu vào qua nhiều khối Conv đã loại bỏ rất nhiều thông tin và gần như không còn thông tin để mô hình có thể nhận
biết được các đối tượng có kích thước nhỏ.

- Pyramidal feature hierarchy: Việc sử dụng nhiều feature maps có kích thước khác nhau cùng đưa ra kết quả được sử dụng trong mô hình object detection khá nổi tiếng là SSD. 
Tuy nhiên, thay vì tận dụng toàn bộ các feature maps sinh ra từ các khối Conv của backbone VGG-16, SSD chỉ sử dụng feature map từ khối Conv thứ 5 và bổ sung thêm các lớp Conv.
Điều này khiến cho SSD bỏ qua những feature map có kích thước lớn, có ý nghĩa quan trọng trong việc detect các đối tượng có kích thước nhỏ.

- Feature Pyramid Network: Dựa trên vấn đề trên từ SSD, nhóm tác giả đề xuất FPN tận dụng tối đa các feature maps trích xuất được từ backbone nhằm tạo ra bộ feature map mới gồm nhiều kích thước khác nhau và chứa rất nhiều thông tin về nội dung của ảnh đầu vào. 
Để đạt được điều này, nhóm tác giả thiết kế kiến trúc kết hợp những feature maps có kích thước lớn và những feature maps có kích thước nhỏ bằng top-down pathway và lateral connections.

### 4.2. Kiến trúc mô hình Feature Pyramid Networks

Ý tưởng về việc sử dụng kiến trúc mô hình theo dạng top-down không phải là mới và đã được nhắc đến trong một số nghiên cứu. 
Tuy nhiên, điểm giống nhau của các nghiên cứu có thiết kế mô hình theo kiểu top-down đó là mô hình chỉ sử dụng một feature map cuối cùng, sau khi đã tổng hợp các thông tin trong suốt quá trình top-down, để đưa ra quyết định dự đoán cuối cùng.

Trong khi đó, đối với FPN, nhóm tác giả đưa ra quyết định dự đoán trên từng feature maps trong suốt quá trình top-down. 
Từ đó, đặc biệt nâng cao chất lượng của mô hình object detection khi có thể vừa trích xuất được thông tin của các đối tượng có kích thước lớn từ các feature map có kích thước nhỏ vừa trích xuất được thông tin của các đối tượng có kích thước nhỏ từ các feature map có kích thước lớn.

HÌNH ẢNH

Kiến trúc FPN có thể được áp dụng với nhiều backbone Conv khác nhau như AlexNet, VGG hay ResNet, cụ thể trong nghiên cứu, nhóm tác giả lựa chọn ResNet làm mô hình backbone. 

Kiến trúc FPN có thể được chia làm hai phần:
- Bottom-up pathway là quá trình mà ta đưa ảnh đầu vào qua mô hình backbone Conv như ResNet và thu được các feature map. 
Tuy nhiên, trong các mô hình backbone Conv, sẽ có một nhóm các lớp Conv tạo ra các feature map có kích thước giống nhau, và nhóm các lớp Conv này được gọi là một khối Conv. 
Đối với FPN, nhóm tác giả lựa chọn các feature map được sinh ra từ các lớp Conv cuối cùng trong mỗi khối Conv để sử dụng cho nhánh Top-down pathway. 

Cụ thể đối với mô hình backbone ResNet, nhóm tác giả sử dụng các feature maps được sinh ra từ residual block cuối cùng của mỗi khối Conv (trừ khối Conv đầu tiên do kích thước của feature maps này lớn và gây ra vấn đề về bộ nhớ), ký hiệu là C2,C3,C4,C5. Các feature maps này có kích thước lần lượt bằng 1/4, 1/8, 1/16 và 1/32 so với kích thước của ảnh đầu vào.

- Top-down pathway và lateral connections là quá trình mà FPN sinh ra thêm các feature maps mới từ các feature maps của Bottom-up pathway và kết hợp chúng lại thông qua lateral connections. 

Cụ thể, các feature maps của Bottom-up pathway được đưa qua các lớp Conv có kích thước 1x1, stride bằng 1 nhằm giữ nguyên kích thước chiều dài, chiều rộng và chỉ thay đổi kích thước chiều channel của feature maps. 
Các feature maps ở vị trí cao hơn (có kích thước nhỏ hơn) được upsample thông qua thuật toán nearest neighbor và cộng ma trận với feature maps đầu ra từ lớp Conv 1x1 nói trên. 
Cuối cùng, các feature maps đầu ra từ phép cộng ma trận nói trên được đi qua một lớp Conv 3x3 có cùng số đầu ra channel của feature maps nhằm giảm bớt hiệu ứng của thuật toán nearest neighbor và tạo ra các feature maps đầu ra cuối cùng có cùng số channel với nhau. Tập hợp feature maps này được gọi là P2,P3,P4,P5 tương ứng với các feature maps có cùng kích thước C2,C3,C4,C5.

<img src="https://pic2.zhimg.com/80/v2-dacf2d16e42d6bb90596f947ec0044f9_1440w.webp" style="width: 500px;"/>


### 4.3. Vấn đề tồn đọng của kiến trúc mô hình FPN

Kiến trúc FPN ra đời đã tạo ra một trong số những kiến trúc backbone kinh điển trong các bài toán computer vision nói chung và bài toán object detection nói riêng. 
Kiến trúc FPN đã giúp cho nhiều mô hình đạt độ chính xác cao hơn và trong khi tốc độ của mô hình không bị tăng một cách đáng kể. 

Tuy nhiên, đối với cụ thể bài toán object detection, việc kết hợp kiến trúc FPN vào mô hình Faster R-CNN mới chỉ cải thiện về mặt độ chính xác cho mô hình Faster R-CNN mà chưa giúp tăng tốc mô hình Faster R-CNN.
Vẫn còn một câu hỏi cần phải được giải quyết đó là làm sao để duy trì được độ chính xác mà FPN mang lại những mô hình object detection vẫn có để đạt tốc độ nhanh hơn nữa.

## 5. Mô hình RetinaNet

RetinaNet là một mô hình single-stage object detection cân bằng giữa độ chính xác của các mô hình two-stage và tốc độ của các mô hình singlestage ở thời điểm đó. 
Nhóm tác giả của RetinaNet đưa ra vấn đề về các mô hình single-stage như YOLO hay SSD dù đạt tốc độ rất nhanh nhưng lại kém các mô hình two-stage một khoảng rất xa về độ chính xác và đề xuất giải pháp khắc phục vấn đề này.

### 5.1. Tổng quan về các mô hình single-stage object detection

Các mô hình single-stage object detection ở thời điểm đó đa phần đều chỉ sử dụng một backbone CNN kết hợp thêm với các lớp Conv và lớp fully connected để đưa ra dự đoán về lớp của đối tượng trong ảnh và độ lệch của bbox so với groundtruth.

HÌNH ẢNH 

Việc loại bỏ Region proposals module khiến các mô hình single-stage object detection cần phải xây dựng một phương pháp riêng nhằm đề xuất ra các anchor chứa đối tượng. Hai mô hình single-stage object detection nổi tiếng vào thời điểm đó là YOLO và SSD có các cách đề xuất ra anchor tương tự với nhau.

HÌNH ẢNH 

YOLO đề xuất ra các anchor thông qua việc chia ảnh đầu vào thành dạng grid có kích thước SS và với mỗi grid sẽ trả đầu ra dự đoán có kích thước SS(B∗ 5+C). 
Nếu tâm của một bbox nằm trong ô nào trên grid, ô đó sẽ cần phải được dự đoán là chứa đối tượng. 
Mỗi ô trên grid sẽ được mô hình dự đoán (B∗ 5+C) giá trị, trong đó:
- B là số lượng bbox dự đoán.
- 5 là các giá trị trong đó có 4 giá trị x, y, w, h đại diện cho bbox được dự đoán và 1 giá trị confidence. 
Thay vì được học là 1 nếu anchor có IoU cao với groundtruth bbox và ngược lại là 0 nếu anchor có IoU thấp với groundtruth bbox, điểm đặc biệt về giá trị confidence mà nhóm tác giả thiết kế trong mô hình YOLO là nó bằng chính giá trị IoU so với groundtruth.
- C là số lượng lớp đối tượng trong bài toán object detection. Mỗi giá trị dự đoán trong C là giá trị xác suất điều kiện nếu ô trên grid chứa đối tượng thì đó là đối tượng nào.

Trong nghiên cứu, nhóm tác giả của YOLO sử dụng S = 7,B = 2,C = 20.

HÌNH ẢNH 

SSD cũng sử dụng feature maps như là các dạng grid của ảnh đầu vào nhưng thay vì sử dụng một grid như YOLO thì SSD sử dụng nhiều grid từ nhiều feature maps có cách kích thước khác nhau. 
Với mỗi grid tạo bởi một feature maps có kích thước mn, SSD trả đầu ra dự đoán có kích thước mn(k(c+4)). 
Nếu tâm của một bbox nằm trong ô nào trên grid, ô đó sẽ cần phải được dự đoán là chứa đối tượng. 
Mỗi ô trên grid sẽ được mô hình dự đoán (k(c+4)) giá trị, trong đó:
- k là số lượng bbox dự đoán.
- 4 là 4 giá trị x, y, w, h đại diện cho bbox được dự đoán.
- c là số lượng lớp đối tượng trong bài toán object detection. Mỗi giá trị dự đoán trong c là giá trị xác suất anchor đó là đối tượng nào.

Với ý tưởng khởi tạo anchor như trên, nhóm tác giả của RetinaNet đã chỉ ra một vấn đề nghiêm trọng mà các mô hình single stage object detection nói chung gặp phải đó là vấn đề mất cân bằng dữ liệu trong quá trình train mô hình. 
Cụ thể, vấn đề mất cân bằng ở đây xảy ra chủ yếu do sự chênh lệch giữa phần ảnh là foreground và phần ảnh là background, hay nói cách khác là phần ảnh chứa đối tượng và phần ảnh không chứa đối tượng.

Các mô hình two-stage object detection không thật sự gặp phải vấn đề mất cân bằng dữ liệu này bởi vì trong quá trình đưa các region proposal từ Region proposals module sang Feature extraction module thường đã có một bước lọc và lựa chọn. 
Cụ thể hơn, với số lượng lớn các khu vực không chứa đối tượng được đề xuất bởi Region proposals module, chỉ có một số ít trong đó được lựa chọn để làm đầu vào cho Feature extraction module và lúc này, tỷ lệ giữa các khu vực chứa và không chứa đối tượng thường là 1:3 - một tỷ lệ mất cân bằng không quá nghiêm trọng và không ảnh hưởng tới việc train mô hình object detection.

### 5.2. Hàm Focal loss

Để giải quyết vấn đề mất cân bằng dữ liệu nói trên, nhóm tác giả của RetinaNet đã đề xuất hàm Focal loss dựa trên nền tảng của hàm binary cross entropy loss giải quyết vấn đề mất cân bằng dữ liệu nghiêm trọng.
Nhóm tác giả chú thích rằng hàm Focal loss hiệu quả đối với cả bài toán phân lớp với nhiều hơn hai lớp nhưng để đơn giản hoá, nhóm tác giả sử dụng hàm binary cross entropy loss.

CÔNG THỨC 

trong đó:
- y là giá trị groundtruth (0 đối với anchor không chứa object và 1 đối với anchor chứa object)
- p là giá trị xác suất mà mô hình dự đoán anchor đó chứa object

từ đó, hàm cross entropy loss được viết lại thành

CÔNG THỨC 

Một cấu hình khác của hàm cross entropy loss là balanced cross entropy loss, được sinh ra bằng việc đánh trọng số cho từng số hạng của hàm cross entropy loss ban đầu

CÔNG THỨC 

trong đó:
- αt là trọng số tương ứng với số hạng pt. Trọng số αt có thể được tính dựa trên tần suất xuất hiện của các lớp trong bộ dữ liệu hoặc là một hyperpameter

Hàm balanced cross entropy loss có thể đã giúp giảm bớt hiệu ứng mất cân bằng dữ liệu lên trên giá trị hàm loss. Tuy nhiên, việc gán trọng số như hàm balanced cross entropy loss không phân biệt được giữa những mẫu dữ liệu dễ và khó. 
Nhóm tác giả, từ đó, đề xuất hàm Focal loss không những giúp giải quyết vấn đề mất cân bằng dữ liệu mà còn giúp mô hình tập trung vào những mẫu dữ liệu không chứa đối tượng nhưng khó và dễ nhầm lẫn thành chứa đối tượng.

CÔNG THỨC 

trong đó:
- (1− pt) là thành phần đánh giá độ dễ hay khó của mẫu dữ liệu. 
Với những mẫu dễ và mô hình đã được train tốt, giá trị (1− pt) sẽ nhỏ và những mẫu này sẽ gây ít ảnh hưởng trong quá trình train mô hình.
- γ được nhóm tác giả gọi là focusing parameter, dùng để xác định mức độ tập trung của mô hình lên các mẫu dữ liệu không chứa đối tượng. 
Với γ = 0, hàm FL lúc này tương tự với hàm CE. Trong các thí nghiệm của RetinaNet, giá trị γ = 2 là tốt nhất.

HÌNH ẢNH 

Ngoài ra, nhóm tác giả còn đề xuất một dạng khác của hàm FL bằng việc sử dụng thêm một tham số α và trong các thí nghiệm, dạng này cho kết quả tốt hơn một chút so với dạng hàm FL không sử dụng α.

CÔNG THỨC 

### 5.3. Kiến trúc mô hình RetinaNet

RetinaNet là mô hình single-stage object detection gồm có các thành phần:
- Phần backbone Feature Pyramid Networks được sử dụng nhằm trích xuất đặc trưng của ảnh đầu vào với nhiều kích thước đặc trưng khác nhau. 
Chi tiết về sức mạnh của FPN đã được thảo luận ở phần 2.2. Kiến trúc Feature Pyramid Networks. 
- Phần trích xuất anchor được thực hiện tương tự với cách trích xuất của mô hình RPN biến thể đã phân tích ở phần 2.2.
Tuy nhiên, nhóm tác giả đã thử nghiệm và bổ sung thêm các kích thước CÔNG THỨC của anchor để đạt kết quả tốt hơn. Các anchor được gán groundtruth với chiến lược tương tự như trong phần 2.1.3.

Mô hình Faster R-CNN nhưng điều chỉnh một số điểm: 
(1) thay đổi trở thành bài toán multi-class classification (nhóm tác giả của phần 2.1.3 sử dụng bài toán binary classification phân lớp giữa anchor có chứa object và anchor không chứa object) 
và (2) thay đổi threshold IoU để gán nhãn cho từng anchor.

HÌNH ẢNH 

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

 -->
