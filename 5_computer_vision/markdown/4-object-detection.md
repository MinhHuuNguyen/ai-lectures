---
time: 05/10/2022
title: Bài toán object detection
description: Object detection là bài toán kết hợp giữa hai bài toán object localization và object classification. Trong đó, mô hình giải quyết bài toán object detection đầu tiên sẽ định vị vị trí có thể chứa đối tượng trong ảnh (object localization), sau đó mô hình sẽ thực hiện phân lớp đối tượng để nhận diện đối tượng đó là đối tượng nào (object classification). Object detection là bài toán cực kỳ quan trọng và có nhiều ứng dụng trong lĩnh vực Computer vision.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về object detection

Object detection là một bài toán rất phổ biến trong computer vision và được coi là một trong số các bài toán machine learning kinh điển.
Object detection một nhiệm vụ quan trọng trong thị giác máy tính, nhằm xác định vị trí (bounding box) và nhãn (label) của các đối tượng trong ảnh hoặc video (đối với video, ta có bài toán object tracking - theo dõi đối tượng trong video).
Khác với bài toán phân loại ảnh chỉ ra nhãn chung của ảnh, phát hiện đối tượng đòi hỏi dự đoán đồng thời tọa độ và loại của từng đối tượng. 

Tính ứng dụng của Object detection trong thực tiễn là rất lớn trong nhiều ngành nghề khác nhau.
Một số ứng dụng có thể kể đến như:
- **Xe tự hành:** Phát hiện các xe khác, người đi bộ, đèn giao thông, biển báo để đưa ra quyết định lái xe an toàn.
- **Giám sát an ninh:** Tự động phát hiện người xâm nhập, hành vi đáng ngờ hoặc vật thể bị bỏ lại.
- **Bán lẻ thông minh:** Phân tích hành vi khách hàng, quản lý hàng tồn kho trên kệ, thanh toán tự động.
- **Y tế:** Phát hiện các khối u, tế bào bất thường trong ảnh chụp X-quang, MRI.
- **Nông nghiệp chính xác:** Nhận diện sâu bệnh, cỏ dại hoặc theo dõi sức khỏe cây trồng qua hình ảnh từ drone.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/object_detection.jpeg" style="width: 1000px;"/>

Bài toán object detection là sự tổng hợp của hai bài toán con: object localization và object classification.
- **Object localization**: là bài toán định vị vị trí của object trong ảnh, nhận đầu vào là một ảnh và trả đầu ra là một hoặc nhiều bbox của từng đối tượng.
- **Object classification**: là bài toán phân lớp ảnh, nhận đầu vào là một ảnh và trả đầu ra là lớp của đối tượng đó.

Object detection kết hợp cả hai bài toán trên, yêu cầu mô hình vừa định vị vị trí của một hoặc nhiều đối tượng trong ảnh vừa xác định lớp của từng đối tượng đó.

## 2. Một số khái niệm trong object detection

### 2.1. Bounding box - Label - Confidence score

Bounding box (bbox) là một hình chữ nhật được vẽ xung quanh vật thể được phát hiện giúp xác định vị trí của vật thể.

Trong đa số các trường hợp, bbox thường là hình chữ nhật đứng với chiều dài và chiều rộng song song với hai cạnh của ảnh.
Trong một số trường hợp đặc biệt, nhằm mục đích giúp xác định vị trí của đối tượng một cách chính xác hơn nữa, bbox có thể là hình chữ nhật xoay với chiều dài và chiều rộng KHÔNG song song với hai cạnh của ảnh.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/straight_vs_rotated_bbox.jpeg" style="width: 800px;"/>

Đối với bbox thẳng, một bbox thường được xác định bởi 4 giá trị.
4 giá trị này sẽ có ý nghĩa khác nhau theo từng format của bộ dữ liệu:
- $x_1, y_1, x_2, y_2$ (hoặc có thể gọi là $x_{min}, y_{min}, x_{max}, y_{max}$): gồm $x_1, y_1$ và $x_2, y_2$ lần lượt là toạ độ 2 điểm góc trái trên và góc phải dưới.
- $x, y, w, h$: gồm $x, y$ là toạ độ góc trái trên và $w, h$ là chiều rộng và chiều cao của bbox.
- $x_{center}, y_{center}, w, h$: gồm $x_{center}, y_{center}$ là toạ độ tâm của hình chữ nhật và $w, h$ là chiều rộng và chiều cao của bbox.
- Ngoài các format trên, bbox còn có thể được xác định bởi một số format khác nữa nhưng ít phổ biến hơn.

Đối với bbox xoay, một bbox có thể được xác định như sau:
- $x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4$: gồm 4 cặp toạ độ $x, y$ đại diện cho 4 điểm góc của bbox.
- $x_{center}, y_{center}, w, h, \theta$: gồm $x_{center}, y_{center}$ là toạ độ tâm của hình chữ nhật, $w, h$ là chiều rộng và chiều cao của bbox và $\theta$ là góc xoay của bbox.

Các toạ độ vị trí của góc $x_1, y_1, x_2, y_2$ hay $x_{center}, y_{center}$ có thể được biểu diễn dưới dạng **absolute value** là vị trí tuyệt đối của pixel trên ảnh hoặc **relative value** là vị trí tương đối được chuẩn hoá về khoảng $[0, 1]$.
Đối với bbox xoay và góc xoay $\theta$, giá trị $\theta$ có thể nằm trong khoảng $[0, 360]$ hoặc $[-180, 180]$.

Các quy ước về format và khoảng giá trị của bounding box cần được kiểm tra chi tiết trong phần mô tả bộ dữ liệu vì với mỗi bộ dữ liệu khác nhau sẽ có cách để quy ước khác nhau.

Trong bài toán object detection, class label là tên của vật thể được phát hiện (ví dụ: "mèo", "ô tô", "người") và confidence score là một giá trị (thường từ 0 đến 1) thể hiện mức độ "tự tin" của mô hình rằng vật thể nó phát hiện là chính xác, tương tự như trong bài toán image classification.

### 2.2. Intersection Over Union (IoU)

Intersection over Union (IoU), hay còn gọi là Jaccard Index, là một thước đo dùng để đánh giá mức độ trùng khớp giữa hai vùng, trong object detection, đó là hai bbox.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/iou.jpeg" style="width: 600px;"/>

Nói một cách đơn giản, IoU cho bạn biết predicted box "chồng" lên ground truth box tốt đến mức nào.
Giá trị của IoU luôn nằm trong khoảng từ 0 đến 1:
- **IoU = 0:** Hai hộp không giao nhau chút nào.
- **IoU = 1:** Hai hộp trùng khớp hoàn toàn.
- **0 < IoU < 1:** Hai hộp có một phần giao nhau. Giá trị càng gần 1 thì mức độ trùng khớp càng cao.

### 2.3. Anchor

Trong một bức ảnh, các vật thể có thể xuất hiện ở bất kỳ đâu, với đủ mọi kích thước và hình dạng:
- **Đa dạng về kích thước (Scale):** Một người đứng gần camera sẽ lớn hơn rất nhiều so với một người đứng ở xa.
- **Đa dạng về tỷ lệ (Aspect Ratio):** Một người đang đứng (cao, hẹp) có tỷ lệ khác với một chiếc xe hơi (thấp, rộng).

Nếu yêu cầu mô hình phải dự đoán tọa độ (x, y, chiều rộng, chiều cao) của một vật thể từ con số không, đây là một bài toán cực kỳ khó. Sẽ có vô số khả năng về vị trí và kích thước mà mô hình phải học.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/anchor.jpeg" style="width: 600px;"/>

Anchor ra đời để giải quyết vấn đề này. Thay vì bắt mô hình "vẽ" một chiếc hộp từ đầu, chúng ta cung cấp cho nó một loạt các "hộp mẫu" (chính là các anchor) ở mọi vị trí trên ảnh.
Anchor (hay Anchor Box) là một tập hợp các hộp chữ nhật (bounding box) được định nghĩa trước với các kích thước và tỷ lệ khung hình (aspect ratio) khác nhau. Chúng được sử dụng như những "khuôn mẫu" hay "dự đoán ban đầu" để giúp mô hình mạng neuron học cách xác định vị trí và kích thước của các vật thể trong một bức ảnh.

Khi có các Anchor được định nghĩa trước, nhiệm vụ của mô hình giờ đây được đơn giản hóa thành hai việc:
- **Phân loại (Classification):** Với mỗi anchor, xác định xem nó có chứa vật thể nào không? Nếu có thì đó là vật thể gì (người, xe, chó, mèo...)?
- **Tinh chỉnh (Regression):** Nếu anchor đó chứa vật thể, hãy điều chỉnh nhẹ tọa độ và kích thước của anchor đó để nó khớp chính xác với vật thể thật.

Việc dự đoán những sự thay đổi nhỏ (offset) so với một hộp tham chiếu có sẵn dễ dàng hơn rất nhiều so với việc dự đoán tọa độ tuyệt đối từ đầu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/anchor_example.jpeg" style="width: 800px;"/>

Quy trình hoạt động của anchor:
- **Bước 1: Tạo ra các anchor:** từ các feature map được trích ra từ mô hình backbone, trên mỗi pixel, ta đặt một tập hợp các anchor box. Ví dụ: tập hợp 9 anchor box mỗi pixel với 3 kích thước khác nhau (small, medium, large) và 3 tỷ lệ khác nhau (1:2, 1:1, 2:1).
- **Bước 2: Gán nhãn cho mỗi anchor:** dựa vào IoU giữa từng anchor và ground truth bounding box, ta gán nhãn cho anchor **có chứa đối tượng** (VD: IoU > 0.7 so với GT bbox), **không chứa đối tượng** (VD: IoU < 0.3 so với GT bbox) và bỏ qua (VD: 0.3 <= IoU <= 0.7 so với GT bbox).
- **Bước 3: Huấn luyện mô hình với các anchor:** với mỗi anchor, mô hình sẽ được huấn luyện để xác định là anchor có chứa đối tượng hay không chứa đối tượng (classification head) và tinh chỉnh toạ độ, kích thước của anchor để tạo ra bounding box bao lấy đối tượng chính xác hơn (regression head).

Nhược điểm của việc sử dụng anchor là **Phụ thuộc vào siêu tham số (Hyperparameters)**.
Việc chọn kích thước, tỷ lệ, và số lượng anchor không tối ưu cho bộ dữ liệu có thể làm giảm hiệu suất của mô hình.
Ví dụ, nếu trong bộ dữ liệu có những vật thể có dạng hình chữ nhật dài và hẹp, việc chọn các tỷ lệ anchor gần vuông sẽ không tối ưu.
Đây chính là động lực cho sự ra đời của một số mô hình không dùng anchor (anchor-free model) như FCOS, CenterNet, CornerNet.

### 2.4. Non-maximum suppression

Ví dụ, khi sử dụng mô hình object detection để tìm các con mèo trong một bức ảnh, thay vì chỉ tìm thấy một bounding box duy nhất cho mỗi con mèo, mô hình thường sẽ đưa ra hàng trăm bbox chồng chéo lên nhau xung quanh cùng một con mèo.
Đây chính là lúc Non-Maximum Suppression (NMS) phát huy tác dụng.

NMS là một thuật toán hậu xử lý (post-processing), được áp dụng sau khi mô hình đã đưa ra các dự đoán.
Nhiệm vụ của NMS là "dọn dẹp" các bbox dư thừa, chỉ giữ lại những hộp có độ tin cậy cao nhất và không bị chồng chéo quá nhiều với nhau.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/nms.jpeg" style="width: 650px;"/>

NMS hoạt động dựa trên Confidence Score và Intersection over Union (IoU).
Các bước của thuật toán NMS:
- **Bước 1: Lọc ban đầu:** Loại bỏ tất cả các bbox có confidence score thấp hơn một ngưỡng nhất định (VD: 0.5) giúp giảm số lượng hộp cần xử lý.
- **Bước 2: Sắp xếp:** Sắp xếp các bbox còn lại theo thứ tự confidence score từ cao xuống thấp.
- **Bước 3: Lặp và triệt tiêu:** Bắt đầu vòng lặp:
    - **Bước 3.1:** Lấy bbox có confidence score cao nhất (gọi là M) và thêm nó vào danh sách kết quả cuối cùng.
    - **Bước 3.2:** Tính toán IoU của M với tất cả các bbox còn lại trong danh sách.
    - **Bước 3.3:** Loại bỏ tất cả các bbox có IoU với M lớn hơn một ngưỡng IoU xác định trước (VD: 0.6).
    Lý do là vì các bbox này có khả năng cao đang chỉ vào cùng một đối tượng với M.
    - **Bước 3.4:** Quay lại bước (a) với các bbox chưa bị loại bỏ, lặp lại quy trình cho đến khi không còn bbox nào trong danh sách.
- **Kết quả:** Danh sách kết quả cuối cùng chứa các bbox đã được "dọn dẹp", mỗi đối tượng chỉ còn lại một bbox đại diện tốt nhất.

Nhược điểm của NMS bộc lộ khi các đối tượng cùng loại ở rất gần nhau.
Ví dụ, trong một đám đông, hai người đứng sát nhau có thể có các bbox chồng chéo nhiều dẫn đến việc NMS có thể sẽ loại bỏ nhầm bbox của một trong hai người vì IoU quá cao.

Để giải quyết hạn chế này, nhiều biến thể của NMS đã ra đời:
- **Soft-NMS:** Thay vì loại bỏ hoàn toàn các bbox có IoU cao, Soft-NMS sẽ giảm điểm tự tin của chúng.
Bbox nào càng chồng chéo nhiều thì điểm tự tin càng bị giảm mạnh.
Cách tiếp cận này "mềm dẻo" hơn và giúp giữ lại các đối tượng ở gần nhau.
- **DIoU-NMS (Distance-IoU NMS):** Một phiên bản cải tiến hơn nữa, không chỉ xét đến IoU mà còn xem xét khoảng cách giữa tâm của hai bbox.
Điều này giúp nó phân biệt tốt hơn giữa một bbox "tệ" và một bbox của đối tượng khác đang ở gần.

## 3. Nhóm các phương pháp giải bài toán object detection

### 3.1. Nhóm các phương pháp truyền thống (trước Deep Learning)

Các phương pháp này dựa trên việc trích xuất đặc trưng thủ công.
Ta cần phải tự định nghĩa các đặc trưng (như cạnh, góc, màu sắc) để máy tính có thể nhận diện.

Một số mô hình phổ biến trong nhóm này như: Viola-Jones Framework, HOG (Histogram of Oriented Gradients), DPM (Deformable Part-based Models).

### 3.2. Nhóm các phương pháp dựa trên Deep Learning

#### 3.2.1. Nhóm mô hình two-stage

Các mô hình thuộc nhóm two-stage ra đời khá sớm từ năm 2014 đến 2017, với các cái tên nổi tiếng là R-CNN, Fast R-CNN, Faster R-CNN.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/5-two-stage-od-models/banner.jpeg" style="width: 800px;"/>

Nhóm này có đặc điểm chung về kiến trúc gồm hai phần:
- **Region proposals module**: module này nhận đầu vào là ảnh đầu vào cần thực thi object detection và trả đầu ra là các khu vực trên ảnh mà **có khả năng** chứa đối tượng.
- **Feature extraction module**: module này nhận đầu vào là **các khu vực mà Region proposals module trả ra** và trả đầu ra giúp xác định chính xác đối tượng trong khu vực đó là đối tượng nào và tinh chỉnh toạ độ của khu vực để đưa ra lời dự đoán chính xác hơn.

Ưu điểm của Nhóm mô hình two-stage object detection là độ chính xác rất cao, tuy nhiên, nhược điểm là tương đối chậm, không phù hợp cho các ứng dụng thời gian thực.

#### 3.2.2. Nhóm mô hình single-stage

Các mô hình thuộc nhóm single-stage ra đời muộn hơn từ năm 2016 đến nay, với các cái tên nổi tiếng là SSD, chuỗi các phiên bản của YOLO, RetinaNet.
Hiện nay, các mô hình thuộc nhóm single-stage đang nhận được sự quan tâm rất lớn của giới nghiên cứu trong thời gian trở lại đây vì tính ứng dụng trong thực tiễn cao của chúng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/6-single-stage-od-models/banner.jpeg" style="width: 800px;"/>

Các mô hình single-stage đều dựa vào động lực trong việc loại bỏ Region proposals module nhằm giảm khối lượng tính toán, qua đó tăng tốc độ và đưa mô hình đến gần hơn với khả năng chạy real-time.

Ưu điểm của Nhóm mô hình single-stage object detection là tốc độ rất nhanh, phù hợp cho video và các ứng dụng real-time, nhưng nhược điểm là độ chính xác thường thấp hơn một chút so với các mô hình two-stage, tuy nhiên khoảng cách này đang ngày càng được thu hẹp.

Trong thời gian gần đây, với sự bùng nổ của các mô hình computer vision dựa trên kiến trúc Transformer, ta cũng có mô hình DETR (DEtection TRansformers) khá nổi tiếng, giải bài toán object detection với kiến trúc transformer.

## 4. Các metrics trong object detection

Với đặc thù của bài toán object detection là tính toán độ chính xác dựa trên đơn vị là các bbox chứ không phải đơn vị ảnh, các metrics trong object detection cũng khác chút so với image classification.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/metrics.jpeg" style="width: 800px;"/>

### 4.1. TP, FP, FN và TN

Các khái niệm TP, FP, FN và TN trong bài toán classification đã được chia sẻ trong bài viết [Metrics đánh giá cho bài toán classification](/blog/metrics-danh-gia-cho-bai-toan-classification).

Từ khái niệm về IoU, ta có các khái niệm tương ứng trong bài toán object detection:
- **True Positive (TP):** Bbox prediction được gọi là TP nếu IoU giữa bbox groundtruth và bbox prediction $\geq$ ngưỡng IoU (IoU threshold).
- **False Positive (FP):** Bbox prediction được gọi là FP nếu IoU giữa bbox groundtruth và bbox prediction $\lt$ ngưỡng IoU (IoU threshold).
- **False Negative (FN):** Là các bbox groundtruth không được dự đoán
- **True Negative (TN):** Không được sử dụng trong bài toán object detection

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/tp_fp_tn_fn.jpeg" style="width: 1000px;"/>

Đối với các bài toán object detection có nhiều class hơn 2 (class object và class background), bbox prediction được gọi là TP không những phải thoả mãn yêu cầu về IoU mà còn phải thoả mãn yêu cầu về class dự đoán phải đúng.

Trong quá trình đánh giá mô hình object detection, ta thường sử dụng ba ngưỡng IoU là 0.5, 0.75 và 0.9 để xác định TP, FP, FN và tính các metrics.

### 4.2. Precision và Recall

Các khái niệm TP, FP, FN và TN trong bài toán classification đã được chia sẻ trong bài viết [Metrics đánh giá cho bài toán classification](/blog/metrics-danh-gia-cho-bai-toan-classification).

Tương tự như việc đánh giá mô hình classification, precision và recall cũng được sử dụng để đánh giá mô hình object detection.

$$ \text{precision} = \frac{\text{TP}}{\text{TP + FP}} $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_precision.jpeg" style="width: 500px;"/>

$$ \text{recall} = \frac{\text{TP}}{\text{TP + FN}} $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_recall.jpeg" style="width: 500px;"/>

### 4.3. Precision x Recall curve

Trong thực tế, Precision và Recall thường có mối quan hệ nghịch đảo và không dễ dàng tối ưu cả hai cùng một lúc.
- **Muốn tăng Precision:** Cần dùng một threshold rất cao, ví dụ, chỉ chấp nhận một bbox là đối tượng nếu mô hình có độ tự tin đạt tới 99%.
Điều này làm giảm số lượng FP (giảm đoán nhầm bbox tại những vị trí background), nhưng sẽ làm tăng số lượng FN (bỏ sót các đối tượng mà mô hình dự đoán với độ tự tin thấp hơn chút).
- **Muốn tăng Recall:** Cần dùng một threshold rất thấp, ví dụ, chấp nhận một bbox là đối tượng nếu mô hình có độ tự tin chỉ cần thấp từ 30%.
Điều này giúp giảm FN (bắt được gần như tất cả các đối tượng), nhưng sẽ tăng FP (bắt nhầm nhiều khu vực background là đối tượng).

Sự đánh đổi này chính là lý do tại sao chúng ta cần Precision x Recall curve.

Precision x Recall curve là một đồ thị biểu diễn mối quan hệ giữa Precision (trục Y) và Recall (trục X) tại các threshold khác nhau.
Precision x Recall curve là một biểu đồ giúp đánh giá tương quan về đánh đổi giữa precision và recall.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/precision_recall_curve.jpeg" style="width: 900px;"/>

Với mỗi giá trị ngưỡng, chúng ta sẽ tính một cặp giá trị Precision và Recall tương ứng.
Vẽ tất cả các cặp điểm này lên đồ thị với Recall là trục hoành (X) và Precision là trục tung (Y), sau đó nối các điểm lại, ta sẽ có được đường cong PR.

- **Mô hình lý tưởng:** đường cong đi thẳng từ điểm $(0,1)$ lên điểm $(1,1)$ rồi đi ngang sang.
Điều này có nghĩa là mô hình có thể đạt được cả Precision = 1 và Recall = 1 nhưng trong thực tế, điều này gần như không thể.
- **Mô hình tốt:** đường cong càng gần góc trên bên phải của đồ thị điểm $(1,1)$ càng tốt.
Điều này cho thấy mô hình có thể duy trì Precision cao ngay cả khi Recall tăng lên.
- **Mô hình kém:** Đường cong nằm gần phía dưới.

### 4.4. Area under Curve (AuC), Average Precision (AP) và Mean Average Precision (mAP)

Average Precision (gọi tắt là AP) là metrics phổ biến nhất dùng đánh giá mô hình object detection, được tính bằng chỉ số Area under Curve (AuC) của Precision x Recall curve.
Thay vì quan sát hai Precision x Recall curve xem đường nào tốt hơn thì ta tính toán ra phần diện tích nằm phía dưới Precision x Recall curve để có thể dễ dàng so sánh kết quả của hai mô hình.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/auc.jpeg" style="width: 600px;"/>

Trong thực tế, tên gọi **Average Precision** có nghĩa là tính **Average** các giá trị **Precision** trên khoảng **Recall từ 0 đến 1**.

Ngoài ra, đối với những mô hình object detection với nhiều class object khác nhau, ta có chỉ số Mean Average Precision (mAP) được tính bằng trung bình chỉ số AP trên tất cả các class object.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/ap_vs_map.jpeg" style="width: 700px;"/>

### 4.5. Tóm tắt cách tính AP

- **Bước 1:** Ta có danh sách các bbox prediction với toạ độ và độ tự tin tương ứng.
- **Bước 2:** Với IoU $\geq$ threshold nào đó, ta xác định được mỗi bbox predictionlà các bbox TP hay FN.
- **Bước 3:** Sau khi xác định các bbox prediction là TP hay FN, ta sắp xếp chúng theo thứ tự về confidence để tính toán Precision và Recall.
- **Bước 4:** Với mỗi cặp giá trị Precision và Recall vừa tính toán được, ta thu được Precision x Recall curve.
- **Bước 5:** Ta xấp xỉ Precision x Recall curve trên bằng một đường gọi là Interpolated Precision và ta có hai cách để tính đường interpolated precision.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/ap_example.jpeg" style="width: 1000px;"/>

#### Cách 1:  11-point interpolation

Cách đầu tiên được gọi là 11-point interpolation, ta chia trục recall của Precision x Recall curve thành 10 phần với 11 mốc recall (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1).
Với mỗi mốc recall, ta lấy chỉ số precision cao nhất mà nhận chỉ số recall lớn hơn hoặc bằng mốc recall đang xét.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/11_point_interpolation.jpeg" style="width: 800px;"/>

##### Cách 2: All-point interpolation

Cách thứ hai được gọi là all-point interpolation, ta tính interpolated precision với chính xác các mốc recall của Precision x Recall curve.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/all_point_interpolation.jpeg" style="width: 800px;"/>

Hai cách tính chỉ số AP khác nhau là 11-point interpolation và All-point interpolation cho ta hai giá trị AP khác nhau một chút nhưng ta vẫn có thể sử dụng một trong hai cách này để tính toán và so sánh kết quả của các mô hình object detection.

## 5. Các kỹ thuật data augmentation cho object detection

Giống như các bài toán computer vision nói riêng và deep learning nói chung khác, các mô hình giải quyết bài toán object detection cũng được hưởng lợi rất nhiều từ các kỹ thuật image data augmentation, giúp mô hình tăng khả năng khái quát và giảm thiểu triệt để hiện tượng overfit.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/augmentations.jpeg" style="width: 600px;"/>

### 5.1. Mix Up:

Mục tiêu của Mix Up là tạo ra các hình ảnh mới bằng cách kết hợp thông tin từ hai hình ảnh gốc.

Quá trình Mix Up gồm các bước:
- Chọn ngẫu nhiên hai hình ảnh gốc và một hệ số $\alpha$ thuộc khoảng $[0, 1]$.
- Kết hợp hai hình ảnh này bằng cách tính trung bình có trọng số của các pixel từ hai hình ảnh theo công thức: $new\_img = \alpha * img\_1 + (1 - \alpha) * img\_2$.
- Tính trung bình có trọng số tương ứng cho nhãn của hai hình ảnh theo công thức: $new\_label\_1 = \alpha * label\_1$ và $new\_label\_2 = (1 - \alpha) * label\_2$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/mix_up.jpeg" style="width: 700px;"/>

### 5.2. Cut Out:

Mục tiêu của Cut Out là ẩn đi một phần ngẫu nhiên của hình ảnh bằng một hình chữ nhật đen.

Quá trình Cut Out gồm các bước:
- Chọn ngẫu nhiên một vùng hình vuông có kích thước và vị trí ngẫu nhiên trên hình ảnh.
- Thay vùng này bằng một hình vuông đen.

- Cut Out tạo ra một hiệu ứng giúp mô hình phải học cách xử lý đối tượng khi một phần của nó bị ẩn đi.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/cut_out.jpeg" style="width: 600px;"/>

Một ứng dụng khá hay của CutOut là được sử dụng trong bài toán nhận diện khuôn mặt với khẩu trang hoặc kính râm.
Việc che đi nửa khuôn mặt giúp mô hình học được cách nhận diện khuôn mặt đã bị che, từ đó tạo tiền đề cho việc học các khuôn mặt đeo khẩu trang hoặc đeo kính râm.

### 5.3. Cut Mix:

Cut Mix là một biến thể phức tạp hơn của Cut Out, nó kết hợp hai hình ảnh lại với nhau để tạo ra một hình ảnh mới.

Quá trình Cut Mix gồm các bước:
- Chọn ngẫu nhiên một vùng chữ nhật có kích thước và vị trí ngẫu nhiên trên một hình ảnh 1.
- Chọn một hình ảnh 2 và chọn một vùng có kích thước tương tự trên ảnh 2.
- Thay vùng hình vuông trên hình ảnh 1 bằng vùng từ hình ảnh 2.
- Tính trung bình có trọng số của hai nhãn tương ứng với vùng hình vuông.

Cut Mix tạo ra hình ảnh mới chứa thông tin từ cả hai hình ảnh gốc và giảm bớt thông tin về đối tượng ban đầu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/cut_mix.jpeg" style="width: 800px;"/>

## 6. Khó khăn khi huấn luyện mô hình giải bài toán object detection

### 6.1. Vấn đề mất cân bằng dữ liệu

Vấn đề mất cân bằng dữ liệu giữa vật thể (foreground) và nền (background) là một trong những thách thức cốt lõi và phổ biến nhất khi huấn luyện các mô hình phát hiện vật thể (object detection).

Nguyên nhân của vấn đề này xuất phát từ chính bản chất của một bức ảnh: diện tích của nền thường chiếm phần lớn, trong khi các vật thể mà chúng ta cần phát hiện chỉ chiếm một phần nhỏ.
Khi các mô hình, đặc biệt là các mô hình one-stage detectors, tạo ra một lưới dày đặc gồm hàng ngàn hoặc hàng chục ngàn các anchor boxes trên toàn bộ ảnh để dự đoán, đại đa số các hộp này sẽ không chứa bất kỳ vật thể nào và được gán nhãn là "nền".
Điều này tạo ra một sự chênh lệch cực lớn, với tỷ lệ giữa các mẫu nền (negative samples) và mẫu vật thể (positive samples) có thể lên tới 1000:1 hoặc thậm chí cao hơn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/imbalance_data.jpeg" style="width: 700px;"/>

Để giải quyết vấn đề này, nổi bật nhất là Focal Loss, được giới thiệu trong [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002), giúp điều chỉnh hàm cross-entropy tiêu chuẩn bằng cách giảm trọng số của các phần nền dễ phân loại, và tập trung vào việc học từ các phần nền khó và phần đối tượng.

Một phương pháp khác là Online Hard Negative Mining (OHEM) được giới thiệu trong [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/pdf/1604.03540), trong đó mô hình chỉ chọn ra những phần nền khó nhất để đưa vào tính toán loss và cập nhật trọng số.

### 6.2. Vấn đề chênh lệch kích thước đối tượng và kích thước ảnh

Vấn đề chênh lệch kích thước đối tượng và kích thước ảnh được đưa ra bàn luận rất chi tiết trong bài báo [An Analysis of Scale Invariance in Object Detection – SNIP](https://arxiv.org/pdf/1711.08189).

Nhóm tác giả của bài báo này chứng minh rằng CNN là mô hình nhạy cảm với kích thước đối tượng đầu vào.
Cụ thể hơn, nếu ta ép mô hình phải học trong cùng 1 bộ dữ liệu cả những đối tượng có kích thước lớn (tỷ lệ giữa kích thước đối tượng và kích thước ảnh là lớn) và cả những đối tượng có kích thước nhỏ (tỷ lệ giữa kích thước đối tượng và kích thước ảnh là nhỏ) thì kết quả của mô hình sẽ không thật sự tối ưu, đặc biệt đối với những đối tượng có kích thước nhỏ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/scale_problem.jpeg" style="width: 1000px;"/>

Một ví dụ về vấn đề chênh lệch kích thước đối tượng và kích thước ảnh cũng được nhắc đến trong bài báo giới thiệu bộ dữ liệu rất nổi tiếng về nhận diện khuôn mặt [WIDER FACE: A Face Detection Benchmark](https://arxiv.org/pdf/1511.06523).
Trong bộ dữ liệu này, nhóm tác giả đã chia những hình ảnh chứa những khuôn mặt rất nhỏ vào trong một bộ dữ liệu con gọi là **WIDER FACE Test Hard**.
Và sau này, các mô hình khi làm việc trên bộ dữ liệu WIDER FACE đều gặp khó khăn với bộ **WIDER FACE Test Hard** này.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/scale_problem_widerface.jpeg" style="width: 1000px;"/>
