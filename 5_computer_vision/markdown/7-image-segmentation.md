---
time: 06/04/2022
title: Bài toán image segmentation
description: Phân đoạn ảnh (Image Segmentation) là một trong những nhiệm vụ cốt lõi và quan trọng nhất của lĩnh vực thị giác máy tính. Mục tiêu của nó là phân chia một hình ảnh kỹ thuật số thành nhiều vùng hoặc đối tượng khác nhau. Việc phân nhóm các mô hình phân đoạn ảnh giúp chúng ta hiểu rõ hơn về cách tiếp cận, ưu nhược điểm và ứng dụng của từng loại. Các mô hình này có thể được phân thành hai nhóm chính là Phương pháp truyền thống và Phương pháp dựa trên Deep Learning. Hiện nay, các mô hình Deep Learning chiếm ưu thế tuyệt đối về độ chính xác và hiệu quả trong các bài toán phức tạp.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về image segmentation

Thị giác máy tính (Computer Vision) là một lĩnh vực của trí tuệ nhân tạo, với mục tiêu là dạy cho máy tính cách "nhìn" và "hiểu" thế giới hình ảnh giống như con người.
Khi chúng ta nhìn vào một bức ảnh, chúng ta không chỉ nhận ra có "một chiếc ô tô" **bài toán Image Classification**, hay khoanh vùng "chiếc ô tô nằm ở đây" **bài toán Object Detection**.
Bộ não của chúng ta làm được một điều tinh vi hơn nhiều: chúng ta nhận thức được chính xác hình dạng, đường viền, và ranh giới của chiếc ô tô đó, tách biệt nó hoàn toàn khỏi con đường, vỉa hè, hay bầu trời.
Đây là **bài toán Phân đoạn ảnh - Image Segmentation**.

Image Segmentation là quá trình phân chia một hình ảnh kỹ thuật số thành nhiều vùng hoặc phân đoạn (segments) khác nhau.
Mục tiêu là đơn giản hóa hoặc thay đổi cách biểu diễn của một hình ảnh thành một dạng có ý nghĩa hơn và dễ phân tích hơn.
Về cơ bản, nó là quá trình gán một nhãn cho mỗi pixel trong ảnh, sao cho các pixel có cùng nhãn sẽ có chung một đặc điểm nhất định.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/auto_pilot.jpeg" style="width: 700px;"/>

Tính ứng dụng của Image Segmentation trong thực tiễn là rất lớn trong nhiều ngành nghề khác nhau.
Một số ứng dụng có thể kể đến như:
- **Xe tự hành:** Xe cần hiểu chính xác đâu là làn đường, vạch kẻ đường, người đi bộ, phương tiện khác, biển báo giao thông để đưa ra quyết định an toàn.
- **Sáng tạo nội dung:** Tính năng xóa nền ảnh tự động trong các ứng dụng chỉnh sửa ảnh hoặc trên các trang thương mại điện tử.
- **Bán lẻ:** Ứng dụng "thử đồ ảo" (virtual try-on), phân đoạn cơ thể người để mặc quần áo ảo lên.
- **Y tế:** Phát hiện và tính toán kích thước của khối u trong ảnh MRI/CT scan để hỗ trợ chẩn đoán và lên kế hoạch xạ trị.

### 1.1. Semantic segmentation

**Semantic segmentation** là một dạng của bài toán classification, trong đó, mô hình, thay vì phân lớp trên cả ảnh, sẽ phân lớp từng pixel trên ảnh thuộc lớp nào.
Tất cả các đối tượng thuộc cùng một lớp sẽ được tô cùng một màu (gán cùng một nhãn).
 
Ví dụ, trong một bức ảnh đường phố, tất cả các xe ô tô sẽ được tô màu xanh, tất cả người đi bộ sẽ được tô màu đỏ, và toàn bộ con đường sẽ được tô màu xám.
Mô hình KHÔNG phân biệt được *từng chiếc xe* riêng lẻ, *từng người* riêng lẻ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/semantic_segmentation.jpeg" style="width: 1000px;"/>

### 1.2. Instance segmentation

**Instance segmentation** là một phiên bản cao hơn của semantic segmentation, bên cạnh việc phân lớp các pixel trên ảnh thuộc lớp nào, đối với những pixel thuộc cùng một lớp, mô hình cần phải phân lớp rõ pixel đó thuộc đối tượng nào

Ví dụ, trong một bức ảnh đường phố, không chỉ tất cả các xe ô tô được nhận diện, mà "chiếc xe ô tô số 1" sẽ được tô màu xanh lam, "chiếc xe ô tô số 2" sẽ được tô màu xanh lá, dù cả hai đều thuộc lớp "ô tô".

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/instance_segmentation.jpeg" style="width: 1000px;"/>

### 1.3. Panoptic Segmentation:

**Panoptic Segmentation** là loại hình toàn diện nhất, kết hợp những gì tốt nhất của cả hai loại trên.
Nó vừa gán một nhãn lớp (VD: ô tô, con đường) cho mỗi pixel (giống Semantic), vừa phân biệt các thực thể riêng biệt (VD: chiếc ô tô thứ 1, chiếc ô tô thứ 2) (giống Instance).
Nó cung cấp một cái nhìn tổng thể, toàn cảnh về bức ảnh, phân loại cả những thứ có thể đếm được (như xe cộ, con người) và những thứ không đếm được, mang tính nền (như bầu trời, cỏ, con đường).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/panoptic.jpeg" style="width: 600px;"/>

## 2. Nhóm các phương pháp giải bài toán image segmentation

### 2.1. Nhóm các phương pháp truyền thống (trước Deep Learning)

Các phương pháp này dựa trên việc trích xuất đặc trưng thủ công.
Ta cần phải tự định nghĩa các đặc trưng (như cạnh, góc, màu sắc) để máy tính có thể nhận diện.
- **Thresholding:** Phương pháp đơn giản nhất, phân chia pixel dựa trên giá trị cường độ sáng. Nếu pixel sáng hơn một ngưỡng nào đó, nó thuộc vùng A; ngược lại, thuộc vùng B.
- **Edge-based Segmentation:** Tìm kiếm các đường biên, tức là những nơi có sự thay đổi đột ngột về cường độ sáng, sau đó coi các vùng được bao bọc bởi các đường biên này là các phân đoạn.
- **Region-based Segmentation:** Bắt đầu từ một hoặc nhiều "điểm hạt giống" (seed points) rồi phát triển các vùng ra xung quanh bằng cách gộp các pixel lân cận có tính chất tương đồng như cùng màu sắc, cùng kết cấu.
- **Clustering-based Segmentation:** Sử dụng các thuật toán như K-Means để nhóm các pixel vào các cụm khác nhau dựa trên các đặc trưng của chúng như màu sắc, vị trí.

Các phương pháp này đều gặp phải những khó khăn và cho ra kết quả không tốt khi phải xử lý những trường hợp dữ liệu đa dạng và nhiều đối tượng.

### 2.2. Nhóm các phương pháp dựa trên Deep Learning

Sự ra đời của các bộ dữ liệu có nhãn phục vụ cho segmentation như PASCAL VOC, COCO, ADE20K, Cityscapes, CamVid... tạo ra đề tảng cho sự phát triển của các mô hình deep learning.
Ngoài ra, các mô hình image segmentation cũng được kế thừa nhiều từ các ý tưởng xây dựng của các mô hình object detection.

#### 2.2.1. Nhóm các mô hình dựa trên CNN

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/fcn.jpeg" style="width: 700px;"/>

**Fully Convolutional Networks (FCN)** được giới thiệu bởi bài nghiên cứu [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038) và **U-Net** được giới thiệu bởi bài nghiên cứu [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597) là hai trong số các mô hình image segmentation phổ biến dựa trên mô hình CNN.

FCN thay thế các fully connected layers ở cuối các CNN phân loại thông thường bằng các lớp convolution.
Điều này cho phép mô hình xuất ra một heatmap có cùng kích thước với ảnh đầu vào, trong đó mỗi pixel chứa thông tin về lớp của nó.

U-Net là một kiến trúc cực kỳ nổi tiếng, đặc biệt trong lĩnh vực y tế có kiến trúc hình chữ U gồm:
- **Encoder:** Giảm dần kích thước không gian của ảnh để nắm bắt thông tin ngữ cảnh.
- **Decoder:** Tăng dần kích thước trở lại để xác định vị trí chính xác của đối tượng.
- **Skip Connections:** Điểm đột phá của U-Net, giúp kết hợp thông tin chi tiết từ nhánh mã hóa vào nhánh giải mã, giúp cho kết quả phân đoạn có đường biên sắc nét và chính xác hơn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/unet.jpeg" style="width: 700px;"/>

**Mask R-CNN** được giới thiệu bởi bài nghiên cứu [Mask R-CNN](https://arxiv.org/pdf/1703.06870) là kiến trúc tiêu chuẩn cho Instance Segmentation.
Mask R-CNN hoạt động theo hai giai đoạn:
- Giai đoạn 1: Hoạt động như một mô hình object detection (giống Faster R-CNN) để đề xuất các vùng chứa đối tượng (bounding boxes).
- Giai đoạn 2: Với mỗi vùng được đề xuất, nó sử dụng kỹ thuật RoI Align và chạy một mạng convolution nhỏ để tạo ra một mask phân đoạn cho đối tượng bên trong vùng đó.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/mask_rcnn.jpeg" style="width: 700px;"/>

#### 2.2.2. Nhóm các mô hình dựa trên Transformer và Open-vocabulary segmentation

Với kiến trúc backbone Vision Transformer đã chứng minh được sự hiệu quả khi làm việc với những bộ dữ liệu hình ảnh lớn so sánh với các kiến trúc backbone CNN, các mô hình image segmentation được xây dựng với nền tảng transformer đang rất đáng chú ý ở thời điểm hiện tại.

Chi tiết hơn về kiến trúc Transformer và mô hình Vision Transformer đã được mình viết trong [bài viết này](/blog/mo-hinh-transformer).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/setr.jpeg" style="width: 500px;"/>

Mô hình SETR từ bài báo [Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers](https://arxiv.org/pdf/2012.15840), mô hình Swin Transformer từ bài báo [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030) và mô hình SegFormer từ bài báo [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/pdf/2105.15203) là ba trong số những mô hình image segmentation đầu tiên và nổi tiếng dựa vào kiến trúc Transformer.
- **SETR (SEgmentation TRansformer):** Sử dụng một ViT làm encoder để trích xuất đặc trưng và một decoder đơn giản để tạo ra segmentation map từ các đặc trưng đó.
- **Swin Transformer:** Giới thiệu hai khái niệm chính là Shifted Windows - giúp tính toán self-attention trong các cửa sổ cục bộ và Hierarchical Architecture - giúp tạo ra các feature maps ở nhiều độ phân giải khác nhau, tương tự như cách CNN hoạt động.
- **SegFormer:** Kết hợp một encoder Transformer với hierarchical architecture (tương tự Swin) với một decoder cực kỳ nhẹ chỉ bao gồm các MLP giúp vừa mạnh mẽ trong việc nắm bắt bối cảnh vừa hiệu quả về mặt tính toán.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/swin_transformer.jpeg" style="width: 500px;"/>

**Open-vocabulary segmentation** (hay **Zero-shot segmentation**) cho phép mô hình phân vùng các đối tượng dựa trên mô tả văn bản tự do ngay tại thời điểm dự đoán, mà không cần huấn luyện lại.
Nền tảng của phương pháp này là các Vision-Language Models - VLMs được huấn luyện trên quy mô lớn, với CLIP (Contrastive Language-Image Pre-training) của OpenAI là ví dụ nổi tiếng nhất.
Chi tiết hơn về CLIP và một số biến thể nâng cấp đã được mình viết trong [bài viết này](/blog/transfer-learning-weakly-semi-un-va-self-supervised-learning).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/sam.jpeg" style="width: 1000px;"/>

Segment Anything được giới thiệu trong bài báo [Segment Anything](https://arxiv.org/pdf/2304.02643) là một trong số những mô hình open-vocabulary image segmentation rất nổi tiếng.

Điểm khác biệt lớn nhất của SAM so với các mô hình trước đây là nó hoạt động dựa trên prompt.
Thay vì được huấn luyện để nhận dạng một danh sách các đối tượng cố định (VD: chỉ nhận dạng được "chó", "mèo", "xe hơi"), SAM có thể phân vùng bất kỳ đối tượng nào trong bất kỳ bức ảnh nào, ngay cả khi nó chưa từng thấy đối tượng đó trước đây.

Sức mạnh của SAM đến từ việc nó được huấn luyện trên một tập dữ liệu khổng lồ chưa từng có tên là SA-1B, bao gồm 1.1 tỷ mask chất lượng cao từ 11 triệu bức ảnh.
Điều này giúp nó học được một khái niệm rất tổng quát về "đối tượng là gì", cho phép nó phân vùng mọi thứ một cách chính xác.

Với sự ra đời của những mô hình rất phát triển dựa trên kiến trúc transformer cho bài toán image segmentation, các mô hình giải bài toán object detection cũng được hưởng lợi theo rất nhiều vì ta có thể hiểu: Khi đã giải được bài toán image segmentation thì sẽ giải được bài toán object detection.

#### 2.3. Nhóm các mô hình self-supervised learning

Một trong số những mô hình đầu tiên và nổi bật trong nhóm này là mô hình DINO được giới thiệu trong bài báo [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) sử dụng kiến trúc **Teacher - Student**.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-un-self-supervised-learning/self_dino_teacher_student.jpeg" style="width: 600px;"/>

Chi tiết hơn về mô hình DINO đã được mình viết trong [bài viết này](/blog/transfer-learning-weakly-semi-un-va-self-supervised-learning).
Sự ra đời của DINO đã tạo ra một làn sóng các nghiên cứu mới về nhóm các mô hình self-supervised learning giải quyết bài toán image segmentation.
Nhóm các mô hình này đã giúp phần nào tiết kiệm một lượng lớn chi phí chuẩn bị dữ liệu có nhãn trong bài toán image segmentation.

## 3. Các metrics trong image segmentation

Đối với bài toán image classification, ta tính toán các chỉ số độ chính xác dựa vào đơn vị từng ảnh.
Đối với bài toán object detection, ta tính toán các chỉ số độ chính xác dựa vào đơn vị từng bbox.
Còn đối với bài toán image segmentation, với mục tiêu phân lớp từng pixel trong ảnh thuộc lớp nào, ta sẽ tính toán các chỉ số độ chính xác dựa vào đơn vị từng pixel trên ảnh.

### 3.1. Các khái niệm TP, FP, FN, TN và Pixel Accuracy

Các khái niệm TP, FP, FN và TN trong bài toán classification đã được chia sẻ trong bài viết [Metrics đánh giá cho bài toán classification](/blog/metrics-danh-gia-cho-bai-toan-classification).

Từ đó, ta có các khái niệm tương ứng trong bài toán image segmentation:
- **True Positive (TP):** Pixel được dự đoán là lớp A và thực tế cũng là lớp A.
- **True Negative (TN):** Pixel được dự đoán không phải lớp A và thực tế cũng không phải lớp A.
- **False Positive (FP):** Pixel được dự đoán là lớp A nhưng thực tế không phải lớp A.
- **False Negative (FN):**  Pixel được dự đoán không phải lớp A nhưng thực tế lại là lớp A.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/pixel_accuracy.jpeg" style="width: 600px;"/>

Từ các khái niệm TP, FP, FN, TN trên, ta có công thức tính Pixel Accuracy như sau:
$$\text{pixel accuracy} = \frac{\text{number of true pixel predictions}}{\text{number of pixels}} = \frac{\text{TP + TN}}{\text{TP + FP + FN + TN}}$$

Pixel Accuracy đơn giản, dễ hiểu và có thể mang lại đánh giá tổng quát về chất lượng của mô hình nhưng nó lại cực kỳ nhạy cảm với sự mất cân bằng lớp (Class Imbalance).
Đây là nhược điểm lớn nhất và khiến Pixel Accuracy có thể trở nên vô nghĩa trong nhiều trường hợp.

### 3.2. Intersection Over Union IoU và mIoU

IoU là khái niệm đã được nhắc đến trong bài viết giới thiệu về [Bài toán object detection](/blog/bai-toan-object-detection).
Công thức và ý nghĩa của IoU dành cho bài toán image segmentation không có gì khác biệt so với trong bài toán object detection.

Đặc biệt hơn, trong bài toán image segmentation, IoU được sử dụng trực tiếp như một metrics đánh giá độ chính xác của mô hình.
Việc sử dụng IoU giúp quá trình đánh giá mô hình image segmentation trở nên chính xác và loại bỏ nhạy cảm với class imbalance.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/iou.jpeg" style="width: 600px;"/>

Ví dụ:
- **Diện tích Intersection:** là số pixel được dự đoán là "ô tô" và thực tế cũng là "ô tô" (True Positive (TP)).
- **Diện tích Union:** là tổng số pixel của:
    - Số pixel nằm trong phần **Diện tích Intersection**
    - Số pixel được dự đoán là "ô tô" nhưng thực tế là nền (background) hoặc một lớp khác (False Positive (FP)).
    - Số pixel thực tế là "ô tô" nhưng mô hình lại dự đoán là nền hoặc lớp khác (False Negative (FN)).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/miou.jpeg" style="width: 800px;"/>

**IoU** là một độ đo tuyệt vời, nhưng nó chỉ tính cho một lớp đối tượng duy nhất.
Trong thực tế, một bài toán segmentation thường có nhiều lớp (ví dụ: ô tô, người, đường, cây cối, v.v.).
Chúng ta cần một độ đo tổng thể để đánh giá hiệu suất của mô hình trên tất cả các lớp.
Đó là lúc **mIoU** ra đời.

**mIoU (Mean IoU) đơn giản là giá trị trung bình cộng của IoU trên tất cả các lớp có trong tập dữ liệu.**
Để tính mIoU, trước tiên, ta sẽ tính IoU cho mỗi lớp xuất hiện trong bộ dữ liệu, sau đó, ta tính trung bình cộng các giá trị IoU đã tính được.

mIoU cung cấp một con số duy nhất để tóm tắt hiệu suất của mô hình trên toàn bộ các lớp. Điều này giúp so sánh các mô hình khác nhau một cách công bằng.

### 3.3. Dice coefficient

Hệ số Dice (Dice Coefficient), còn được gọi là Dice Similarity Coefficient (DSC) hoặc F1-Score, là một độ đo thống kê dùng để đánh giá mức độ tương đồng giữa hai tập hợp, ở đây cụ thể là tập hợp các pixel được gán nhãn thuộc một lớp nào đó và tập hợp các pixel được mô hình dự đoán thuộc lớp đó.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/dice_vs_iou.jpeg" style="width: 800px;"/>

So sánh với IoU với công thức,

$$ \text{IoU} = \frac{\text{TP}}{\text{TP + FP + FN}} $$

Dice có công thức

$$ \text{Dice Coefficient} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}} $$

Ta có thể thấy mối quan hệ giữa $dice$ và $iou$

$$ \text{Dice Coefficient} = \frac{2 \times \text{IoU}}{\text{IoU} + 1} $$

Việc đánh 2x trọng số các TP trong công thức của Dice mang lại cho các pixel được dự đoán đúng một trọng số cao hơn so với các pixel lỗi (FP và FN).
Điều này làm cho Dice "tha thứ" hơn cho các lỗi so với IoU.

Ví dụ: Với TP = 50, FP = 25, FN = 25 thì IoU = 0.5 và Dice = 0.67.

Trong thực tế, Dice Coefficient có thể được dùng làm hàm loss để huấn luyện mô hình image segmentation (Dice Loss).

$$ \text{Dice Loss} = 1 - \text{Dice Coefficient} $$

### 3.4. Precision x Recall curve, Area under Curve (AuC), Average Precision (AP) và Mean Average Precision (mAP)

Precision x Recall curve, Area under Curve (AuC), Average Precision (AP) và Mean Average Precision (mAP) là các khái niệm đã được nhắc đến rất chi tiết trong bài viết giới thiệu về [Bài toán object detection](/blog/bai-toan-object-detection).
Công thức và ý nghĩa của chúng dành cho bài toán image segmentation không có gì khác biệt so với trong bài toán object detection.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/auc.jpeg" style="width: 600px;"/>

Các bước để tính giá trị AP và mAP đối với bài toán image segmentation như sau:
- **Bước 1:** Từ groundtruth label và prediction và danh sách các confidence threshold được định trước, ta có số lượng các pixel tương ứng là TP, FP, FN, TN.
- **Bước 2:** Ta tính toán Precision và Recall với từng giá trị confidence threshold.
- **Bước 3:** Với mỗi cặp giá trị Precision và Recall vừa tính toán được, ta thu được Precision x Recall curve.
- **Bước 4:** Với Precision x Recall curve, ta tính Area under Curve để ra được chỉ số AP.
- **Bước 5:** Ta tính toán giá trị AP trên mỗi class và lấy trung bình giá trị này để thu được giá trị mAP trên toàn bộ bộ dữ liệu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/4-object-detection/ap_vs_map.jpeg" style="width: 700px;"/>

### 3.5. Panoptic Quality (PQ)

Panoptic Segmentation ra đời để hợp nhất hai bài toán trên: vừa phân loại tất cả các pixel (như Semantic), vừa phân biệt các đối tượng riêng lẻ (như Instance).

Panoptic Quality (PQ) là một thước đo đánh giá hiệu suất của các thuật toán Panoptic Segmentation.
Nó đánh giá đồng thời cả chất lượng phát hiện đối tượng và chất lượng phân đoạn đối tượng và ta có thể hiểu Panoptic Quality đánh giá kết hợp cả khả năng "object detection" và khả năng "image segmentation" của mô hình.
PQ được định nghĩa bằng một công thức đơn giản nhưng rất hiệu quả:

$$ \text{PQ} = \text{SQ} \times \text{RQ} $$

trong đó:
- **Segmentation Quality (SQ):** Chất lượng Phân đoạn.
- **Recognition Quality (RQ):** Chất lượng Nhận dạng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/panoptic_quality.jpeg" style="width: 800px;"/>

##### Bước 1: Xác định matching pairs

Các matching pairs được tạo ra giữa các đối tượng được mô hình dự đoán (predicted segments) và các đối tượng thực tế trong ảnh gốc (ground truth segments).
Một cặp được coi là "khớp" nếu chỉ số Intersection over Union (IoU) của chúng lớn hơn một ngưỡng nhất định, thường là 0.5.

Dựa trên việc khớp nối này, chúng ta xác định được 3 tập hợp:
- **True Positives (TP):** Các cặp dự đoán và thực tế khớp với nhau (IoU > 0.5).
- **False Positives (FP):** Các đối tượng dự đoán không khớp với bất kỳ đối tượng thực tế nào.
- **False Negatives (FN):** Các đối tượng thực tế không được khớp với bất kỳ đối tượng dự đoán nào.

##### Bước 2: Tính toán Segmentation Quality (SQ) - Chất lượng Phân đoạn

SQ đo lường mức độ chính xác của việc phân đoạn cho các đối tượng đã được nhận dạng đúng.
Nó được tính bằng trung bình cộng của chỉ số IoU của tất cả các cặp True Positive (TP).

$$ \text{SQ} = \frac{\sum_{(p, g) \in TP} IoU(p, g)}{|TP|} $$

trong đó:
- $p$ là đối tượng được mô hình dự đoán.
- $g$ là đối tượng thực tế trong ảnh gốc.
- $IoU(p, g)$ là chỉ số Intersection over Union giữa đối tượng dự đoán và đối tượng thực tế.
- $|TP|$ là số lượng các cặp True Positive.

Nếu mô hình của bạn phát hiện đúng một chiếc xe, SQ sẽ cho biết vùng pixel mà mô hình tô cho chiếc xe đó có khớp chính xác với vùng pixel thực tế của chiếc xe hay không.
SQ cao nghĩa là các đối tượng được phát hiện đúng được phân đoạn rất chính xác.

##### Bước 3: Tính toán Recognition Quality (RQ) - Chất lượng Nhận dạng

RQ đo lường khả năng "phát hiện" đối tượng của mô hình, tương tự như chỉ số F1-score trong các bài toán phân loại.
Nó được tính dựa trên số lượng TP, FP và FN.

$$ \text{RQ} = \frac{|TP|}{|TP| + 0.5 \times |FP| + 0.5 \times |FN|} $$

trong đó:
- $|TP|$ là số lượng các cặp True Positive.
- $|FP|$ là số lượng các đối tượng dự đoán không khớp với bất kỳ đối tượng thực tế nào.
- $|FN|$ là số lượng các đối tượng thực tế không được khớp với bất kỳ đối tượng dự đoán nào.

RQ phạt mô hình nếu nó tạo ra các đối tượng "ma" không tồn tại (FP).
RQ phạt mô hình nếu nó bỏ sót các đối tượng có thật (FN).
RQ cao có nghĩa là mô hình phát hiện đối tượng rất tốt, không bỏ sót và cũng không dự đoán sai.

##### Bước 4: Kết hợp lại thành PQ

Khi nhân SQ và RQ với nhau, ta có công thức đầy đủ của PQ:

$$ \text{PQ} = \text{SQ} \times \text{RQ} $$
$$ \text{PQ} = \left( \frac{\sum_{(p, g) \in TP} IoU(p, g)}{|TP|} \right) \times \left( \frac{|TP|}{|TP| + 0.5 \times |FP| + 0.5 \times |FN|} \right) $$
$$ \text{PQ} = \frac{\sum_{(p, g) \in TP} IoU(p, g)}{|TP| + 0.5 \times |FP| + 0.5 \times |FN|} $$

trong đó:
- Tử số: Tổng chất lượng phân đoạn của các dự đoán đúng.
- Mẫu số: Tổng số đối tượng dự đoán đúng (TP), cộng với một nửa số đối tượng dự đoán sai (FP) và một nửa số đối tượng bị bỏ sót (FN).

## 4. Các kỹ thuật data augmentation cho image segmentation

Điểm khác biệt cốt lõi khi áp dụng data augmentation cho image segmentation so với các bài toán như image classification là bất kỳ phép biến đổi hình học nào được áp dụng lên ảnh gốc đều phải được áp dụng một cách chính xác và đồng bộ lên segmentation mask tương ứng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/rotate.jpeg" style="width: 1000px;"/>

Sau đó, các kỹ thuật data augmentation nói chung thường được sử dụng trong các bài toán computer vision được áp dụng tương tự cho bài toán image segmentation:
- Các phép biến đổi hình học: Random Flip, Random Rotate, Random Resize, Random Crop ...
- Các phép biến đổi chất lượng ảnh: Random Brightness, Random Contrast, Random Color, Random Saturation ...

Các kỹ thuật data augmentation được sử dụng trong bài toán object detection như MixUp, CutMix, CutOut đều có giá trị trong quá trình huấn luyện mô hình image segmentation.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/elastic_deformations.jpeg" style="width: 1000px;"/>

Ngoài ra, đối với image segmentation được sử dụng trong y tế như nhận diện với ảnh tế bào, ảnh khối u ..., Biến dạng đàn hồi (Elastic Deformations) là một kỹ thuật cực kỳ hữu ích.
Nó làm biến dạng cục bộ ảnh và mask một cách mượt mà, giống như kéo dãn một tấm cao su.
Kỹ thuật này mô phỏng sự biến dạng tự nhiên của các mô sinh học, giúp mô hình khái quát hóa tốt hơn với các hình dạng đa dạng trong thực tế.

## 5. Khó khăn khi huấn luyện mô hình giải bài toán image segmentation

### 5.1. Vấn đề mất cân bằng dữ liệu

Mất cân bằng dữ liệu giữa mask (vùng đối tượng quan tâm) và background (nền) là một trong những thách thức phổ biến và nghiêm trọng nhất khi huấn luyện các mô hình image segmentation.

Vấn đề này xảy ra khi số lượng pixel thuộc về background lớn hơn rất nhiều so với số lượng pixel của mask trong tập dữ liệu huấn luyện.
Ví dụ điển hình là trong các bài toán y tế như phát hiện khối u, khối u (mask) chỉ chiếm một phần rất nhỏ trên toàn bộ ảnh chụp y khoa (background), hay trong bài toán phát hiện vết nứt trên đường, vết nứt cũng chỉ là một vài đường mảnh trên một bề mặt bê tông rộng lớn.

Để giải quyết vấn đề này, phương pháp phổ biến nhất là sử dụng các hàm loss chuyên dụng như **Dice Loss** được giới thiệu trong [Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/pdf/1707.03237), **Tversky Loss** được giới thiệu trong [Tversky loss function for image segmentation using 3D fully convolutional deep networks](https://arxiv.org/pdf/1706.05721) và **Focal Los**s được giới thiệu trong [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002).
Các hàm loss này được thiết kế để "trừng phạt" nặng hơn khi mô hình dự đoán sai các pixel thuộc lớp thiểu số (mask), giúp cân bằng lại tầm quan trọng của hai lớp.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/imbalance_data.jpeg" style="width: 700px;"/>

### 5.2. Vấn đề chênh lệch kích thước đối tượng và kích thước ảnh

Tương tự như bài toán object detection, các mô hình image segmentation cũng gặp phải khó khăn với vấn đề chênh lệch kích thước đối tượng và kích thước ảnh, và thậm chí có thể còn thách thức hơn, khi xử lý các đối tượng nhỏ.

Các nguyên nhân gốc rễ rất giống với những gì xảy ra trong object detection, chủ yếu liên quan đến cách hoạt động của các mạng nơ-ron tích chập (CNN).
Hầu hết các kiến trúc segmentation hiện đại đều có một phần encoder để trích xuất các đặc trưng ngữ nghĩa.
Trong quá trình này, các đối tượng nhỏ (chỉ chiếm vài pixel) có thể bị "biến mất" hoàn toàn hoặc thông tin chi tiết về hình dạng và vị trí của chúng bị hòa lẫn vào các pixel xung quanh.
Khi đến phần decoder để khôi phục lại kích thước ban đầu để tạo ra mask đầu ra thì thông tin về đối tượng nhỏ đã không còn để có thể tái tạo lại một cách chính xác.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/scale_problem.jpeg" style="width: 700px;"/>

Đối với segmentation, việc xác định chính xác đường biên của đối tượng là cực kỳ quan trọng. 
Với một đối tượng chỉ rộng vài pixel, việc phân biệt pixel nào thuộc về đối tượng và pixel nào thuộc về nền là vô cùng khó khăn.
Một sai lệch nhỏ cũng có thể làm thay đổi đáng kể hình dạng của mặt nạ (mask) được tạo ra.

Một số giải pháp hỗ trợ mô hình học tốt hơn với các đối tượng nhỏ như:
- Cải thiện kiến trúc mô hình giúp mô hình giữ được nhiều đặc trưng về đối tượng nhỏ hơn như [High-Resolution Networks - HRNet](https://arxiv.org/pdf/1908.07919), [Feature Pyramid Network](https://arxiv.org/pdf/1612.03144) ...
- Cải thiện chiến lược cho quá trình dự đoán: Thư viện SAHI (Slicing Aided Hyper Inference) chia một ảnh lớn thành nhiều ảnh nhỏ hơn (patch/tile) và đưa vào mô hình dự đoán.
