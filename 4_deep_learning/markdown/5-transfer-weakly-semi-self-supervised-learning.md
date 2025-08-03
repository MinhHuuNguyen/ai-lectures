---
time: 10/08/2022
title: Transfer learning, weakly, semi và self supervised learning
description: Transfer learning là một kỹ thuật quan trọng trong machine learning và deep learning, giúp cải thiện hiệu suất mô hình khi dữ liệu có hạn hoặc tăng tốc quá trình huấn luyện. Ngoài ra, các kỹ thuật weakly, semi và self supervised learning cũng đóng vai trò quan trọng trong việc tận dụng dữ liệu không có nhãn hoặc có nhãn không chính xác để cải thiện chất lượng mô hình.
banner_url:
tags: [machine-learning, deep-learning]
is_highlight: false
is_published: true
---

## 1. Các vấn đề đối khi huấn luyện mô hình Supervised learning

Trong quá trình phát triển của machine learning và deep learning, các mô hình supervised learning vẫn cho ta những kết quả rất tốt trên nhiều bài toán khác nhau, trên cả dữ liệu hình ảnh Computer Vision, dữ liệu văn bản Natural Language Processing hay dữ liệu âm thanh Computer Audition ...

Tuy nhiên, điểm mạnh nhất của các mô hình supervised learning chính là khả năng học được những mối quan hệ phức tạp giữa dữ liệu đầu vào và label đầu ra, lại trở thành điểm yếu lớn nhất của chúng.
Các mô hình supervised learning yêu cầu sử dụng dữ liệu có nhãn trong xuyên suốt quá trình huấn luyện mô hình.
Điều này tạo ra rào cản khi ta muốn tăng thêm lượng dữ liệu nhằm cải thiện độ chính xác của mô hình.

Một số giải pháp có thể được sử dụng để giảm thiểu và phần nào giải quyết được vấn đề này là:
- **Tận dụng dữ liệu có nhãn từ các bộ dữ liệu khác**: Trên thế giới có thể đã có những bộ dữ liệu tương tự với bài toán mà ta đang muốn huấn luyện mô hình, ta có thể tận dụng những bộ dữ liệu này để huấn luyện mô hình.
- **Tận dụng dữ liệu không có nhãn**: Lượng dữ liệu không có nhãn thường rất lớn, ta có thể tìm cách tận dụng lượng dữ liệu này để cải thiện chất lượng mô hình.

## 2. Giới thiệu về Transfer learning

## 3. Zero shot, one shot, few shot learning

## 4. Các bài toán trong weakly supervised learning

Weakly supervised learning là một nhóm các phương pháp huấn luyện mô hình supervised learning với weakly labeled data.


Weakly labeled data bao gồm ba trường hợp:
- **Incomplete supervision data** là những bộ dữ liệu mà chỉ có một phần nhỏ là dữ liệu đã có label, còn lại rất nhiều là dữ liệu hoàn toàn chưa có label.
- **Inexact supervision data** là những bộ dữ liệu mà nhãn của một số điểm dữ liệu không đúng như những gì chúng ta cần cho bài toán.
- **Inaccurate supervision data** là những bộ dữ liệu mà nhãn của một số điểm dữ liệu không chính xác hoàn toàn, không hoàn toàn là ground-truth.


Đối với **Incomplete supervision data**, ta có các phương pháp semi supervised learning hay self supervised learning giúp tận dụng dữ liệu không có label để cải thiện chất lượng của mô hình huấn luyện trên dữ liệu có label.

Ví dụ: Xét một bộ dữ liệu ảnh với 1000 ảnh đã có label đúng và 1000 ảnh chưa có label.
Với mô hình A đã được huấn luyện trên 1000 ảnh có label.
Nếu ta có thể tận dụng 1000 ảnh chưa có label kết hợp với 1000 ảnh đã có label để huấn luyện mô hình B, ta kỳ vọng rằng mô hình B sẽ đạt được độ chính xác cao hơn so với mô hình A.

Đối với **Inaccurate supervision data**, ta có các phương pháp xác định và loại bỏ các dữ liệu có label không chính xác (dữ liệu có label noise) từ đó thu được bộ dữ liệu sạch và chất lượng của mô hình trên bộ dữ liệu sạch của tốt hơn.

Ví dụ: Xét một bộ dữ liệu ảnh với 1000 ảnh đã có label nhưng trong đó có 100 ảnh có label không đúng (label noise).
Nếu ta có thể xác định và loại bỏ 100 ảnh này, ta sẽ thu được bộ dữ liệu sạch với 900 ảnh có label đúng.
Với mô hình A đã được huấn luyện trên 900 ảnh có label đúng, ta kỳ vọng rằng mô hình A sẽ đạt được độ chính xác cao hơn so với mô hình B đã được huấn luyện trên 1000 ảnh có label (bao gồm cả 100 ảnh có label không đúng).

Đối với **Inexact supervision data**, đây là phần nghiên cứu chính của weakly supervised learning.
Mục tiêu của các nghiên cứu xử lý Inexact supervision data là với dữ liệu có label đơn giản hơn (như image classification) nhưng vẫn có thể giải quyết được những bài toán đòi hỏi label phức tạp hơn (như object detection hay instance segmentation).

Ví dụ: Xét một bộ dữ liệu ảnh với 1000 ảnh đã có label là các nhãn của ảnh (image classification).
Nếu ta có thể sử dụng bộ dữ liệu này để huấn luyện mô hình có thể dự đoán được các bounding box của các đối tượng trong ảnh (object detection) hoặc phân đoạn các đối tượng trong ảnh (instance segmentation), ta đã giải quyết được bài toán Inexact supervision data.

## 5. Giới thiệu về semi supervised learning

Ý tưởng chung của semi supervised learning để giải quyết vấn đề **Incomplete supervision data** nằm ở việc huấn luyện mô hình với số lượng ít dữ liệu có label, sau đó sử dụng mô hình này đã được huấn luyện để dự đoán label của số lượng nhiều các dữ liệu chưa có label, các label được dự đoán ra này được gọi là pseudo label.

Có hai phương pháp phổ biến để hiện thực hoá ý tưởng này là Self-training và Co-training

### 5.1. Self-training

Self-training là phương pháp mà trong đó mô hình được huấn luyện ban đầu trên tập dữ liệu có label, sau đó được sử dụng để dự đoán label cho tập dữ liệu chưa có label.
Các label được dự đoán này (pseudo label) sẽ được sử dụng để tiếp tục huấn luyện mô hình.

Trong quá trình sử dụng các pseudo label của self-training, ta cần chú ý đến việc lọc ra những pseudo label có độ tin cậy cao để sử dụng cho việc huấn luyện.

### 5.2. Co-training

Co-training là phương pháp mà trong đó hai mô hình được huấn luyện song song trên hai tập dữ liệu khác nhau nhưng có cùng một nhiệm vụ.
Mỗi mô hình sẽ dự đoán label cho dữ liệu chưa có label và chia sẻ các dự đoán này với nhau để cải thiện chất lượng của cả hai mô hình.

## 6. Giới thiệu về self supervised learning

Ý tưởng chung của self supervised learning để giải quyết vấn đề **Incomplete supervision data** nằm ở việc định nghĩa ra và huấn luyện mô hình theo một bài toán mới từ bộ dữ liệu không có label, sau đó kết hợp với transfer learning để huấn luyện bộ dữ liệu có label trên bài toán gốc ban đầu.

Hay nói cách khác, self supervised learning là quá trình ta tự định nghĩa bài toán mới, tự sinh ra label cho dữ liệu và huấn luyện mô hình.
Từ đó, ta kỳ vọng rằng mô hình sau quá trình self supervised learning sẽ cung cấp nhiều thông tin hữu ích cho việc huấn luyện mô hình trên bài toán ban đầu.

Vì vậy, điểm quan trọng nhất của self supervised learning chính là việc định nghĩa bài toán mới cho mô hình (pretext task) sao cho nó phù hợp nhất với bài toán gốc (downstream task).

### 6.1. Pretext task với dữ liệu ảnh

#### Color Transformation

#### Geometric Transformation

#### Jigsaw puzzle

#### Contrastive learning

### 6.2. Pretext task với dữ liệu text

#### Word prediction

Center word prediction

Neighbor word prediction

Masked language modelling

#### Sentence prediction

Neighbor sentence prediction

Next Sentence Prediction

#### Emoji Prediction

## 7. Contrastive Language-Image Pre-Training (CLIP)







Phương pháp giải quyết vấn đề này xoay quanh việc sử dụng feature maps activation trong quá trình huấn luyện mô hình hoặc trích xuất và tinh chỉnh feature maps activation để lấy ra được những phần dự đoán như bài toán mong muốn.

<img src="https://lh4.googleusercontent.com/ObRTkjqqlsR8lb-dQJ0ZRonjKoD5t4_yFSRpJHOvg0y-gEm_wsPqvYVpBHo9LpfxGi4hwkRimpGcEoodQGjXiYFa-ZHYUuS5g9PmwHXWRkWfjnQj0N-J52TGpC9nmJXb_uD3HtJE2p__71uTeOKjqwdtfICDyXI2v5jheswO0NQMN-Trk8pmwsqi4sK_1w" style="width: 1200px;"/>


<!-- semi supervised learning Assumption -->

<!-- label propagation -->


<img src="https://rl.uni-freiburg.de/img/teaching/selfsup-seminar" style="width: 1200px;"/>

