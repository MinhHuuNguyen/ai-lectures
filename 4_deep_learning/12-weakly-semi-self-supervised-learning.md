---
layout: "post"
title:  "Weakly, semi & self supervised learning"
author: "Nguyễn Hữu Minh"
permalink: "/deep-learning/weakly-semi-self-supervised-learning"
parent: "Deep learning"
nav_order: 12
---

# Weakly, semi và self supervised learning

## 1. Các vấn đề đối với supervised learning

Trong quá trình phát triển của machine learning và deep learning, các mô hình supervised learning vẫn cho ta những kết quả rất tốt trên nhiều bài toán khác nhau, tuy nhiên, vấn đề lớn của các mô hình supervised learning nằm ở vấn đề dữ liệu.

Các mô hình supervised learning yêu cầu sử dụng dữ liệu có nhãn trong xuyên suốt quá trình huấn luyện mô hình.
Điều này tạo ra rào cản khi ta muốn tăng thêm lượng dữ liệu nhằm cải thiện độ chính xác của mô hình.

<img src="https://drive.google.com/uc?id=1Gogu1aJAYvzm2FBhTnQhxWUuASgTjoX9" style="width: 800px;"/>

## 2. Các bài toán trong weakly supervised learning

Weakly supervised learning là một nhóm các phương pháp huấn luyện mô hình supervised learning với weakly labeled data.

<img src="https://drive.google.com/uc?id=1Jv3uRwN0OzpJpNjkvUj6XKkkTGqRWe7x" style="width: 1000px;"/>

Weakly labeled data bao gồm ba trường hợp:
- Incomplete supervision data là những tình huống mà bộ dữ liệu chỉ có một phần nhỏ là dữ liệu đã có label, còn lại rất nhiều là dữ liệu chưa có label.
- Inexact supervision data là những tình huống mà nhãn của bộ dữ liệu không đúng như những gì cta mong đợi.
- Inaccurate supervision data là những tình huống mà nhãn của bộ dữ liệu không chính xác hoàn toàn, không hoàn toàn là ground-truth.

Đối với Incomplete supervision data, ta có các phương pháp semi supervised learning hay self supervised learning giúp tận dụng dữ liệu không có label để cải thiện chất lượng của mô hình huấn luyện trên dữ liệu có label.
Chi tiết về các kỹ thuật semi supervised learning và self supervised learning được nêu ở phần phía dưới.

Còn đối với Inaccurate supervision data, ta có các phương pháp xác định và loại bỏ các dữ liệu có label không chính xác (dữ liệu có label noise) từ đó thu được bộ dữ liệu sạch và chất lượng của mô hình trên bộ dữ liệu sạch của tốt hơn.

<img src="https://drive.google.com/uc?id=1kGq7XuM6PIMsspuEzzagLpL6Y9cZMuOh" style="width: 800px;"/>

Đối với Inexact supervision data, đây là phần nghiên cứu chính của weakly supervised learning.
Mục tiêu của các nghiên cứu xử lý Inexact supervision data là với dữ liệu có label đơn giản hơn (như image classification) nhưng vẫn có thể giải quyết được những bài toán đòi hỏi label phức tạp hơn (như object detection hay instance segmentation).

Phương pháp giải quyết vấn đề này xoay quanh việc sử dụng feature maps activation trong quá trình huấn luyện mô hình hoặc trích xuất và tinh chỉnh feature maps activation để lấy ra được những phần dự đoán như bài toán mong muốn.

## 3. Ý tưởng của semi supervised learning

Ý tưởng chung của semi supervised learning để giải quyết vấn đề Incomplete supervision data nằm ở việc huấn luyện mô hình với số lượng ít dữ liệu có label, sau đó sử dụng mô hình này đã được huấn luyện để dự đoán label của số lượng nhiều các dữ liệu chưa có label, các label được dự đoán ra này được gọi là pseudo label.

Có hai phương pháp phổ biến để hiện thực hoá ý tưởng này là Self-training và Co-training:

<!-- semi supervised learning Assumption -->

### 3.1. Pseudo labelling và Self training

Ý tưởng của pseudo labelling và self training khá đơn giản gồm các bước sau:
- Huấn luyện mô hình với một số ít dữ liệu đã có label
- Dùng mô hình đã được huấn luyện để dự đoán với dữ liệu chưa có label (pseudo label)
- Dùng kết hợp dữ liệu có label và dữ liệu vừa mới được dự đoán để tiếp tục huấn luyện mô hình

<img src="https://lh4.googleusercontent.com/ObRTkjqqlsR8lb-dQJ0ZRonjKoD5t4_yFSRpJHOvg0y-gEm_wsPqvYVpBHo9LpfxGi4hwkRimpGcEoodQGjXiYFa-ZHYUuS5g9PmwHXWRkWfjnQj0N-J52TGpC9nmJXb_uD3HtJE2p__71uTeOKjqwdtfICDyXI2v5jheswO0NQMN-Trk8pmwsqi4sK_1w" style="width: 400px;"/>

Self-training nâng cấp hơn một chút so với pseudo labelling ở điểm ta chỉ sử dụng những pseudo label có confidence cao cho việc huấn luyện mô hình ở những vòng tiếp theo.

### 3.2. Co training

Ý tưởng của co training được thể hiện thông qua việc huấn luyện song song hai mô hình:
- Huấn luyện hai mô hình với hai phần khác nhau của dữ liệu đã có label
- Dùng hai mô hình đã được huấn luyện để dự đoán với dữ liệu chưa có label (pseudo label)
- Ta dùng pseudo label với confidence cao của mô hình này để huấn luyện cho mô hình kia và ngược lại.
- Cuối cùng, ta thu được pseudo label cuối cùng của bộ dữ liệu không có label bằng việc kết hợp kết quả dự đoán của cả hai mô hình.

<img src="https://drive.google.com/uc?id=12A_iWHf3jR0zEZ-Wk4Qe2SUSPe6EgGhg" style="width: 500px;"/>

<!-- label propagation -->

## 4. Ý tưởng của self supervised learning

Ý tưởng chung của self supervised learning để giải quyết vấn đề Incomplete supervision data nằm ở việc định nghĩa ra và huấn luyện mô hình theo một bài toán mới từ bộ dữ liệu không có label, sau đó kết hợp với transfer learning để huấn luyện bộ dữ liệu có label trên bài toán gốc ban đầu.

<img src="https://rl.uni-freiburg.de/img/teaching/selfsup-seminar" style="width: 400px;"/>

Hay nói cách khác, self supervised learning là quá trình ta tự định nghĩa bài toán mới, tự sinh ra label cho dữ liệu và huấn luyện mô hình.
Từ đó, ta kỳ vọng rằng mô hình sau quá trình self supervised learning sẽ cung cấp nhiều thông tin hữu ích cho việc huấn luyện mô hình trên bài toán ban đầu.

Vì vậy, điểm quan trọng nhất của self supervised learning chính là việc định nghĩa bài toán mới cho mô hình (pretext task) sao cho nó phù hợp nhất với bài toán gốc (downstream task).
Một số pretext task cơ bản có thể được sử dụng như:

### 4.1. Pretext task với dữ liệu ảnh

#### 4.1.1. Color Transformation

<img src="https://drive.google.com/uc?id=1NE4KeiI8QXRmhdr4rNLvvc8MCTUZytlJ" style="width: 400px;"/>

#### 4.1.2. Geometric Transformation

<img src="https://drive.google.com/uc?id=1BBU_zpyVrwjSTqqbeD8vTwQuje4nPuLl" style="width: 400px;"/>

#### 4.1.3. Jigsaw puzzle

<img src="https://drive.google.com/uc?id=1wWTH-1inYx9IxG7UisiwqGEclkBeR9-a" style="width: 600px;"/>

#### 4.1.4. Contrastive learning

<img src="https://drive.google.com/uc?id=1gtH_XWJX8NFaYT3FvS9l33CdrdgdTWS4" style="width: 700px;"/>

### 4.2. Pretext task với dữ liệu text

#### 4.2.1. Word prediction

Center word prediction

<img src="https://amitness.com/images/nlp-ssl-center-word-prediction.gif" style="width: 700px;"/>

Neighbor word prediction

<img src="https://amitness.com/images/nlp-ssl-neighbor-word-prediction.gif" style="width: 700px;"/>

Masked language modelling

<img src="https://amitness.com/images/nlp-ssl-masked-lm.png" style="width: 700px;"/>

#### 4.2.2. Sentence prediction

Neighbor sentence prediction

<img src="https://amitness.com/images/nlp-ssl-neighbor-sentence.gif" style="width: 700px;"/>

Next Sentence Prediction

<img src="https://amitness.com/images/nlp-ssl-nsp-sampling.png" style="width: 700px;"/>

<img src="https://amitness.com/images/nlp-ssl-next-sentence-prediction.png" style="width: 700px;"/>

#### 4.2.3. Emoji Prediction

<img src="https://amitness.com/images/nlp-ssl-deepmoji.gif" style="width: 700px;"/>
