---
time: 10/08/2022
title: Transfer learning, weakly, semi, un và self supervised learning
description: Transfer learning là một kỹ thuật quan trọng trong machine learning và deep learning, giúp cải thiện hiệu suất mô hình khi dữ liệu có hạn hoặc tăng tốc quá trình huấn luyện. Ngoài ra, các kỹ thuật weakly, semi, un và self supervised learning cũng đóng vai trò quan trọng trong việc tận dụng dữ liệu không có nhãn hoặc có nhãn không chính xác để cải thiện chất lượng mô hình.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/banner.png
tags: [machine-learning, deep-learning]
is_highlight: false
is_published: true
---

## 1. Các vấn đề đối khi huấn luyện mô hình Supervised learning

Trong quá trình phát triển của machine learning và deep learning, các mô hình supervised learning vẫn cho ta những kết quả rất tốt trên nhiều bài toán khác nhau, trên cả dữ liệu hình ảnh Computer Vision, dữ liệu văn bản Natural Language Processing hay dữ liệu âm thanh Computer Audition ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/annotation_levels.png" style="width: 900px;"/>

Tuy nhiên, điểm mạnh nhất của các mô hình supervised learning chính là khả năng học được những mối quan hệ phức tạp giữa dữ liệu đầu vào và label đầu ra, lại trở thành điểm yếu lớn nhất của chúng.
Các mô hình supervised learning yêu cầu sử dụng dữ liệu có nhãn trong xuyên suốt quá trình huấn luyện mô hình.
Điều này tạo ra rào cản khi ta muốn tăng thêm lượng dữ liệu nhằm cải thiện độ chính xác của mô hình.

Một số giải pháp có thể được sử dụng để giảm thiểu và phần nào giải quyết được vấn đề này là:
- **Tận dụng dữ liệu có nhãn từ các bộ dữ liệu khác**: Trên thế giới có thể đã có những bộ dữ liệu tương tự với bài toán mà ta đang muốn huấn luyện mô hình, ta có thể tận dụng những bộ dữ liệu này để huấn luyện mô hình.
- **Tận dụng dữ liệu không có nhãn**: Lượng dữ liệu không có nhãn thường rất lớn, ta có thể tìm cách tận dụng lượng dữ liệu này để cải thiện chất lượng mô hình.

## 2. Giới thiệu về Transfer learning

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/transfer_examples.png" style="width: 900px;"/>


<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/transfer_statistics.png" style="width: 900px;"/>


<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/transfer_architecture.png" style="width: 900px;"/>

## 3. Zero shot, one shot, few shot learning

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/zero_one_few_shot.png" style="width: 900px;"/>

## 4. Các bài toán trong weakly supervised learning

Weakly supervised learning là một nhóm các phương pháp huấn luyện mô hình supervised learning với weakly labeled data.


Weakly labeled data bao gồm ba trường hợp:
- **Incomplete supervision data** là những bộ dữ liệu mà chỉ có một phần nhỏ là dữ liệu đã có label, còn lại rất nhiều là dữ liệu hoàn toàn chưa có label.
- **Inexact supervision data** là những bộ dữ liệu mà nhãn của một số điểm dữ liệu không đúng như những gì chúng ta cần cho bài toán.
- **Inaccurate supervision data** là những bộ dữ liệu mà nhãn của một số điểm dữ liệu không chính xác hoàn toàn, không hoàn toàn là ground-truth.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/weak_supervision.png" style="width: 900px;"/>

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

### 5.1. Graph-based Regularization

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_graph_based.png" style="width: 900px;"/>

### 5.2. Generative Models

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_gan_based.png" style="width: 900px;"/>

### 5.3. Pseudo Labels

Ý tưởng chung của semi supervised learning để giải quyết vấn đề **Incomplete supervision data** nằm ở việc huấn luyện mô hình với số lượng ít dữ liệu có label, sau đó sử dụng mô hình này đã được huấn luyện để dự đoán label của số lượng nhiều các dữ liệu chưa có label, các label được dự đoán ra này được gọi là .

#### Self-training

Self-training là phương pháp mà trong đó mô hình được huấn luyện ban đầu trên tập dữ liệu có label, sau đó được sử dụng để dự đoán label cho tập dữ liệu chưa có label.
Các label được dự đoán này (pseudo label) sẽ được sử dụng để tiếp tục huấn luyện mô hình.

Trong quá trình sử dụng các pseudo label của self-training, ta cần chú ý đến việc lọc ra những pseudo label có độ tin cậy cao để sử dụng cho việc huấn luyện.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_pseudo_labels.png" style="width: 900px;"/>

#### Co-training

Co-training là phương pháp mà trong đó hai mô hình được huấn luyện song song trên hai tập dữ liệu khác nhau nhưng có cùng một nhiệm vụ.
Mỗi mô hình sẽ dự đoán label cho dữ liệu chưa có label và chia sẻ các dự đoán này với nhau để cải thiện chất lượng của cả hai mô hình.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_co_training.png" style="width: 900px;"/>

#### Meta Pseudo Label

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_meta_pseudo_labels.png" style="width: 900px;"/>

### 5.3. Consistency Regularization và Contrastive Learning

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_consistency_regularization.png" style="width: 900px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_contrastive_learning.png" style="width: 900px;"/>

## 6. Giới thiệu về self supervised learning - unsupervised learning

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_supervised_learning.png" style="width: 900px;"/>

Ý tưởng chung của self supervised learning để giải quyết vấn đề **Incomplete supervision data** nằm ở việc định nghĩa ra và huấn luyện mô hình theo một bài toán mới từ bộ dữ liệu không có label, sau đó kết hợp với transfer learning để huấn luyện bộ dữ liệu có label trên bài toán gốc ban đầu.

Hay nói cách khác, self supervised learning là quá trình ta tự định nghĩa bài toán mới, tự sinh ra label cho dữ liệu và huấn luyện mô hình.
Từ đó, ta kỳ vọng rằng mô hình sau quá trình self supervised learning sẽ cung cấp nhiều thông tin hữu ích cho việc huấn luyện mô hình trên bài toán ban đầu.

Vì vậy, điểm quan trọng nhất của self supervised learning chính là việc định nghĩa bài toán mới cho mô hình (pretext task) sao cho nó phù hợp nhất với bài toán gốc (downstream task).

### 6.1. Pretext task với dữ liệu ảnh

#### Autoencoders

Pretext task **Autoencoders** là một trong những phương pháp phổ biến nhất trong self supervised learning với dữ liệu hình ảnh.

Trong phương pháp này, mô hình sẽ được huấn luyện để mã hoá (encode) dữ liệu đầu vào thành một biểu diễn (representation) và sau đó giải mã (decode) biểu diễn này trở lại thành dữ liệu đầu vào ban đầu.
Mục tiêu của mô hình là giảm thiểu sự khác biệt giữa dữ liệu đầu vào và dữ liệu đầu ra sau khi giải mã.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_supervised_autoencoders.png" style="width: 900px;"/>

Hình ảnh trên được lấy từ bài báo [A Survey on Self-Supervised Representation Learning](https://arxiv.org/abs/2308.11455), mô tả cách thức hoạt động của pretext task **Autoencoders**.

Phần encoder sau khi được huấn luyện trên pretext task này sẽ có khả năng trích xuất các đặc trưng (features) của dữ liệu đầu vào tốt, vì nó đã lưu giữ được những thông tin quan trọng nhất của dữ liệu đầu vào vào trong biểu diễn (representation) và có thể được sử dụng để huấn luyện các mô hình khác trên các bài toán downstream task.

#### Masked Autoencoders

Một phiên bản khác cũng hướng đến việc khôi phục lại dữ liệu đầu vào là **Masked Autoencoders**.

Trong phương pháp này, dữ liệu đầu vào sẽ được chia thành các mảnh nhỏ (patches) và một phần các patches sẽ bị che đi (mask) và mô hình sẽ được huấn luyện để khôi phục lại phần dữ liệu đã bị che.
Tuy nhiên, thay vì phải huấn luyện phần decoder để phục dựng lại toàn bộ ảnh đầu vào, trong **Masked Autoencoders**, mô hình chỉ cần học để tìm lại đúng các patches đã bị che đi.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_masked.png" style="width: 900px;"/>

Hình ảnh trên được lấy từ bài báo [A Survey on Self-Supervised Representation Learning](https://arxiv.org/abs/2308.11455), mô tả cách thức hoạt động của pretext task **Masked Autoencoders**.

Với phương pháp này, mô hình cũng sẽ được học cách trích xuất các đặc trưng (features) của dữ liệu đầu vào tốt nhưng quá trình huấn luyện sẽ đơn giản hơn so với pretext task **Autoencoders** vì mô hình chỉ cần giải quyết bài toán classification cho các patches đã bị che đi thay vì phải khôi phục lại toàn bộ ảnh đầu vào.

#### Rotation Transformation

Pretext task **Rotation Transformation** là một phương pháp self supervised learning với dữ liệu hình ảnh.

Trong phương pháp này, dữ liệu đầu vào sẽ được xoay một góc ngẫu nhiên (0, 90, 180 hoặc 270 độ) và góc xoay này sẽ được sử dụng làm nhãn (label) cho mô hình.
Mô hình sẽ được huấn luyện để dự đoán góc xoay của dữ liệu sau khi xoay.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_supervised_rotation.png" style="width: 900px;"/>

Hình ảnh trên được lấy từ bài báo [A Survey on Self-Supervised Representation Learning](https://arxiv.org/abs/2308.11455), mô tả cách thức hoạt động của pretext task **Rotation Transformation**.

Mô hình sau khi được huấn luyện trên pretext task này sẽ có khả năng hiểu các đặc trưng không gian của dữ liệu hình ảnh, vì nó đã học được cách phân biệt các góc xoay khác nhau và có thể được sử dụng để huấn luyện các mô hình khác trên các bài toán downstream task.

#### Jigsaw puzzle

Một pretext task khác cũng hướng đến việc hiểu cấu trúc không gian của dữ liệu hình ảnh là **Jigsaw puzzle**.
Trong phương pháp này, dữ liệu đầu vào sẽ được chia thành các mảnh nhỏ (patches) và các mảnh này sẽ được xáo trộn ngẫu nhiên.
Mô hình sẽ được huấn luyện để dự đoán thứ tự của các mảnh này, tức là xác định cách sắp xếp các mảnh để chúng tạo thành một bức tranh hoàn chỉnh.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_supervised_jigsaw.png" style="width: 900px;"/>

Hình ảnh trên được lấy từ bài báo [A Survey on Self-Supervised Representation Learning](https://arxiv.org/abs/2308.11455), mô tả cách thức hoạt động của pretext task **Jigsaw puzzle**.

Tương tự như pretext task **Rotation Transformation**, mô hình sau khi được huấn luyện trên pretext task này sẽ có khả năng hiểu cấu trúc không gian của dữ liệu hình ảnh, vì nó đã học được cách phân biệt các mảnh khác nhau và cách sắp xếp chúng và có thể được sử dụng để huấn luyện các mô hình khác trên các bài toán downstream task.

#### Image Transformation

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_supervised_transformations.png" style="width: 900px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_supervised_transformations_train.png" style="width: 900px;"/>

#### Teacher - Student

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_supervised_dino_teacher_student.png" style="width: 900px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_supervised_dino.png" style="width: 900px;"/>

### 6.2. Pretext task với dữ liệu text

#### Masked language modelling

Phương pháp **Masked language modelling** là phương pháp, trong đó, một số từ trong câu bị che đi (mask) và mô hình phải đoán những từ này dựa vào cả ngữ cảnh bên trái lẫn phải của từ đó.

Đây là một trong những phương pháp phổ biến nhất trong self supervised learning với dữ liệu văn bản, được sử dụng trong các mô hình như BERT.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/bert.png" style="width: 600px;"/>

Xét ví dụ trên hình được lấy từ bài báo ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805), một cặp câu A và B được đưa vào mô hình BERT. Để phục vụ cho bài toán **Masked Language Model**:
- Trên hai câu này, một lượng khoảng 15% số lượng token đã được che đi (mask).
- Ví dụ: 80% lượng dữ liệu sẽ được mask với dạng "my dog is hairy" thành "my dog is [MASK]", 10% sẽ được thay thế bằng một token ngẫu nhiên khác "my dog is hairy" thành "my dog is apple", và 10% sẽ giữ nguyên token gốc "my dog is hairy" là "my dog is hairy".
- Các token đã bị che trong cặp câu trên sau khi được đưa vào mô hình sẽ được đi qua lớp softmax để dự đoán token gốc của chúng.

#### Standard language modelling

Phương pháp **Standard Language Modeling** là phương pháp yêu cầu mô hình dự đoán từ tiếp theo trong chuỗi văn bản dựa trên các từ trước đó.

Hình ảnh dưới đây được lấy từ bài báo ["Improving Language Understanding by Generative Pre-Training"](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), mô tả cách thức huấn luyện **Standard Language Modeling**.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/gpt.png" style="width: 600px;"/>

Đối với bài toán **text classification** (phân lớp văn bản), mô hình GPT sẽ đơn giản là nhận đầu vào là một chuỗi văn bản, sau đó đưa đầu ra qua một lớp softmax để dự đoán nhãn của văn bản.

Đối với bài toán **text entailment** (suy diễn mối quan hệ văn bản), mô hình GPT sẽ nhận đầu vào câu tiền đề (premise) được nối với câu giả thuyết (hypothesis) và đưa ra đầu ra là xác suất của các nhãn như "entailment" (suy diễn), "contradiction" (mâu thuẫn) và "neutral" (trung lập).

Đối với bài toán **text similarity** (tương đồng văn bản), mô hình GPT sẽ nhận đầu vào là hai câu và nối hai câu này với thứ tự ngược nhau ("sentence A" + "sentence B" và "sentence B" + "sentence A") và đưa ra đầu ra là xác suất của các nhãn như "similar" (tương đồng) và "dissimilar" (không tương đồng).
Phương pháp này được áp dụng tương tự cho bài toán **Multiple-Choice Question Answering** (trả lời câu hỏi nhiều lựa chọn) bằng cách nối câu hỏi với các lựa chọn trả lời và đưa ra xác suất cho từng lựa chọn.

#### Next Sentence Prediction

Phương pháp **Next Sentence Prediction** là phương pháp, trong đó, mô hình sẽ được huấn luyện để dự đoán xem một câu có phải là câu tiếp theo của một câu khác hay không.

Đây là một trong những phương pháp phổ biến nhất trong self supervised learning với dữ liệu văn bản, được sử dụng trong các mô hình như BERT.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/bert.png" style="width: 600px;"/>

Xét ví dụ trên hình được lấy từ bài báo ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805), một cặp câu A và B được đưa vào mô hình BERT. Để phục vụ cho bài toán **Next Sentence Prediction**:
- Input của mô hình sẽ được thêm một token đặc biệt [CLS] ở đầu câu A và nhiệm vụ của mô hình là dự đoán giá trị của token này là 1 (IsNext) hoặc 0 (NotNext) để xác định xem câu B có phải là câu tiếp theo của câu A hay không.
- Ví dụ: "[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]" sẽ có label của token [CLS] là IsNext.

## 7. Contrastive Language-Image Pre-Training (CLIP)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/clip.png" style="width: 900px;"/>







Phương pháp giải quyết vấn đề này xoay quanh việc sử dụng feature maps activation trong quá trình huấn luyện mô hình hoặc trích xuất và tinh chỉnh feature maps activation để lấy ra được những phần dự đoán như bài toán mong muốn.


<!-- semi supervised learning Assumption -->

<!-- label propagation -->


