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

Tuy nhiên, điểm mạnh nhất của các mô hình supervised learning chính là khả năng học được những mối quan hệ phức tạp giữa dữ liệu đầu vào và label đầu ra, lại trở thành điểm yếu lớn nhất của chúng.
Các mô hình supervised learning yêu cầu sử dụng dữ liệu có nhãn trong xuyên suốt quá trình huấn luyện mô hình.
Điều này tạo ra rào cản khi ta muốn tăng thêm lượng dữ liệu nhằm cải thiện độ chính xác của mô hình.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/annotation_levels.png" style="width: 800px;"/>

Hình ảnh trên được lấy từ bài báo [Weakly Supervised Object Localization and Detection: A Survey](https://arxiv.org/abs/2104.07918), mô tả các cấp độ của dữ liệu có nhãn annotation levels từ dễ đến khó và chi phí cần để gán nhãn được các cấp độ này từ rẻ, đến đắt và đến mức độ khó mà chỉ có thể gán nhãn bởi các chuyên gia trong lĩnh vực.
Ngoài ra, sơ đồ phía trên cũng thể hiện được một số trường hợp weak label như:
- Yêu cầu nhãn bounding box cho một số đối tượng trong ảnh, nhưng chỉ có nhãn class của ảnh.
- Yêu cầu nhãn segmentation theo từng pixel cho một số đối tượng trong ảnh, nhưng chỉ có nhãn class của ảnh hoặc nhãn bounding box của đối tượng.
- Yêu cầu nhãn mesh 3D cho một số đối tượng trong ảnh, nhưng chỉ có nhãn class của ảnh hoặc nhãn bounding box của đối tượng hoặc nhãn segmentation theo từng pixel của đối tượng.

Một số giải pháp có thể được sử dụng để giảm thiểu và phần nào giải quyết được vấn đề này là:
- **Tận dụng dữ liệu có nhãn từ các bộ dữ liệu khác**: Trên thế giới có thể đã có những bộ dữ liệu tương tự với bài toán mà ta đang muốn huấn luyện mô hình, ta có thể tận dụng những bộ dữ liệu này để huấn luyện mô hình.
- **Tận dụng dữ liệu không có nhãn**: Lượng dữ liệu không có nhãn thường rất lớn, ta có thể tìm cách tận dụng lượng dữ liệu này để cải thiện chất lượng mô hình.

## 2. Giới thiệu về Transfer learning

Ý tưởng chính của transfer learning là tận dụng mô hình đã được huấn luyện trên một bài toán hoặc một bộ dữ liệu nào đó để giải quyết một bài toán khác hoặc một bộ dữ liệu khác.

Hình ảnh dưới đây được lấy từ bài báo [A Comprehensive Survey on Transfer Learning](https://arxiv.org/abs/1911.02685), mô tả đơn giản về kiến thức mà mô hình đã được học được kế thừa sang các bài toán mới.
Ví dụ, mô hình đã được học với dữ liệu cờ tướng có thể được sử dụng để giải quyết bài toán cờ vua, mô hình đã được học với dữ liệu về đàn violin có thể được sử dụng để giải quyết bài toán đàn piano, mô hình đã được học với dữ liệu về xe đạp có thể được sử dụng để giải quyết bài toán xe máy.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/transfer_examples.png" style="width: 500px;"/>

Một thống kê được lấy từ bài báo [Deep transfer learning for image classification: a survey](https://arxiv.org/abs/2205.09904) cho thấy rằng kích thước của mô hình càng hơn thì độ chính xác của mô hình càng cao, tương tự, mô hình được huấn luyện trên bộ dữ liệu lớn hơn thì độ chính xác của mô hình càng cao.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/transfer_statistics.png" style="width: 1000px;"/>

Điều này đặt ra vấn đề việc làm thế nào để tăng được kích thước của mô hình song hành với kích thước của bộ dữ liệu huấn luyện mô hình, để cực đại hoá độ chính xác của mô hình.
Không những thế, ta cần cực đại hoá độ chính xác của mô hình với một lượng chi phí cực tiểu, cả chi phí chuẩn bị dữ liệu và chi phí huấn luyện mô hình.

Từ đó, việc sử dụng transfer learning với những mô hình đã được huấn luyện trước trên những bộ dữ liệu public lớn gần như trở thành một tiêu chuẩn trong quá trình phát triển mô hình machine learning và deep learning.
Không những vậy, một số mô hình được huấn luyện trên những bộ dữ liệu đặc thù private, thậm chí còn được bán với chi phí không rẻ để được sử dụng transfer learning cho các mô hình mới.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/transfer_architecture.png" style="width: 1000px;"/>

Phương án đơn giản nhất để sử dụng transfer learning là ta sẽ sử dụng phần backbone của mô hình đã được huấn luyện trước, sau đó thêm một vài lớp fully connected (FC) để phù hợp với bài toán mới.

Hình ảnh trên được lấy từ bài báo [Deep transfer learning for image classification: a survey](https://arxiv.org/abs/2205.09904), mô tả về quá trình transfer learning mô hình đã được huấn luyện trước (pre-trained model) sang bài toán mới (downstream task) bằng việc kế thừa các lớp convolutional từ pretrained model và thêm mới các lớp fully connected (FC) để phù hợp với bài toán mới.
- Cách 1: **Khoá freeze các lớp kế thừa từ pretrained model** để không cập nhật trọng số của chúng trong quá trình huấn luyện mô hình mới và chỉ cập nhật trọng số của các lớp FC mới.
Trong trường hợp này, lượng tham số cần cập nhật sẽ ít hơn, thời gian huấn luyện sẽ nhanh hơn và lượng dữ liệu cần có để huấn luyện mô hình cũng sẽ ít hơn.
- Cách 2: **Mở unfreeze một số lớp kế thừa từ pretrained model** để cập nhật trọng số của chúng đi kèm với các lớp FC mới.
Trong trường hợp này, lượng tham số cần cập nhật sẽ nhiều hơn một chút so với cách 1.
- Cách 3: **Mở unfreeze tất cả các lớp kế thừa từ pretrained model** để cập nhật trọng số của chúng đi kèm với các lớp FC mới.
Trong trường hợp này, lượng tham số cần cập nhật sẽ nhiều nhất, tương đương với việc huấn luyện một mô hình mới từ đầu.
Tuy nhiên, việc mô hình đã được huấn luyện trước trên một bộ dữ liệu lớn, thời gian cần thiết để mô hình mới đạt được điểm hội tụ sẽ nhanh hơn rất nhiều so với việc huấn luyện một mô hình mới từ đầu.

Một số backbone nổi tiếng thường được sử dụng trong transfer learning như: ResNet hoặc ViT được pretrained với bộ dữ liệu ImageNet, BERT hoặc GPT được pretrained với bộ dữ liệu văn bản lớn ...

## 3. Zero shot, one shot, few shot learning

Zero shot, one shot và few shot learning là các kỹ thuật trong transfer learning giúp mô hình có thể học được từ rất ít dữ liệu hoặc thậm chí không cần dữ liệu nào.

**Zero-shot learning** là phương pháp cho phép mô hình dự đoán chính xác các lớp chưa từng thấy trong quá trình huấn luyện, mà không cần bất kỳ mẫu nào thuộc các lớp đó.

Yêu cầu tiên quyết của zero-shot learning là mô hình được sử dụng phải có khả năng khái quát hoá tốt, hiểu được các đặc trưng của các lớp chưa thấy thông qua mô tả bằng ngôn ngữ tự nhiên hoặc các đặc trưng khác.

Ví dụ: Một hệ thống phân loại động vật được huấn luyện trên "mèo", "chó", "voi" nhưng có thể nhận diện "hươu cao cổ" dựa trên mô tả bằng ngôn ngữ tự nhiên như “động vật cổ dài, ăn lá cây, sống ở châu Phi”.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/zero_one_few_shot.png" style="width: 700px;"/>

**One-shot learning** yêu cầu mô hình học được khái niệm hoặc nhiệm vụ chỉ từ một ví dụ duy nhất.

Các mô hình one-shot learning cũng cần có tính khái quát hoá tốt, tuy nhiên, vẫn cần một điểm dữ liệu để làm tham chiếu cho các điểm dữ liệu khác sau này.

Ví dụ: Một hệ thống face recognition cần một ảnh đăng ký của Minh, sau đó có thể nhận diện Minh trong các hình ảnh khác sau này dựa trên ảnh đăng ký đó.

**Few-shot learning** mở rộng từ one-shot learning, nơi mô hình cần học một nhiệm vụ với chỉ một số lượng nhỏ mẫu huấn luyện (ví dụ 5 hoặc 10 mẫu mỗi lớp).

Few-shot learning giúp mô hình có thể học được các nhiệm vụ mới với ít dữ liệu hơn so với các phương pháp truyền thống.

Ví dụ: Một mô hình đã được huấn luyện tốt với bài toán phân lớp chó và mèo, có thể chỉ cần một lượng dữ liệu rất nhỏ (few-shot) để học được cách phân loại chính xác chó Poodle, chó Corgi, mèo Sphynx và mèo Ragdoll.

## 4. Các bài toán trong weakly supervised learning

Weakly supervised learning là một nhóm các phương pháp huấn luyện mô hình supervised learning với weakly labeled data.

Hình ảnh dưới đây được lấy từ bài báo [A Brief Introduction to Weakly Supervised Learning](https://www.researchgate.net/publication/319299592_A_Brief_Introduction_to_Weakly_Supervised_Learning) tóm tắt các trường hợp weakly labeled data:
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

Ý tưởng chung của semi supervised learning là tìm giải pháp để sử dụng kết hợp giữa dữ liệu có label và dữ liệu không có label để cải thiện chất lượng của mô hình.

### 5.1. Graph-based Regularization

Graph-based Semi-supervised Learning hoạt động dựa trên giả định cơ bản rằng có thể trích xuất một đồ thị từ tập dữ liệu gốc, trong đó mỗi đỉnh (node) đại diện cho một điểm dữ liệu và mỗi cạnh (edge) thể hiện mức độ tương đồng giữa các cặp đỉnh.
Cụ thể hơn, phương pháp này được xây dựng dựa trên giả định Graph Smoothness Assumption: **"Nếu hai đỉnh gần nhau trên đồ thị (tức là kết nối mạnh), thì chúng nên có vector đại diện tương tự nhau (gần nhau trong không gian vector đại diện)."**

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_graph_based.png" style="width: 900px;"/>

Hình ảnh trên đây được lấy từ bài báo [Semi-Supervised and Unsupervised Deep Visual Learning: A Survey](https://arxiv.org/abs/2208.11296), mô tả cách thức hoạt động của Graph-based Semi-supervised Learning.

Sau khi đưa điểm dữ liệu qua một mô hình mã hoá encoder, ta sẽ thu được các vector đại diện cho các điểm dữ liệu.
Các vector này sẽ được sử dụng để tính toán hàm loss gồm hai thành phần như sau:

$$ L = L_{supervised} + \lambda L_{unsupervised} $$
$$ L_{unsupervised} = \sum_{i,j} w_{ij} ||f(x_i) - f(x_j)||^2 $$

Ở đây $L_{supervised}$ là hàm loss của các điểm dữ liệu có nhãn, $L_{unsupervised}$ là hàm loss của các điểm dữ liệu chưa có nhãn, cụ thể là hàm loss trên đồ thị.
Trong đó:
- $w_{ij}$ là trọng số của cạnh giữa hai đỉnh $i$ và $j$ trong đồ thị.
- $f(x_i)$ và $f(x_j)$ là các vector đại diện của hai điểm dữ liệu tương ứng.

Cụ thể hơn, $w_{ij}$ được tính dựa trên độ tường đồng của hai điểm dữ liệu $i$ và $j$ trong không gian vector đại diện.

| Đỉnh trong đồ thị | Vector được mã hoá | $W_{ij}$ | $\|f(x_i) - f(x_j)\|^2$ | $W_{ij} \|f(x_i) - f(x_j)\|^2$ | Ý nghĩa                                                       |
| ----------------- | ------------------ | -------- | ------------------ | ----------------------------------- | ------------------------------------------------------------- |
| Đỉnh gần nhau     | Vector gần nhau    | Lớn      | Nhỏ                | Nhỏ                                 | Đúng kỳ vọng, mô hình không bị phạt                           |
| Đỉnh gần nhau     | Vector xa nhau     | Lớn      | Lớn                | Lớn                                 | Sai kỳ vọng, mô hình bị phạt mạnh, cần phải ép vector gần lại |
| Đỉnh xa nhau      | Vector gần nhau    | Nhỏ      | Nhỏ                | Rất nhỏ                             | Không ảnh hưởng nhiều đến quá trình huấn luyện                |
| Đỉnh xa nhau      | Vector xa nhau     | Nhỏ      | Lớn                | Nhỏ                                 | Đúng kỳ vọng, mô hình không bị phạt                           |

Cơ chế cốt lõi của các kỹ thuật này là khai thác cấu trúc đồ thị của dữ liệu để học hiệu quả ngay cả khi chỉ có rất ít nhãn, bằng cách lan truyền thông tin từ các đỉnh có nhãn sang các đỉnh chưa có nhãn (label propagation).
Thành phần $L_{unsupervised}$ chỉ đóng vai trò là một regularization term, giúp mô hình khai thác thêm mối quan hệ giữa các điểm dữ liệu chưa có nhãn.

### 5.2. Generative Models

Các mô hình sinh dữ liệu Generative Models là một trong những hướng quan trọng trong semi supervised learning bao gồm cả mô hình Generative Adversarial Networks (GANs) và Variational Autoencoders (VAEs).
Ý tưởng cốt lõi của các phương pháp này là học ra các đặc trưng tiềm ẩn của dữ liệu và mô hình hóa phân bố dữ liệu, từ đó sinh ra dữ liệu mới dựa trên các phân bố đã học được.

Khi áp dụng GANs và VAEs vào semi supervised learning, ta gọi là **semi supervised GANs** và **semi supervised VAEs**.
Nguyên lý nền tảng của phương pháp này là sử dụng lượng dữ liệu không có nhãn để huấn luyện mô hình GANs hoặc VAEs.
Ta kỳ vọng rằng trong quá trình này, mô hình sẽ học được các đặc trưng từ bộ dữ liệu không có nhãn và cải thiện hiệu suất trên bài toán của bộ dữ liệu có nhãn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_gan_based.png" style="width: 900px;"/>

Hình ảnh trên đây được lấy từ bài báo [Semi-Supervised and Unsupervised Deep Visual Learning: A Survey](https://arxiv.org/abs/2208.11296), mô tả cách thức hoạt động của semi supervised GANs.
Thành phần Discriminator của mô hình GANs ở đây bên cạnh việc phân biệt giữa dữ liệu thật từ bộ dữ liệu (kết hợp giữa có nhãn và không có nhãn) và dữ liệu sinh ra từ Generator, còn có nhiệm vụ phân loại các dữ liệu có nhãn thành các lớp khác nhau.

Ngoài ra, có các mô hình GANs khác sử dụng thêm thành phần Encoder để mã hoá dữ liệu đầu vào (kết hợp giữa có nhãn và không có nhãn) thành các feature vector.
Các feature vector này sẽ được sử dụng quá Generator để sinh ra dữ liệu giả.
Discriminator sẽ tiếp tục được huấn luyện để phân biệt giữa dữ liệu thật và dữ liệu giả, đồng thời phân loại các dữ liệu có nhãn thành các lớp khác nhau.

Đối với semi supervised VAEs, ý tưởng tương tự khi sử dụng lượng dữ liệu (kết hợp giữa có nhãn và không có nhãn) để huấn luyện mô hình VAE.
Sau đó tận dụng thành phần Encoder đã được huấn luyện để mã hoá dữ liệu đầu vào thành các feature vector, ta kỳ vọng rằng feature vector từ Encoder này sẽ chứa các thông tin hữu ích để giải quyết bài toán trên dữ liệu có nhãn.

### 5.3. Pseudo Labels

Cả hai phương pháp Graph-based Regularization và Generative Models đều có những vấn đề nhất định.
Phương pháp Graph-based Regularization chỉ mang tính chất regularization cho mô hình, giúp cải thiện một chút về độ đa dạng của dữ liệu mà mô hình được học.
Phương pháp Generative Models yêu cầu quá trình huấn luyện mô hình phức tạp hơn, cần nhiều tài nguyên tính toán hơn và thời gian huấn luyện lâu hơn.

Một phương pháp khác, được sử dụng nhiều hơn trong thực tế là **Pseudo Labels**.
Pseudo Labels là một phương pháp đơn giản và hiệu quả để cải thiện chất lượng của mô hình trong trường hợp có dữ liệu chưa có nhãn.

Ý tưởng chính của Pseudo Labels nằm ở việc huấn luyện mô hình với số lượng ít dữ liệu có label, sau đó sử dụng mô hình này đã được huấn luyện để dự đoán label của số lượng nhiều các dữ liệu chưa có label, các label được dự đoán ra này được gọi là pseudo label.
Cuối cùng, ta kết hợp các dữ liệu có label và các dữ liệu có pseudo label để huấn luyện mô hình mới.

#### Self-training

Self-training là phương pháp mà ta sẽ huấn luyện một mô hình trên tập dữ liệu có label, sau đó được sử dụng để dự đoán label cho tập dữ liệu chưa có label.

Hai hình ảnh dưới đây được lấy từ bài báo [A Survey on Deep Semi-supervised Learning](https://arxiv.org/abs/2103.00550), mô tả cách thức hoạt động của self-training.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_pseudo_labels.png" style="width: 600px;"/>

Trong quá trình sử dụng các pseudo label của self-training, ta cần chú ý đến việc lọc ra những pseudo label có độ tin cậy cao để sử dụng cho việc huấn luyện.

#### Co-training

Co-training là phương pháp mà ta sẽ huấn luyện hai mô hình song song trên hai tập dữ liệu có nhãn khác nhau nhưng có cùng một nhiệm vụ.
Mỗi mô hình sẽ dự đoán label trên hai tập dữ liệu chưa có label để tạo ra pseudo label và các pseudo label này được đổi chéo để huấn luyện mô hình kia.
Việc chia sẻ các dự đoán này với nhau mang lại những góc nhìn khác nhau trên cùng một nhiệm vụ và cải thiện chất lượng của cả hai mô hình.

Hai hình ảnh dưới đây được lấy từ bài báo [A Survey on Deep Semi-supervised Learning](https://arxiv.org/abs/2103.00550), mô tả cách thức hoạt động của self-training.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_co_training.png" style="width: 600px;"/>

Trong co-training, thường ta sẽ sử dụng hai mô hình khác nhau với hai kiến trúc khác nhau để huấn luyện song song.
Co-training là kỹ thuật nền tảng cho các kỹ thuật khác trong semi supervised learning và self supervised learning.

### 5.3. Consistency Regularization và Contrastive Learning

Ngoài các phương pháp trên, ta còn có các phương pháp khác trong semi supervised learning như **Consistency Regularization** và **Contrastive Learning**.
Hai hình ảnh dưới đây được lấy từ bài báo [Semi-Supervised and Unsupervised Deep Visual Learning: A Survey](https://arxiv.org/abs/2208.11296), mô tả cách thức hoạt động của hai phương pháp này.

Consistency Regularization là phương pháp mà ta sẽ huấn luyện mô hình để dự đoán cùng một label cho cùng một dữ liệu đầu vào nhưng với các biến thể khác nhau của nó.
Các biến thể khác nhau này có thể được sinh ra nhờ các kỹ thuật xử lý dữ liệu nhưng cũng có thể được sinh ra nhờ việc thêm noise vào trong mô hình.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_consistency_regularization.png" style="width: 800px;"/>

Contrastive Learning là phương pháp mà ta sẽ huấn luyện mô hình để phân biệt giữa các cặp dữ liệu tương tự và không tương tự.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/semi_contrastive_learning.png" style="width: 800px;"/>

Hai kỹ thuật này, hiện nay được sử dụng rất nhiều trong self supervised learning, ta sẽ tìm hiểu kỹ hơn trong phần giới thiệu về self supervised learning.

## 6. Giới thiệu về self supervised learning - unsupervised learning

Ý tưởng chung của self supervised learning nằm ở việc định nghĩa ra và huấn luyện mô hình theo một bài toán mới từ bộ dữ liệu không có label, sau đó kết hợp với transfer learning để huấn luyện bộ dữ liệu có label trên bài toán gốc ban đầu.

Hình ảnh dưới đây được lấy từ bài báo [Semi-Supervised and Unsupervised Deep Visual Learning: A Survey](https://arxiv.org/abs/2208.11296), mô tả ý tưởng transfer learning trong self supervised learning.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_supervised_learning.png" style="width: 700px;"/>

Hay nói cách khác, self supervised learning là quá trình ta tự định nghĩa bài toán mới, tự sinh ra label cho dữ liệu và huấn luyện mô hình.
Từ đó, ta kỳ vọng rằng mô hình sau quá trình self supervised learning sẽ cung cấp nhiều thông tin hữu ích cho việc huấn luyện mô hình trên bài toán ban đầu.

Vì vậy, điểm quan trọng nhất của self supervised learning chính là việc định nghĩa bài toán mới cho mô hình (pretext task) sao cho nó phù hợp nhất với bài toán gốc (downstream task).

### 6.1. Pretext task với dữ liệu ảnh

#### Autoencoders

Pretext task **Autoencoders** là một trong những phương pháp phổ biến nhất trong self supervised learning với dữ liệu hình ảnh.

Trong phương pháp này, mô hình sẽ được huấn luyện để mã hoá (encode) dữ liệu đầu vào thành một biểu diễn (representation) và sau đó giải mã (decode) biểu diễn này trở lại thành dữ liệu đầu vào ban đầu.
Mục tiêu của mô hình là giảm thiểu sự khác biệt giữa dữ liệu đầu vào và dữ liệu đầu ra sau khi giải mã.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_autoencoders.png" style="width: 700px;"/>

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

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_rotation.png" style="width: 900px;"/>

Hình ảnh trên được lấy từ bài báo [A Survey on Self-Supervised Representation Learning](https://arxiv.org/abs/2308.11455), mô tả cách thức hoạt động của pretext task **Rotation Transformation**.

Mô hình sau khi được huấn luyện trên pretext task này sẽ có khả năng hiểu các đặc trưng không gian của dữ liệu hình ảnh, vì nó đã học được cách phân biệt các góc xoay khác nhau và có thể được sử dụng để huấn luyện các mô hình khác trên các bài toán downstream task.

#### Jigsaw puzzle

Một pretext task khác cũng hướng đến việc hiểu cấu trúc không gian của dữ liệu hình ảnh là **Jigsaw puzzle**.
Trong phương pháp này, dữ liệu đầu vào sẽ được chia thành các mảnh nhỏ (patches) và các mảnh này sẽ được xáo trộn ngẫu nhiên.
Mô hình sẽ được huấn luyện để dự đoán thứ tự của các mảnh này, tức là xác định cách sắp xếp các mảnh để chúng tạo thành một bức tranh hoàn chỉnh.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_jigsaw.png" style="width: 900px;"/>

Hình ảnh trên được lấy từ bài báo [A Survey on Self-Supervised Representation Learning](https://arxiv.org/abs/2308.11455), mô tả cách thức hoạt động của pretext task **Jigsaw puzzle**.

Tương tự như pretext task **Rotation Transformation**, mô hình sau khi được huấn luyện trên pretext task này sẽ có khả năng hiểu cấu trúc không gian của dữ liệu hình ảnh, vì nó đã học được cách phân biệt các mảnh khác nhau và cách sắp xếp chúng và có thể được sử dụng để huấn luyện các mô hình khác trên các bài toán downstream task.

#### Image Transformation

Pretext task **Image Transformation** là một phương pháp self supervised learning với dữ liệu hình ảnh, trong đó, mô hình sẽ được huấn luyện để dự đoán các biến thể của dữ liệu đầu vào.
Pretext task này tương tự với ý tưởng được sử dụng trong **Consistency Regularization** trong semi supervised learning.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_transformations.png" style="width: 800px;"/>

Một số biến thể của dữ liệu ảnh đầu vào được sử dụng là:
- **Gaussian noise**: Thêm nhiễu Gaussian vào ảnh đầu vào.
- **Gaussian blur**: Làm mờ ảnh đầu vào bằng cách sử dụng bộ lọc Gaussian.
- **Color jitter**: Thay đổi độ sáng, độ tương phản, độ bão hòa và sắc độ của ảnh đầu vào.
- **Grayscale**: Chuyển ảnh đầu vào sang ảnh đen trắng.
- **Random crop**: Cắt ngẫu nhiên một phần của ảnh đầu vào.
- **Flip**: Lật ngang ảnh đầu vào.
- **Sobel filter**: Sử dụng bộ lọc Sobel để phát hiện cạnh trong ảnh đầu vào.

Các kỹ thuật xử lý ảnh này khá giống với các kỹ thuật được sử dụng trong **Data Augmentation** trong quá trình huấn luyện mô hình với dữ liệu có nhãn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_transformations_train.png" style="width: 600px;"/>

Hai hình ảnh trên được lấy từ bài báo [A Survey on Self-Supervised Representation Learning](https://arxiv.org/abs/2308.11455), mô tả cách thức hoạt động của pretext task **Image Transformation**.
Trong sơ đồ trên:
- Mỗi biến thể của ảnh được gọi là một "view" của ảnh gốc.
- Các view khác nhau được mã hoá thành các vector đại diện khác nhau.
- Các vector đại diện này được "project" vào một không gian, tạo ra các vector đại diện mới.
- Các vector đại diện mới này sẽ được tính toán độ tương quan với nhau.
Độ tương quan này được sử dụng như một hàm loss để huấn luyện mô hình.

#### Teacher - Student

Một nghiên cứu khá đột phá trong self supervised learning là mô hình DINO được giới thiệu trong bài báo [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) sử dụng kiến trúc **Teacher - Student**.

Kiến trúc này bao gồm hai mô hình: một mô hình Teacher và một mô hình Student.
Hai mô hình Teacher và Student này có cùng kiến trúc mô hình nhưng sử dụng bộ trọng số khác nhau.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_dino_teacher_student.png" style="width: 600px;"/>

Hình ảnh trên được lấy từ bài báo [A Survey on Self-Supervised Representation Learning](https://arxiv.org/abs/2308.11455), mô tả cách thức chuẩn bị dữ liệu cho mô hình Teacher và Student trong DINO.
Mô hình Teacher sẽ được học trên hai miếng crop lớn (với tổng diện tích trên 50% diện tích của dữ liệu đầu vào) của mỗi dữ liệu đầu vào, trong khi mô hình Student sẽ được học trên các miếng crop nhỏ hơn (với tổng diện tích dưới 50% diện tích của dữ liệu đầu vào) của cùng một dữ liệu đầu vào.
Điều này mang lại nhiều thông tin của ảnh đầu vào hơn cho mô hình Teacher, trong khi mô hình Student sẽ học được các đặc trưng chi tiết và nhỏ hơn của ảnh đầu vào.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_dino.png" style="width: 600px;"/>

Hình ảnh trên được lấy từ bài báo [A Survey on Self-Supervised Representation Learning](https://arxiv.org/abs/2308.11455), mô tả cách thức huấn luyện mô hình Teacher và Student trong DINO.

- Các view lớn và nhỏ sẽ tương ứng được đưa qua các mô hình Teacher và Student, được project vào một không gian vector và được tính softmax trên các vector này.
- DINO sử dụng một hàm loss tương tự như Cross Entropy để huấn luyện mô hình Student dựa trên đầu ra của mô hình Teacher.
- Mô hình Student được cập nhật trọng số dựa trên gradient của hàm loss trên.
- Mô hình Teacher được cập nhật trọng số bằng cách sử dụng một hàm moving average của trọng số của mô hình Student, giúp mô hình cả hai ổn định hơn trong quá trình huấn luyện, nói cách khác, hai mô hình Teacher và Student sẽ có trọng số gần giống nhau, "luôn nhìn về cùng một hướng" trong quá trình huấn luyện.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/self_dino_results.png" style="width: 900px;"/>

Hình ảnh trên được lấy từ bài báo [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), mô tả kết quả của mô hình DINO.
Kết quả cho thấy mô hình DINO có thể học được các đặc trưng của ảnh đầu vào mà không cần sử dụng bất kỳ nhãn nào.
Không những thế, kết quả của DINO có thể được sử dụng để phục vụ cho các bài toán đòi hỏi dữ liệu có nhãn phức tạp như object detection hay instance segmentation.

### 6.2. Pretext task với dữ liệu text

#### Masked language modelling

Phương pháp **Masked language modelling** là phương pháp, trong đó, một số từ trong câu bị che đi (mask) và mô hình phải đoán những từ này dựa vào cả ngữ cảnh bên trái lẫn phải của từ đó.

Đây là một trong những phương pháp phổ biến nhất trong self supervised learning với dữ liệu văn bản, được sử dụng trong các mô hình như BERT.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/bert.png" style="width: 900px;"/>

Xét ví dụ trên hình được lấy từ bài báo ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805), một cặp câu A và B được đưa vào mô hình BERT. Để phục vụ cho bài toán **Masked Language Model**:
- Trên hai câu này, một lượng khoảng 15% số lượng token đã được che đi (mask).
- Ví dụ: 80% lượng dữ liệu sẽ được mask với dạng "my dog is hairy" thành "my dog is [MASK]", 10% sẽ được thay thế bằng một token ngẫu nhiên khác "my dog is hairy" thành "my dog is apple", và 10% sẽ giữ nguyên token gốc "my dog is hairy" là "my dog is hairy".
- Các token đã bị che trong cặp câu trên sau khi được đưa vào mô hình sẽ được đi qua lớp softmax để dự đoán token gốc của chúng.

#### Standard language modelling

Phương pháp **Standard Language Modeling** là phương pháp yêu cầu mô hình dự đoán từ tiếp theo trong chuỗi văn bản dựa trên các từ trước đó.

Hình ảnh dưới đây được lấy từ bài báo ["Improving Language Understanding by Generative Pre-Training"](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), mô tả cách thức huấn luyện **Standard Language Modeling**.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/gpt.png" style="width: 900px;"/>

Đối với bài toán **text classification** (phân lớp văn bản), mô hình GPT sẽ đơn giản là nhận đầu vào là một chuỗi văn bản, sau đó đưa đầu ra qua một lớp softmax để dự đoán nhãn của văn bản.

Đối với bài toán **text entailment** (suy diễn mối quan hệ văn bản), mô hình GPT sẽ nhận đầu vào câu tiền đề (premise) được nối với câu giả thuyết (hypothesis) và đưa ra đầu ra là xác suất của các nhãn như "entailment" (suy diễn), "contradiction" (mâu thuẫn) và "neutral" (trung lập).

Đối với bài toán **text similarity** (tương đồng văn bản), mô hình GPT sẽ nhận đầu vào là hai câu và nối hai câu này với thứ tự ngược nhau ("sentence A" + "sentence B" và "sentence B" + "sentence A") và đưa ra đầu ra là xác suất của các nhãn như "similar" (tương đồng) và "dissimilar" (không tương đồng).
Phương pháp này được áp dụng tương tự cho bài toán **Multiple-Choice Question Answering** (trả lời câu hỏi nhiều lựa chọn) bằng cách nối câu hỏi với các lựa chọn trả lời và đưa ra xác suất cho từng lựa chọn.

#### Next Sentence Prediction

Phương pháp **Next Sentence Prediction** là phương pháp, trong đó, mô hình sẽ được huấn luyện để dự đoán xem một câu có phải là câu tiếp theo của một câu khác hay không.

Đây là một trong những phương pháp phổ biến nhất trong self supervised learning với dữ liệu văn bản, được sử dụng trong các mô hình như BERT.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/bert.png" style="width: 900px;"/>

Xét ví dụ trên hình được lấy từ bài báo ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805), một cặp câu A và B được đưa vào mô hình BERT. Để phục vụ cho bài toán **Next Sentence Prediction**:
- Input của mô hình sẽ được thêm một token đặc biệt [CLS] ở đầu câu A và nhiệm vụ của mô hình là dự đoán giá trị của token này là 1 (IsNext) hoặc 0 (NotNext) để xác định xem câu B có phải là câu tiếp theo của câu A hay không.
- Ví dụ: "[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]" sẽ có label của token [CLS] là IsNext.

## 7. Contrastive Language-Image Pre-Training (CLIP)

Contrastive Language-Image Pre-Training (CLIP) là một mô hình được giới thiệu bởi OpenAI trong bài báo ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020).

CLIP đề xuất một phương pháp huấn luyện mô hình Computer vision sử dụng dữ liệu ngôn ngữ tự nhiên (text) thay vì các nhãn thủ công truyền thống.
CLIP học cách liên kết hình ảnh và văn bản thông qua **contrastive learning**, sao cho vector đại diện của ảnh và vector đại diện của mô tả văn bản tương ứng sẽ gần nhau trong không gian đặc trưng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/5-transfer-weakly-semi-self-supervised-learning/clip.png" style="width: 1000px;"/>

Cụ thể hơn, CLIP gồm hai thành phần chính:
- **Image Encoder**: Mã hoá hình ảnh thành vector đặc trưng, có thể sử dụng các mô hình ResNet hoặc Vision Transformer (ViT).
- **Text Encoder**: Mã hoá mô tả văn bản thành vector đặc trưng, thường sử dụng mô hình Transformer.

CLIP được huấn luyện trên bộ dữ liệu được lấy từ Internet, bao gồm hàng trăm triệu cặp hình ảnh và mô tả văn bản, mà không qua bước gán nhãn thủ công.

CLIP sử dụng hàm loss **contrastive loss** để tối ưu hoá khoảng cách giữa các vector đặc trưng của hình ảnh và văn bản tương ứng, trong khi đẩy các vector đặc trưng của hình ảnh và văn bản không tương ứng ra xa nhau.

Mô hình CLIP được pretrained có thể đạt kết quả tốt với zero-shot trên các bộ dữ liệu phân loại hình ảnh.
Điều này thể hiện khả năng khái quát hoá mạnh và có tính chuyển giao cao (transferability).
Từ đó, cả phần **Image Encoder** và **Text Encoder** của CLIP đều có thể được sử dụng như một công cụ mã hoá hình ảnh và văn bản cho các tác vụ khác nhau trong Computer Vision và Natural Language Processing.

Một số nâng cấp đáng chú ý đến của CLIP bao gồm:
- **OpenCLIP**: Một phiên bản mở rộng của CLIP với khả năng huấn luyện trên các tập dữ liệu lớn hơn và cải thiện hiệu suất.
- **BLIP/BLIP-2**: Các mô hình kết hợp giữa CLIP và các phương pháp tự động sinh văn bản, cho phép mô hình không chỉ phân loại mà còn mô tả hình ảnh một cách tự nhiên.
