---
time: 02/26/2023
title: Cơ chế Attention Attention Mechanism
description: Attention là cơ chế giúp mô hình học sâu tập trung (attend) vào các thành phần quan trọng trong dữ liệu đầu vào, tương tự như con người chú ý đến chi tiết nổi bật. Cơ chế này ra đời (2014) nhằm khắc phục hạn chế của RNN/LSTM cũ khi phải mã hóa toàn bộ chuỗi vào một vector cố định. Theo kết quả nghiên cứu, Transformer (2017) – kiến trúc dùng hoàn toàn attention – đã trở thành nền tảng của các mô hình ngôn ngữ lớn (LLMs) như GPT, BERT hiện đại. Trong thị giác máy tính, cơ chế Attention cũng được áp dụng thành công qua Vision Transformer (ViT) và các mô hình DETR.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/4-attention-mechanism/banner.png
tags: [deep-learning, natural-language-processing, computer-vision]
is_highlight: true
is_published: true
---

## 1. Mô hình Seq2Seq truyền thống

Mô hình Encoder-Decoder (Seq2Seq) bao gồm hai thành phần chính: một bộ mã hoá (Encoder) và một bộ giải mã (Decoder).

Trong bài toán xử lý ngôn ngữ tự nhiên, cả Encoder và Decoder thường là mạng RNN/LSTM/GRU, trong đó:
- Encoder nhận chuỗi đầu vào và chuyển thành một vector ngữ cảnh tổng quát.
- Decoder lấy vector này và tạo từng phần tử chuỗi đầu ra mới.

Điểm đặc biệt của mô hình này là nó có thể xử lý các bài toán dịch máy machine translation, tóm tắt văn bản text summarization, hỏi đáp question answering, nơi độ dài chuỗi đầu vào và đầu ra có thể khác nhau.
Mô hình Seq2Seq đã mang lại kết quả ấn tượng trong các tác vụ NLP này.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/4-attention-mechanism/seq2seq.png" style="width: 1000px;"/>

Trong thực tế, kiến trúc thường bao gồm:
- Encoder: xử lý chuỗi đầu vào và sinh ra trạng thái ẩn cuối cùng (vector ngữ cảnh) chứa thông tin chung về toàn bộ đầu vào.
- Decoder: hoạt động như một mô hình ngôn ngữ điều kiện, lấy vector ngữ cảnh cộng thêm đầu ra trước đó để dự đoán phần tử tiếp theo của chuỗi đầu ra.

Mô hình Seq2Seq cổ điển có hạn chế là phải nén toàn bộ thông tin vào một vector cố định, khiến nó gặp khó khăn với các câu quá dài.
Đặc biệt, khi chuỗi đầu vào dài, vector ngữ cảnh có thể không đủ để lưu trữ tất cả thông tin cần thiết, dẫn đến mất mát ngữ nghĩa và giảm hiệu suất.

Đây cũng chính là vấn đề phụ thuộc xa long-term dependencies đã được đề cập trong bài viết về RNN/LSTM.

## 2. Giới thiệu chung về Attention Mechanism

Vấn đề mà Attention đặt ra để giải quyết là làm thế nào để mô hình học sâu có thể tập trung vào các phần quan trọng của dữ liệu đầu vào, tương tự như cách con người chú ý đến các chi tiết nổi bật trong một bức tranh hoặc một đoạn văn.

Trong các bài toán xử lý ngôn ngữ tự nhiên, Attention cho phép mô hình học được mối quan hệ giữa các từ trong câu mà không cần phải tuần tự qua từng từ, từ đó cải thiện khả năng hiểu ngữ nghĩa và ngữ cảnh của mô hình.

Hơn nữa, Attention cũng giúp mô hình vượt qua giới hạn về vector ngữ cảnh có độ dài cố định, từ đó, xử lý các chuỗi đầu vào dài mà không gặp phải vấn đề mất mát thông tin như trong mô hình Seq2Seq truyền thống.

Ngoài ra, trong thị giác máy tính, các mô hình CNN sử dụng kernel tập trung vào việc trích xuất các đặc trưng cục bộ mà không thể nắm bắt được mối quan hệ toàn cục giữa các phần của hình ảnh.
Điều này dẫn đến việc mô hình không thể hiểu được ngữ cảnh tổng thể của hình ảnh.

Ưu điểm chính của Attention là không hạn chế vị trí (toàn cục), nắm bắt phụ thuộc dài hạn. Ví dụ, Attention có thể xem xét toàn bộ chuỗi cùng lúc (thay vì tuần tự như RNN), và không hạn chế phạm vi cục bộ như CNN.
Cơ chế này cũng cho phép xử lý song song (parallelization), tận dụng được khả năng tính toán đồng thời trên GPU, cho phép huấn luyện nhanh và hiệu quả hơn.

Hình ảnh dưới đây được lấy từ bài báo [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) của Bahdanau et al. vào năm 2014 thể hiện kết quả của phép Attention.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/4-attention-mechanism/attention_example.png" style="width: 900px;"/>

Tuy nhiên, hạn chế lớn nhất của Attention là chi phí tính toán cao, đặc biệt là với chuỗi đầu vào dài hoặc hình ảnh lớn.
Cụ thể, với một chuỗi đầu vào có độ dài $n$, chi phí tính toán của Attention là $O(n^2)$ do phải tính toán độ tương đồng giữa tất cả các cặp từ trong chuỗi.

## 2. Cơ chế Attention cơ bản

Cơ chế Attention làm việc trên đơn vị token, nghĩa là ta phải thực hiện tokenization trên câu đầu vào trước khi áp dụng Attention hoặc phải chia hình ảnh thành các patch nhỏ, mỗi patch là một token.

Từ phần này, ta sẽ sử dụng khái niệm token để mô tả cách thức hoạt động của Attention.
Phiên bản Attention ở đây được gọi là **Dot-Product Attention**, được giới thiệu trong bài báo [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) của Luong et al. vào năm 2015.

### 2.1. Khái niệm Query - Key - Value

Trong cơ chế Attention, mỗi token trong input đầu vào sẽ được ánh xạ thành một vector, bao gồm ba thành phần chính: Query, Key và Value.
- **Query**: Là vector đại diện cho token hiện tại mà mô hình đang muốn tìm hiểu mối quan hệ với các token khác.
- **Key**: Là vector đại diện cho các token trong input đầu vào mà mô hình sẽ so sánh với Query để xác định mức độ liên quan.
- **Value**: Là vector chứa thông tin của token thực sự được sử dụng dựa vào tương quan giữa Query và Key.

**Ví dụ:**
Giả sử, ta có một token dạng text là "dog" và một hình ảnh chứa ba đối tượng là "dog", "cat", "bird" (ở đây, ta tạm coi mỗi đối tượng là một token).
- Query của token "dog" dạng text là vector $Q$.
- Key của token "dog", "cat", "bird" trong hình ảnh lần lượt là vector $K_{dog}$, $K_{cat}$, $K_{bird}$.
- Value của token "dog", "cat", "bird" trong hình ảnh lần lượt là vector $V_{dog}$, $V_{cat}$, $V_{bird}$.

Ta sẽ lần lượt tính toán mối quan hệ giữa $Q$ và $K_{dog}$, $K_{cat}$, $K_{bird}$ và chuẩn hoá chúng về dạng xác suất để tạo ra trọng số về mức độ quan trọng của từng token trong hình ảnh đối với token "dog" dạng text.

Ta kỳ vọng rằng, với một mô hình attention tốt, trọng số sẽ cao nhất với token "dog" trong hình ảnh, thấp hơn với "cat" và thấp nhất với "bird".

### 2.2. Khởi tạo Query, Key, Value

Để khởi tạo Query, Key và Value, ta thường sử dụng các ma trận trọng số để ánh xạ từ đầu vào (input) sang các vector này.
- **Query**: $Q = XW_Q$
- **Key**: $K = XW_K$
- **Value**: $V = XW_V$

Trong đó:
- $X$ là ma trận đầu vào (có thể là embedding của các token).
- $W_Q$, $W_K$, $W_V$ là các ma trận trọng số được học trong quá trình huấn luyện.
- $Q$, $K$, $V$ là Query, Key và Value tương ứng với đầu vào $X$.

Trong quá trình huấn luyện, các ma trận trọng số này sẽ được cập nhật để tối ưu hoá khả năng của mô hình trong việc xác định mối quan hệ giữa các token.

### 2.3. Công thức tính toán Attention

Xét ví dụ gồm một token dạng text là "dog" và một hình ảnh chứa ba đối tượng là "dog", "cat", "bird" (ở đây, ta tạm coi mỗi đối tượng là một token).
- Query của token "dog" dạng text là vector $Q$ có độ dài $d_k$.
- Key của token "dog", "cat", "bird" trong hình ảnh lần lượt là vector $K_{dog}$, $K_{cat}$, $K_{bird}$, mỗi vector có độ dài $d_k$.
- Value của token "dog", "cat", "bird" trong hình ảnh lần lượt là vector $V_{dog}$, $V_{cat}$, $V_{bird}$, mỗi vector có độ dài $d_v$.

**1. Tính độ tương đồng giữa Query và các Key:**
Để tính độ tương đồng giữa Query và các Key, ta sử dụng dot product để đo lường mức độ liên quan giữa chúng.
Công thức tính toán là:

$$ \text{score}(Q, K_i) = Q \cdot K_i^T $$

Với ba Key của các token trong hình ảnh trong ví dụ trên, ta sẽ có ba giá trị độ tương đồng:
- $\text{score}(Q, K_{dog})$ thể hiện mức độ liên quan giữa Query của token "dog" dạng text và Key của token "dog" trong hình ảnh.
- $\text{score}(Q, K_{cat})$ thể hiện mức độ liên quan giữa Query của token "dog" dạng text và Key của token "cat" trong hình ảnh.
- $\text{score}(Q, K_{bird})$ thể hiện mức độ liên quan giữa Query của token "dog" dạng text và Key của token "bird" trong hình ảnh.

**2. Áp dụng hàm softmax để tính toán trọng số:**

Sau khi tính được độ tương đồng, ta sẽ áp dụng hàm softmax để chuẩn hoá các giá trị này về dạng xác suất, tạo ra trọng số cho từng Key:

$$ \alpha_i = \text{softmax}(\text{score}(Q, K_i)) $$

Trong đó, $\alpha_i$ là trọng số tương ứng với Key $K_i$.

Với ba Key của các token trong hình ảnh trong ví dụ trên, ta sẽ có ba trọng số:
- $\alpha_{dog}$ là trọng số của Key "dog" trong hình ảnh với Query "dog" dạng text.
- $\alpha_{cat}$ là trọng số của Key "cat" trong hình ảnh với Query "dog" dạng text.
- $\alpha_{bird}$ là trọng số của Key "bird" trong hình ảnh với Query "dog" dạng text.

**3. Kết hợp các Value dựa trên trọng số:**

Sau khi có trọng số, ta sẽ kết hợp các Value tương ứng với các Key để tạo ra đầu ra của cơ chế Attention:

$$ \text{Attention}(Q, K, V) = \sum_{i} \alpha_i V_i $$

Ở đây, ta hiểu kết quả của Attention là một vector mới đại diện cho token "dog" dạng text, được tính bằng cách kết hợp các Value của các token trong hình ảnh dựa trên trọng số của mối quan hệ.

## 3. Một số phiên bản của cơ chế Attention

### 3.1. Additive Attention

Additive Attention (còn gọi là Bahdanau Attention) là cơ chế attention được giới thiệu trong bài báo [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) của Bahdanau et al. vào năm 2014 và là phiên bản đầu tiên của cơ chế Attention.

Ở phiên bản này, thay vì sử dụng dot product để tính độ tương đồng giữa Query và Key, ta sẽ sử dụng một hàm phi tuyến tính (thường là một mạng nơ-ron đơn giản) để kết hợp Query và Key.

$$ \text{score}(Q, K_i) = v_a^T \tanh(W_a Q + U_a K_i) $$

Trong đó:
- $v_a$, $W_a$, $U_a$ là các ma trận trọng số được học trong quá trình huấn luyện.
- $\tanh$ là hàm kích hoạt phi tuyến tính.

Hàm softmax vẫn được sử dụng để chuẩn hoá các giá trị độ tương đồng thành trọng số, và quá trình kết hợp Value cũng tương tự như trong cơ chế Attention cơ bản.

Phiên bản này phù hợp với mô hình nhỏ, chuỗi ngắn, nơi độ phi tuyến giúp tăng khả năng biểu diễn.
Tuy nhiên, nó có chi phí tính toán cao hơn do sử dụng nhiều phép biến đổi tuyến tính và hàm phi tuyến.

### 3.2. Scaled Dot-Product Attention

Phiên bản Dot-Product Attention, khi độ dài của Query và Key tăng lên, giá trị dot product có thể trở nên quá lớn trước khi áp dụng hàm softmax, dẫn đến gradient vanishing hoặc exploding trong quá trình huấn luyện.

Để giải quyết vấn đề này, Scaled Dot-Product Attention được giới thiệu được giới thiệu trong bài báo [Attention Is All You Need](https://arxiv.org/abs/1706.03762) của Vaswani et al. vào năm 2017.
Hình ảnh dưới đây được lấy từ bài báo này mô tả cơ chế Scaled Dot-Product Attention.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/4-attention-mechanism/scaled_dot_product_attention.png" style="width: 250px;"/>

Trong phiên bản này, giá trị dot product giữa Query và Key sẽ được chia cho căn bậc hai của kích thước của Key để giảm thiểu ảnh hưởng của độ dài:

$$ \text{score}(Q, K_i) = \frac{Q \cdot K_i^T}{\sqrt{d_k}} $$

Trong đó, $d_k$ là kích thước của Key.

Hàm softmax vẫn được sử dụng để chuẩn hoá các giá trị độ tương đồng thành trọng số, và quá trình kết hợp Value cũng tương tự như trong cơ chế Attention cơ bản.

Phiên bản này giúp cải thiện độ ổn định của quá trình huấn luyện và cho phép mô hình học được các mối quan hệ phức tạp hơn giữa các token.

### 3.3. Multi-Head Attention

Multi-Head Attention là một phiên bản mở rộng của cơ chế Attention, cho phép mô hình học được nhiều mối quan hệ khác nhau giữa các token bằng cách sử dụng nhiều "đầu" Attention song song.
Multi-Head Attention được giới thiệu được giới thiệu trong bài báo [Attention Is All You Need](https://arxiv.org/abs/1706.03762) của Vaswani et al. vào năm 2017.
Hình ảnh dưới đây được lấy từ bài báo này mô tả cơ chế Multi-Head Attention.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/4-attention-mechanism/multi_head_attention.png" style="width: 400px;"/>

Thay vì chỉ có một phép attention, Transformer sử dụng đa đầu (Multi-Head): chia đầu vào thành $h$ “head” con, mỗi head học ba ma trận trọng số $W_i^Q, W_i^K, W_i^V$ riêng. Mỗi head tính attention độc lập:

$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

Sau đó, các đầu này được kết hợp lại bằng cách nối (concatenate) và áp dụng một ma trận trọng số cuối cùng:

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O $$

Multi-Head Attention cũng là cách để tăng độ phức tạp của mô hình Attention từ đó cho phép mô hình học được nhiều mối quan hệ khác nhau giữa các token, từ đó cải thiện khả năng biểu diễn và hiểu ngữ cảnh của mô hình nhưng cũng làm tăng chi phí tính toán do phải tính toán nhiều phép attention song song.

Multi-Head Attention là một phần rất quan trọng trong kiến trúc mô hình Transformer.

### 3.4. Self-Attention

Self-Attention là một phiên bản đặc biệt của cơ chế Attention, trong đó Query, Key và Value đều được lấy từ cùng một input đầu vào.
Từ đó, Self-Attention cho phép mô hình học được mối quan hệ giữa các token của chính input đó để tạo ra một biểu diễn ngữ nghĩa tốt hơn cho mỗi token của input đó.
Self-Attention được giới thiệu trong bài báo [Attention Is All You Need](https://arxiv.org/abs/1706.03762) của Vaswani et al. vào năm 2017.
Hình ảnh dưới đây được lấy từ bài báo này mô tả kết quả của một lớp Self-Attention.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/4-attention-mechanism/self_attention.png" style="width: 800px;"/>

Ví dụ: Xét một câu "The rabbit is running on the grass". Giả sử mỗi từ trong câu là một token.
Ta có thể sử dụng Self-Attention để tính toán mối quan hệ đôi một giữa các từ trong câu này.

Query của từ "rabbit" sẽ được so sánh với Key của tất cả các từ khác trong câu để xác định mức độ liên quan.
Từ đó, ta có thể xác định được từ nào là quan trọng nhất trong ngữ cảnh của từ "rabbit".

Sau khi đưa vector embedding của các token của câu trên qua Self-Attention, ta sẽ thu được một biểu diễn mới cho mỗi token dựa trên mối quan hệ của token đó với các token khác trong câu.
Tổng quan, các biểu diễn mới của từng token này sẽ thể hiện ngữ cảnh của câu một cách tốt hơn.

### 3.6. Sparse, Local và Global Attention

Với chuỗi rất dài (hàng nghìn token), Attention chuẩn tốn $O(n^2)$ bộ nhớ và thời gian, nên các biến thể sparse/local được đề xuất.

Hình ảnh này được lấy từ bài báo [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) mô tả cơ chế Local Attention.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/4-attention-mechanism/local_attention.png" style="width: 600px;"/>

- **Local (sliding window) attention:**
Mỗi token chỉ attend tới một cửa sổ giới hạn quanh nó (ví dụ 128 token trước và sau) thay vì toàn bộ chuỗi. Điều này giúp giảm chi phí tính toán và bộ nhớ, nhưng vẫn giữ được mối quan hệ cục bộ.
- **Global attention:**
Một số token quan trọng (như [CLS] được sử dụng trong bài toán text classification hoặc các token đặc biệt) có thể attend tới cả chuỗi, giữ tính bao phủ thông tin toàn cục.
- **Sparse attention:**
Mô hình nhóm các token thành các block và chỉ cho phép attend trong block hoặc vài block lân cận. Nhờ vậy giảm tính liên kết toàn phần nhưng vẫn giữ khả năng tương tác ở mức gần.

Hình ảnh này được lấy từ bài báo [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) mô tả cơ chế Global Attention.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/4-attention-mechanism/global_attention.png" style="width: 600px;"/>

Các cơ chế trên hướng đến mục tiêu cân bằng giữa hiệu quả tính toán và khả năng mô hình hóa phụ thuộc dài hạn.
Các kỹ thuật này giảm chi phí so với Transformer gốc, nhưng có thể hy sinh một phần khả năng phát hiện phụ thuộc xa (tuy nhiên thường chấp nhận được trong thực tế).
Ví dụ, Longformer có thể xử lý văn bản dài hàng nghìn từ với bộ nhớ thấp hơn, nhờ tính chất local+global attention.

### 3.7. Cross-Attention trong mô hình Encoder-Decoder

Ý tưởng của Cross-Attention được giới thiệu đầu tiên từ bài báo [Attention Is All You Need](https://arxiv.org/abs/1706.03762) của Vaswani et al. vào năm 2017 nhưng không được gọi là Cross-Attention.
Hình ảnh dưới đây được lấy từ bài báo này mô tả cách thức hoạt động của Cross-Attention.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/4-attention-mechanism/cross_attention.png" style="width: 600px;"/>

Cái tên Cross-Attention được sử dụng bắt đầu từ hai bài báo [Cross‑Attention Multi‑Scale Vision Transformer](https://arxiv.org/abs/2103.14899) và [CAT: Cross Attention in Vision Transformer](https://arxiv.org/abs/2106.05786) khi các tác giả nghiên cứu cách sử dụng Cross-Attention trong mô hình Vision Transformer kết hợp giữa dữ liệu hình ảnh và văn bản.

Trong mô hình Encoder-Decoder, Cross-Attention là cơ chế cho phép Decoder attend đến các thông tin từ cả Encoder lẫn các bước đầu ra trước đó của Decoder.
Ở đây Query lấy từ Decoder (hoặc từ embedding của bước đầu ra trước), còn Key/Value lấy từ đầu ra của Encoder và các bước đầu ra trước đó của Decoder.
Điều này cho phép Decoder khi sinh từ tại vị trí $i$ có thể “hỏi” thông tin từ toàn bộ chuỗi đầu vào đã mã hóa.

### 3.8. Flash Attention

FlashAttention là một kỹ thuật tối ưu hóa trong mô hình Transformer, được thiết kế để tăng tốc tính toán attention và giảm sử dụng bộ nhớ, đặc biệt là trong các mô hình lớn như GPT, BERT, hay ViT.

Thuật toán này được giới thiệu trong bài báo [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) của Dao et al. vào năm 2022 và phiên bản mới nhất là FlashAttention-2 được giới thiệu trong bài báo [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) của Dao et al. vào năm 2023.
Hình ảnh dưới đây được lấy từ bài báo mô tả cách thức hoạt động của Flash Attention.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/4-attention-mechanism/flash_attention.png" style="width: 1000px;"/>

FlashAttention giải quyết hai vấn đề lớn trong các tính toán attention tiêu chuẩn là Chiếm nhiều bộ nhớ và Chậm khi chuỗi dài.
Cả hai vấn đề này đều gây ra sự tốn kém về tài nguyên tính toán.

FlashAttention được thiết kế dựa trên ba ý tưởng chính:
- **Tính attention một cách "chính xác" nhưng theo khối (block-wise):**
Thay vì tính toàn bộ ma trận attention cùng lúc, thuật toán chia nhỏ thành từng khối (blocks) và tính attention trực tiếp từ Q, K, V mà không lưu toàn bộ ma trận trung gian trong bộ nhớ GPU.
- **IO-aware computation:**
Nó được tối ưu để tối thiểu hóa chi phí đọc/ghi dữ liệu từ DRAM, tận dụng tối đa SRAM (on-chip memory). Vì vậy, tốc độ tính toán tăng lên đáng kể, nhất là trên GPU.
- **Sử dụng tính chất của softmax để tính toán tích lũy (online softmax):**
Thay vì lưu trữ toàn bộ đầu ra để chuẩn hóa sau, FlashAttention chuẩn hóa softmax trực tuyến trong quá trình tính toán, tránh việc lưu trữ ma trận lớn.

FlashAttention giúp:
- Tăng tốc đáng kể từ 2 đến 4 lần so với attention truyền thống.
- Giảm sử dụng bộ nhớ lên đến 10 lần và cho phép training các mô hình lớn hơn hoặc sử dụng batch size lớn hơn.
- Vẫn đảm bảo độ chính xác tuyệt đối (không phải phương pháp xấp xỉ).

FlashAttention đã được tích hợp vào nhiều framework lớn như: Hugging Face Transformers và PyTorch (thông qua xformers hoặc flash-attn library)
