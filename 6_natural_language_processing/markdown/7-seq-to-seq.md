---
time: 06/24/2022
title: Bài toán Sequence-to-Sequence
description: Sequence-to-Sequence (Seq2Seq) là họ mô hình ánh xạ một chuỗi đầu vào sang một chuỗi đầu ra có độ dài khác nhau. Đây là nền tảng của nhiều bài toán NLP quan trọng như dịch máy, tóm tắt văn bản, hỏi đáp và sinh văn bản. Bài giảng này trình bày các phương pháp chính và các chỉ số đánh giá được sử dụng rộng rãi trong cộng đồng nghiên cứu.
banner_url:
tags: [deep-learning, natural-language-processing]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về bài toán Sequence-to-Sequence

Bài toán Sequence-to-Sequence (Seq2Seq) là nhiệm vụ xây dựng mô hình có khả năng chuyển đổi một **chuỗi đầu vào** $(x_1, x_2, \ldots, x_T)$ thành một **chuỗi đầu ra** $(y_1, y_2, \ldots, y_{T'})$ trong đó $T$ và $T'$ có thể khác nhau — đây là điểm khác biệt cơ bản so với các mô hình phân loại hay dự đoán điểm cố định.

Các bài toán tiêu biểu trong nhóm Seq2Seq:
- **Dịch máy (Machine Translation):** "Tôi yêu Việt Nam" → "I love Vietnam"
- **Tóm tắt văn bản (Text Summarization):** Bài báo dài → Đoạn tóm tắt ngắn
- **Hỏi đáp (Question Answering):** Câu hỏi + Ngữ cảnh → Câu trả lời
- **Image Captioning:** Hình ảnh → Câu mô tả (xem [bài giảng Image Captioning](/blog/image-captioning))
- **Sinh mã nguồn (Code Generation):** Mô tả bằng ngôn ngữ tự nhiên → Đoạn code

Điểm chung của tất cả các bài toán này: đầu ra là một **chuỗi token** được sinh ra **tự hồi quy (autoregressive)** — mỗi token được sinh dựa trên các token đã sinh trước đó và thông tin từ chuỗi đầu vào.

## 2. Nhóm các phương pháp giải bài toán Seq2Seq

### 2.1. Mô hình RNN Encoder-Decoder

Kiến trúc Encoder-Decoder với RNN (LSTM/GRU) là nền móng của Seq2Seq hiện đại, được giới thiệu trong bài báo [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) (Sutskever et al., 2014).

Chi tiết về kiến trúc RNN và LSTM đã được trình bày trong [bài giảng Recurrent Neural Network](/blog/recurrent-neural-network).

**Ý tưởng:** Encoder đọc toàn bộ chuỗi đầu vào và nén thành một **context vector** cố định. Decoder nhận context vector và sinh ra chuỗi đầu ra từng token một.

**Hạn chế cốt lõi:** Context vector cố định không thể chứa đầy đủ thông tin khi chuỗi đầu vào dài — đây là "nút cổ chai thông tin" (information bottleneck).

### 2.2. Mô hình Attention-based

Cơ chế Attention khắc phục nút cổ chai bằng cách cho Decoder "nhìn lại" toàn bộ chuỗi encoder states tại mỗi bước sinh.

Chi tiết về cơ chế Attention đã được trình bày trong [bài giảng Attention Mechanism](/blog/co-che-attention-attention-mechanism).

**Ưu điểm cốt lõi:** Decoder có thể tập trung vào phần liên quan nhất của chuỗi đầu vào khi sinh mỗi token — ví dụ khi dịch "con mèo", mô hình tập trung vào token "cat" trong câu tiếng Anh gốc.

**Mô hình tiêu biểu:**
- **Bahdanau Attention (Bahdanau et al., 2015)** — [paper](https://arxiv.org/abs/1409.0473) — additive attention, bài báo khai sinh cơ chế attention trong NLP.
- **Luong Attention (Luong et al., 2015)** — [paper](https://arxiv.org/abs/1508.04025) — multiplicative (dot-product) attention, đơn giản và hiệu quả hơn.

### 2.3. Mô hình Transformer-based

Transformer thay thế hoàn toàn RNN bằng Self-Attention, cho phép xử lý song song toàn bộ chuỗi thay vì tuần tự.

Chi tiết về Transformer đã được trình bày trong [bài giảng Transformer](/blog/mo-hinh-transformer).

**Mô hình tiêu biểu trong Seq2Seq:**
- **T5 (Raffel et al., 2020)** — [paper](https://arxiv.org/abs/1910.10683) — "Text-to-Text Transfer Transformer", thống nhất mọi NLP task thành format seq2seq.
- **BART (Lewis et al., 2020)** — [paper](https://arxiv.org/abs/1910.13461) — denoising autoencoder pre-training cho seq2seq generation.
- **mT5 (Xue et al., 2021)** — [paper](https://arxiv.org/abs/2010.11934) — T5 đa ngôn ngữ trên 101 ngôn ngữ.

## 3. Các metrics đánh giá bài toán Seq2Seq

Đánh giá chất lượng của mô hình Seq2Seq là bài toán phi tầm thường vì:
- Với cùng một đầu vào, có nhiều câu đầu ra đúng (nhiều bản dịch đúng, nhiều câu tóm tắt đúng).
- Câu đúng về nội dung có thể sai về ngữ pháp, và ngược lại.
- Không có một thước đo duy nhất nào nắm bắt được toàn bộ chất lượng.

### 3.1. Perplexity (PPL)

Perplexity là metric nội tại (intrinsic) để đánh giá chất lượng của **mô hình ngôn ngữ** — đo lường mức độ "bất ngờ" của mô hình khi gặp một chuỗi văn bản chưa thấy.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/7-seq-to-seq/perplexity.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

Một mô hình ngôn ngữ tốt sẽ gán xác suất cao cho các chuỗi văn bản tự nhiên và gán xác suất thấp cho các chuỗi vô nghĩa.
Perplexity định lượng điều này: **PPL càng thấp, mô hình càng dự đoán tốt** chuỗi văn bản tiếp theo.

**Công thức đầy đủ:**

Cho chuỗi văn bản $W = (w_1, w_2, \ldots, w_N)$:

$$PPL(W) = P(w_1, w_2, \ldots, w_N)^{-\frac{1}{N}} = \exp\!\left(-\frac{1}{N} \sum_{i=1}^{N} \log p(w_i | w_1, \ldots, w_{i-1})\right)$$

Đây chính là nghịch đảo hình học của xác suất chuỗi, hay tương đương là $\exp(\text{cross-entropy loss trung bình})$.

Trực giác: nếu tại mỗi bước mô hình luôn dự đoán đúng từ tiếp theo với xác suất $\frac{1}{k}$ đồng đều trong $k$ lựa chọn, thì $PPL = k$.
- $PPL = 1$: mô hình hoàn hảo, luôn đoán đúng.
- $PPL$ cao: mô hình "bất ngờ" nhiều, dự đoán kém.

#### Ví dụ

| Mô hình / Benchmark | PPL |
|---|---|
| GPT-2 (117M) trên Penn Treebank | ~35.0 |
| GPT-2 (1.5B) trên Penn Treebank | ~17.5 |
| GPT-3 (175B) trên Penn Treebank | ~20.5 |
| Mô hình ngẫu nhiên (random) trên vocabulary 50k | ~50,000 |

#### Ưu và nhược điểm

**Ưu điểm:**
- **Nhanh và đơn giản:** Tính trực tiếp từ loss của mô hình, không cần câu tham chiếu.
- **Phù hợp để so sánh các language model** trên cùng một test set và cùng tokenizer.

**Nhược điểm:**
- **Không đo chất lượng đầu ra thực tế:** Perplexity thấp không đảm bảo mô hình sinh ra văn bản tốt hay phù hợp với yêu cầu.
- **Phụ thuộc vào tokenizer:** Perplexity không thể so sánh giữa hai mô hình có tokenizer khác nhau.
- **Không dùng được cho Seq2Seq evaluation:** Perplexity là metric của language model, không đánh giá được chất lượng bản dịch hay tóm tắt.

### 3.2. BLEU (Bilingual Evaluation Understudy)

BLEU được giới thiệu vào năm 2002 trong bài báo [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf) (Papineni et al.).
Ban đầu thiết kế cho dịch máy, BLEU sau đó trở thành metric phổ biến nhất trong nhiều bài toán Seq2Seq (dịch máy, image captioning, tóm tắt văn bản...).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/7-seq-to-seq/bleu.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

Ý tưởng cốt lõi: **đo lường độ chính xác (precision)** của các n-gram trong câu dự đoán $\hat{y}$ so với các câu tham chiếu $S = \{s_1, s_2, \ldots, s_m\}$ của con người.

**Công thức đầy đủ:**

$$BLEU = BP \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Với:

**Brevity Penalty (BP):** Phạt câu dự đoán quá ngắn so với tham chiếu:

$$BP = \begin{cases}1 & \text{nếu } c > r \\ e^{1 - r/c} & \text{nếu } c \leq r\end{cases}$$

Trong đó $c$ là độ dài câu dự đoán, $r$ là độ dài câu tham chiếu gần nhất (best match length).

**Modified n-gram precision** (clipped để tránh câu lặp từ):

$$p_n = \frac{\sum_{\text{n-gram} \in \hat{y}} \text{Count}_{\text{clip}}(\text{n-gram})}{\sum_{\text{n-gram} \in \hat{y}} \text{Count}(\text{n-gram})}$$

Với $\text{Count}_{\text{clip}}(\text{n-gram}) = \min\!\left(\text{Count}(\text{n-gram in } \hat{y}),\, \max_j \text{Count}(\text{n-gram in } s_j)\right)$

Thường $w_n = \frac{1}{N}$ và $N = 4$ (BLEU-4). Thực tế báo cáo đồng thời BLEU-1, BLEU-2, BLEU-3, BLEU-4.

#### Ví dụ

Xét ví dụ dịch máy:
- **Câu nguồn (EN):** "The cat is sitting on the mat"
- **Bản dịch mô hình:** "Con mèo đang ngồi trên tấm thảm"
- **Tham chiếu 1:** "Con mèo đang ngồi trên tấm thảm"
- **Tham chiếu 2:** "Một con mèo ngồi trên chiếc thảm"

BLEU-4 ≈ 1.0 so với tham chiếu 1 (khớp hoàn hảo), ≈ 0.45 so với tham chiếu 2 (khớp một phần).

Giá trị BLEU-4 điển hình trên WMT translation benchmarks:

| Hệ thống | BLEU-4 (EN→DE) |
|---|---|
| Google Translate (2020) | ~33–36 |
| mBART-50 | ~27–30 |
| Baseline phrase-based SMT | ~20–22 |
| Mô hình RNN không attention | ~15–18 |

#### Ưu và nhược điểm

**Ưu điểm:**
- **Đơn giản, nhanh và phổ biến:** Dễ tính toán, là chuẩn so sánh phổ biến nhất trong mọi paper dịch máy và captioning.
- **Reproducible:** Cho phép so sánh công bằng giữa các nghiên cứu.

**Nhược điểm:**
- **Không quan tâm đến ngữ nghĩa:** "a man riding a horse" và "a person on horseback" có nghĩa giống nhau nhưng điểm BLEU rất thấp.
- **Không quan tâm đến thứ tự câu:** Với BLEU-1, các từ có thể đảo thứ tự tùy ý mà không bị phạt.
- **Tương quan kém với đánh giá con người** ở mức câu đơn lẻ.
- **Không xử lý đồng nghĩa:** "dog" và "canine" bị coi là hoàn toàn khác nhau.

### 3.3. METEOR (Metric for Evaluation of Translation with Explicit ORdering)

METEOR được giới thiệu trong bài báo [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](https://aclanthology.org/W05-0909.pdf) (Banerjee & Lavie, 2005), thiết kế để khắc phục các nhược điểm của BLEU, đặc biệt về từ đồng nghĩa.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/7-seq-to-seq/meteor.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

METEOR thực hiện so khớp unigram theo ba mức độ ưu tiên giảm dần:
1. **Exact match:** khớp từ chính xác
2. **Stemmed match:** khớp sau khi rút gốc từ ("running" ↔ "runs", "walked" ↔ "walk")
3. **Synonym match:** khớp theo WordNet synonyms ("dog" ↔ "canine", "big" ↔ "large")

**Công thức đầy đủ:**

**Bước 1 — Tính Precision và Recall trên unigrams:**

$$P = \frac{|matched|}{|\hat{y}|}, \quad R = \frac{|matched|}{|y|}$$

**Bước 2 — F-mean lệch về recall** (recall có trọng số gấp 9 lần precision):

$$F_{mean} = \frac{10 \cdot P \cdot R}{R + 9P}$$

Việc ưu tiên recall phù hợp với dịch máy và captioning: câu dự đoán nên bao phủ đủ thông tin của tham chiếu.

**Bước 3 — Chunk penalty** (phạt các từ match không liên tục, tức là bị đảo thứ tự):

$$Pen = 0.5 \cdot \left(\frac{\text{number of chunks}}{\text{number of matched unigrams}}\right)^3$$

Với "chunk" là nhóm các từ matched liên tiếp nhau. Penalty = 0 khi tất cả từ matched liên tiếp (thứ tự hoàn hảo); penalty tăng mạnh khi thứ tự bị đảo lộn nhiều (mũ 3 làm penalty tăng phi tuyến).

**METEOR cuối cùng:**

$$METEOR = F_{mean} \cdot (1 - Pen)$$

#### Ví dụ

| Câu dự đoán | Câu tham chiếu | METEOR | BLEU-4 | Lý do chênh lệch |
|---|---|---|---|---|
| "a dog runs on grass" | "a dog is running in the park" | ~0.32 | ~0.12 | Stemming: "runs" ↔ "running" |
| "a canine chases a feline" | "a dog chases a cat" | ~0.28 | ~0.08 | Synonyms: dog/canine, cat/feline |
| "the quick brown fox" | "the quick brown fox" | 1.0 | 1.0 | Exact match |

#### Ưu và nhược điểm

**Ưu điểm:**
- **Xử lý từ đồng nghĩa và biến thể từ:** Linh hoạt hơn BLEU nhờ stemming và WordNet.
- **Tương quan tốt hơn với đánh giá con người** ở mức câu đơn lẻ so với BLEU.
- **Xét thứ tự từ:** Chunk penalty phạt đảo thứ tự, phản ánh tính trôi chảy của ngôn ngữ.

**Nhược điểm:**
- **Phụ thuộc vào WordNet:** Yêu cầu tài nguyên ngôn ngữ tiếng Anh — khó áp dụng cho các ngôn ngữ thiếu tài nguyên.
- **Chậm hơn BLEU** do cần tra cứu từ điển đồng nghĩa.
- **Vẫn dựa trên n-gram unigram:** Không xét ngữ cảnh, không hiểu ngữ nghĩa sâu.

### 3.4. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE được giới thiệu trong bài báo [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf) (Lin, 2004), thiết kế ban đầu cho **tóm tắt văn bản** nhưng được sử dụng rộng rãi trong nhiều bài toán Seq2Seq khác.

Khác với BLEU tập trung vào **precision** (câu dự đoán có bao nhiêu n-gram khớp với tham chiếu), ROUGE tập trung vào **recall** (tham chiếu có bao nhiêu n-gram xuất hiện trong câu dự đoán).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/7-seq-to-seq/rouge.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

Có nhiều biến thể ROUGE, trong đó phổ biến nhất là ROUGE-N và ROUGE-L.

**ROUGE-N:** Tính recall của n-gram từ câu tham chiếu xuất hiện trong câu dự đoán:

$$ROUGE\text{-}N = \frac{\sum_{s_j \in S} \sum_{\text{n-gram} \in s_j} \text{Count}_{\text{match}}(\text{n-gram})}{\sum_{s_j \in S} \sum_{\text{n-gram} \in s_j} \text{Count}(\text{n-gram})}$$

**ROUGE-L** (phổ biến nhất): Dựa trên Longest Common Subsequence (LCS) — chuỗi con chung dài nhất, không yêu cầu các từ liền kề:

Cho câu dự đoán $\hat{y}$ (độ dài $m$) và tham chiếu $y$ (độ dài $n$):

$$R_{LCS} = \frac{LCS(\hat{y},\, y)}{n}, \quad P_{LCS} = \frac{LCS(\hat{y},\, y)}{m}$$

$$ROUGE\text{-}L = F_{LCS} = \frac{(1 + \beta^2) \cdot R_{LCS} \cdot P_{LCS}}{R_{LCS} + \beta^2 \cdot P_{LCS}}$$

Thường dùng $\beta = 1.2$ (recall được ưu tiên nhẹ hơn precision).

#### Ví dụ

Xét tóm tắt văn bản:
- **Tham chiếu:** "The cat sat on the mat near the window"
- **Tóm tắt A:** "The cat sat on the mat" → ROUGE-L cao (LCS dài)
- **Tóm tắt B:** "A feline rested on a rug" → ROUGE-L thấp (không có từ chung)

ROUGE-1 và ROUGE-2 điển hình trên CNN/DailyMail summarization:

| Mô hình | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| BART-large | 44.2 | 21.3 | 40.9 |
| T5-large | 42.5 | 20.7 | 39.8 |
| PEGASUS | 44.2 | 21.5 | 41.1 |
| Lead-3 baseline | 40.4 | 17.6 | 36.7 |

#### Ưu và nhược điểm

**Ưu điểm:**
- **Tập trung vào recall:** Phù hợp với tóm tắt văn bản, nơi quan trọng là bao phủ đủ thông tin chính từ văn bản gốc.
- **ROUGE-L linh hoạt hơn BLEU:** LCS không yêu cầu từ liền kề, phản ánh tốt hơn sự tương đồng về cấu trúc.
- **Phổ biến và dễ so sánh** trong cộng đồng nghiên cứu text summarization.

**Nhược điểm:**
- **Không xử lý đồng nghĩa:** "cat" và "feline" vẫn không được coi là khớp.
- **Bias về độ dài:** Tóm tắt dài hơn có xu hướng đạt điểm ROUGE cao hơn vì bao phủ nhiều từ hơn từ tham chiếu.
- **Không đo chất lượng ngôn ngữ:** Câu sai ngữ pháp vẫn có thể đạt ROUGE cao nếu chứa đúng n-gram.

### 3.5. BERTScore

BERTScore được giới thiệu trong bài báo [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) (Zhang et al., 2020), sử dụng **contextual embeddings** của BERT để đo lường sự tương đồng ngữ nghĩa giữa câu dự đoán và tham chiếu.

Đây là bước tiến lớn so với các metrics n-gram: thay vì so khớp chuỗi ký tự, BERTScore so sánh **ý nghĩa ngữ cảnh** của từng token.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/7-seq-to-seq/bert_score.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

BERTScore dùng **greedy token matching** dựa trên cosine similarity giữa BERT embeddings.

**Công thức đầy đủ:**

Đưa câu dự đoán $\hat{y} = (\hat{y}_1, \ldots, \hat{y}_m)$ và tham chiếu $y = (y_1, \ldots, y_n)$ qua BERT để nhận contextual embeddings $\hat{\mathbf{y}}_i$ và $\mathbf{y}_j$:

**Recall** (mỗi token tham chiếu được match với token dự đoán gần nhất):

$$R_{BERT} = \frac{1}{|y|} \sum_{y_j \in y} \max_{\hat{y}_i \in \hat{y}} \cos(\mathbf{y}_j,\, \hat{\mathbf{y}}_i)$$

**Precision** (mỗi token dự đoán được match với token tham chiếu gần nhất):

$$P_{BERT} = \frac{1}{|\hat{y}|} \sum_{\hat{y}_i \in \hat{y}} \max_{y_j \in y} \cos(\hat{\mathbf{y}}_i,\, \mathbf{y}_j)$$

**F1-score:**

$$F_{BERT} = \frac{2 \cdot P_{BERT} \cdot R_{BERT}}{P_{BERT} + R_{BERT}}$$

Với $\cos(\mathbf{u}, \mathbf{v}) = \dfrac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$.

Vì BERT là contextual (mỗi token embedding phụ thuộc vào toàn bộ câu), hai từ đồng nghĩa trong cùng ngữ cảnh sẽ có embedding gần nhau, và BERTScore sẽ coi chúng là khớp — điều mà BLEU/ROUGE không làm được.

#### Ví dụ

| Câu dự đoán | Câu tham chiếu | BERTScore F1 | BLEU-4 |
|---|---|---|---|
| "a dog runs on grass" | "a dog is running in the park" | ~0.89 | ~0.12 |
| "a canine sprints through meadow" | "a dog runs across a field" | ~0.86 | ~0.05 |
| "the weather is nice today" | "a dog runs on grass" | ~0.72 | ~0.01 |

#### Ưu và nhược điểm

**Ưu điểm:**
- **Xử lý đồng nghĩa và paraphrase:** Nhờ contextual embeddings, "dog" ≈ "canine", "runs" ≈ "sprints" trong ngữ cảnh tương đương.
- **Tương quan tốt hơn với đánh giá con người** so với BLEU/ROUGE trên nhiều benchmark.
- **Không cần căn chỉnh chính xác theo token:** Greedy matching linh hoạt hơn n-gram exact match.

**Nhược điểm:**
- **Phụ thuộc vào BERT:** Kết quả phụ thuộc vào chất lượng và ngôn ngữ của mô hình BERT được chọn.
- **Chậm hơn:** Phải chạy BERT cho mỗi cặp câu, tốn nhiều tài nguyên hơn BLEU.
- **Không hoàn toàn interpretable:** Khó giải thích tại sao hai câu có BERTScore cao hay thấp.
- **Bias theo BERT training data:** Kế thừa các bias từ dữ liệu pre-training của BERT.

### 3.6. Human Evaluation

Đánh giá bởi con người là tiêu chuẩn vàng cuối cùng cho mọi bài toán Seq2Seq.
Mọi metric tự động đều được phát triển và kiểm chứng bằng cách đo tương quan với đánh giá của con người.

Hai dạng phổ biến:
- **Adequacy (Tính đầy đủ):** Câu đầu ra có chứa đủ nội dung của câu nguồn không?
- **Fluency (Tính trôi chảy):** Câu đầu ra có đúng ngữ pháp và tự nhiên không?

**Ưu điểm:** Đây là thước đo thực sự của chất lượng — nắm bắt được mọi sắc thái mà metric tự động bỏ qua.
**Nhược điểm:** Tốn kém, không thể scale, có sự chủ quan giữa các người đánh giá.

## 4. Các ứng dụng và thách thức

### Ứng dụng tiêu biểu

- **Dịch máy thần kinh (NMT):** Google Translate, DeepL, Microsoft Translator — đều dựa trên kiến trúc Transformer Seq2Seq.
- **Tóm tắt văn bản tự động:** Tóm tắt bài báo, tài liệu pháp lý, báo cáo y tế.
- **Sinh mã nguồn:** GitHub Copilot, ChatGPT code generation — prompt tiếng tự nhiên → code.
- **Hỏi đáp:** Hệ thống customer support tự động, virtual assistants.
- **Image Captioning:** Xem [bài giảng Image Captioning](/blog/image-captioning).

### Các thách thức hiện tại

- **Hallucination:** Mô hình sinh ra thông tin không có trong đầu vào, đặc biệt nguy hiểm trong dịch y tế và pháp lý.
- **Đánh giá tự động chưa hoàn hảo:** BLEU và các metrics n-gram vẫn tương quan kém với chất lượng thực tế theo cảm nhận con người.
- **Kiểm soát phong cách và độ dài:** Khó điều khiển chính xác phong cách, giọng văn, và độ dài đầu ra theo yêu cầu.
- **Đa ngôn ngữ và ngôn ngữ ít tài nguyên:** Chất lượng giảm mạnh với các ngôn ngữ có ít dữ liệu huấn luyện.
- **Tính nhất quán:** Mô hình có thể mâu thuẫn với chính mình trong một đoạn văn dài.
