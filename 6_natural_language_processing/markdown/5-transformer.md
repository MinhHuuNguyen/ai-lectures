---
time: 03/04/2023
title: Mô hình Transformer
description: Mô hình seq2seq (sequence-to-sequence) gồm Encoder (mã hóa) và Decoder (giải mã) là kiến trúc mạng nơ-ron được sử dụng để chuyển đổi một chuỗi đầu vào thành một chuỗi đầu ra. Tuy nhiên, mô hình seq2seq truyền thống gặp khó khăn trong việc xử lý các chuỗi dài do phụ thuộc vào RNN/LSTM. Năm 2017, Transformer ra mắt đã giải quyết vấn đề này bằng cách sử dụng cơ chế Attention hoàn toàn, cho phép mô hình học được mối quan hệ giữa các từ trong chuỗi mà không cần tuần tự.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/seq2seq.png
tags: [deep-learning, natural-language-processing]
is_highlight: false
is_published: true
---

*Note: Một số nội dung trong bài viết này được cập nhật trong thời gian gần đây.*

## 1. Mô hình Encoder - Decoder truyền thống

## 2. Mô hình Transformer

Mô hình Transformer được giới thiệu trong bài báo ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) của Google vào năm 2017.
Hình dưới mô tả kiến trúc của Transformer được lấy từ bài báo trên.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/transformer.png" style="width: 600px;"/>

### 2.1. Attention trong Transformer

Transformer sử dụng cơ chế Attention hoàn toàn, cụ thể, Transformer sử dụng **Self-Attention** để tính toán mối quan hệ giữa các từ trong chuỗi đầu vào và **Multi-Head Attention** để mô hình học được các mối quan hệ phức tạp hơn.

Ngoài ra, Transformer còn sử dụng **Cross-Attention** trong phần Decoder để kết hợp thông tin từ chuỗi đầu vào với các output của chuỗi đầu ra.
Tuy nhiên, trong bài báo gốc, cái tên Cross-Attention không được sử dụng.

Tham khảo về cách thức hoạt động của **Self-Attention**, **Multi-Head Attention** và **Cross-Attention** trong bài viết [này](/blog/co-che-attention-attention-mechanism).

### 2.2. Kiến trúc Transformer


## 3. Mô hình BERT và các biến thể

### 3.1. Mô hình BERT

BERT (Bidirectional Encoder Representations from Transformers) là mô hình được Google giới thiệu năm 2018 qua bài báo ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805).
Các hình ảnh ở phần này được lấy từ bài báo trên.

Trong bài báo, nhóm tác giả giới thiệu 2 mô hình BERT là BERT-Base và BERT-Large với kích thước mô hình lần lượt là 110 triệu và 340 triệu tham số.

BERT chỉ dùng phần Encoder của Transformer và huấn luyện theo phương pháp **Masked Language Model** (MLM) và **Next Sentence Prediction** (NSP).
- Với phương pháp **Masked Language Model**, một số từ trong câu bị che đi (mask) và mô hình phải đoán những từ này dựa vào cả ngữ cảnh bên trái lẫn phải của từ đó, do đó, BERT là mô hình hai chiều (bidirectional).
- Với phương pháp **Next Sentence Prediction**, mô hình sẽ được huấn luyện để dự đoán xem một câu có phải là câu tiếp theo của một câu khác hay không, do đó, BERT có thể hiểu được mối quan hệ giữa các câu trong văn bản.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/bert.png" style="width: 1000px;"/>

Trên hình, một cặp câu A và B được đưa vào mô hình BERT
- Để phục vụ cho bài toán **Masked Language Model**:
    - Trên hai câu này, một lượng khoảng 15% số lượng token đã được che đi (mask).
    - Ví dụ: 80% lượng dữ liệu sẽ được mask với dạng "my dog is hairy" thành "my dog is [MASK]", 10% sẽ được thay thế bằng một token ngẫu nhiên khác "my dog is hairy" thành "my dog is apple", và 10% sẽ giữ nguyên token gốc "my dog is hairy" là "my dog is hairy".
    - Các token đã bị che trong cặp câu trên sau khi được đưa vào mô hình sẽ được đi qua lớp softmax để dự đoán token gốc của chúng.
- Để phục vụ cho bài toán **Next Sentence Prediction**:
    - Input của mô hình sẽ được thêm một token đặc biệt [CLS] ở đầu câu A và nhiệm vụ của mô hình là dự đoán giá trị của token này là 1 (IsNext) hoặc 0 (NotNext) để xác định xem câu B có phải là câu tiếp theo của câu A hay không.
    - Ví dụ: "[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]" sẽ có label của token [CLS] là IsNext.

Với phương pháp huấn luyện này, BERT có thể học được các biểu diễn ngữ nghĩa của từ trong ngữ cảnh, câu trong đoạn văn bản.
Ngoài ra, với cách huấn luyện self-supervised này, BERT có thể được huấn luyện trên một lượng dữ liệu lớn mà không cần gán nhãn, giúp tận dụng tốt các tập dữ liệu lớn về văn bản có sẵn trên internet.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/bert_finetune.png" style="width: 1000px;"/>

Từ đó, BERT được sử dụng như một mô hình pretrained rất tốt cho các tác vụ NLP khác nhau như phân loại văn bản, gán nhãn thực thể, trả lời câu hỏi, v.v.
Để sử dụng BERT cho các tác vụ NLP khác, ta sẽ sử dụng **Fine-tuning** bằng cách thêm một lớp đầu ra (output layer) phù hợp với tác vụ cụ thể trong khi giữ nguyên các tham số của mô hình BERT đã được huấn luyện trước đó.

### 3.2. Các biến thể của BERT

#### RoBERTa

Mô hình RoBERTa (Robustly optimized BERT approach) được giới thiệu bởi Facebook AI vào năm 2019 trong bài báo ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692).

RoBERTa là một biến thể của BERT với các cải tiến trong quá trình huấn luyện:
- **Dynamic Masking:**
Khi huấn luyện BERT, các câu được mask một cách cố định (static) giữa các epoch và các token bị mask sẽ không thay đổi. RoBERTa sử dụng dynamic masking, tức là các token bị mask sẽ được thay đổi trong mỗi epoch, giúp mô hình học được nhiều biểu diễn khác nhau của cùng một câu.
RoBERTa sẽ mask các token theo 10 cách khác nhau, tạo ra sự đa dạng hơn nhiều trong quá trình huấn luyện so với BERT.
- **Loại bỏ Next Sentence Prediction (NSP):**
RoBERTa loại bỏ bài toán Next Sentence Prediction (NSP) trong quá trình huấn luyện, chỉ sử dụng bài toán Masked Language Model (MLM) để huấn luyện mô hình.
Sự thay đổi này giúp mô hình đạt kết quả tốt hơn so với BERT trong một số bài toán finetuning.
- **Tăng batch size:**
RoBERTa sử dụng batch size lớn hơn và đạt kết quả tốt hơn so với BERT trong một số bài toán finetuning.
- **Byte-Pair Encoding (BPE):**
RoBERTa sử dụng phương pháp Byte-Pair Encoding (BPE) để mã hóa token, tuy nhiên, kết quả của RoBERTa không khác biệt nhiều so với BERT.

#### DistilBERT

Mô hình DistilBERT được giới thiệu bởi Hugging Face vào năm 2019 trong bài báo ["DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"](https://arxiv.org/abs/1910.01108).
Hình ảnh dưới đây được lấy từ bài báo, là thống kê về kích thước của các mô hình ngôn ngữ tại thời điểm mà bài báo được công bố.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/distil_bert.png" style="width: 600px;"/>

DistilBERT sử dụng phương pháp **Knowledge Distillation** để giảm kích thước mô hình và là một phiên bản nhẹ hơn của BERT với kích thước mô hình nhỏ hơn (chỉ 66 triệu tham số so với 110 triệu của BERT-Base), thời gian dự đoán nhanh hơn (60% so với BERT-Base) và hiệu suất được duy trì gần như tương đương với BERT-Base (97% hiệu suất của BERT-Base).

#### Pho-BERT

Mô hình Pho-BERT được giới thiệu bởi VinAI Research vào năm 2020 trong bài báo ["PhoBERT: Pre-trained Language Models for Vietnamese"](https://arxiv.org/abs/2003.00744).

Pho-BERT sử dụng kiến trúc mô hình tương đương BERT-Base và BERT-Large và chiến lược huấn luyện của Pho-BERT tương tự như RoBERTa.
Pho-BERT được huấn luyện trên một tập dữ liệu lớn gồm 20GB văn bản tiếng Việt từ nhiều nguồn khác nhau như báo chí, sách, diễn đàn, v.v.

Pho-Bert là mô hình BERT đầu tiên được huấn luyện trên tiếng Việt và đã đạt được kết quả tốt trong các bài toán NLP tiếng Việt như phân loại văn bản, gán nhãn thực thể, trả lời câu hỏi, v.v.

## 4. Mô hình GPT và các biến thể

### 4.1. Mô hình GPT

Mô hình GPT (Generative Pre-trained Transformer) được giới thiệu bởi OpenAI vào năm 2018 trong bài báo ["Improving Language Understanding by Generative Pre-Training"](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).

Mô hình GPT sử dụng kiến trúc Transformer chỉ với phần Decoder, khác với BERT chỉ sử dụng phần Encoder.
Cách thức huấn luyện của GPT cũng khác so với BERT.
Nếu như BERT sử dụng phương pháp **Masked Language Model**, mask đi một số từ nhất định trong câu và dự đoán các từ này, thì GPT sử dụng phương pháp **Standard Language Modeling**, yêu cầu mô hình dự đoán từ tiếp theo trong chuỗi văn bản dựa trên các từ trước đó.

Từ đó, GPT là mô hình unidirectional, chỉ có thể dự đoán từ tiếp theo dựa trên các từ đã thấy trước đó trong khi BERT là mô hình bidirectional.
Giá trị mà hai mô hình này mang lại cho các tác vụ NLP khác là khác nhau:
- BERT mạnh với các tác vụ cần hiểu rõ ngữ nghĩa của câu, hiểu rõ mối quan hệ của các từ trong câu, như phân loại văn bản, gán nhãn thực thể, v.v.
- GPT mạnh với các tác vụ cần sinh văn bản, như sinh câu trả lời, sinh văn bản tự động, v.v.

Hình ảnh dưới đây được lấy từ bài báo, mô tả cách thức huấn luyện mô hình GPT.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/gpt.png" style="width: 600px;"/>

Đối với bài toán **text classification** (phân lớp văn bản), mô hình GPT sẽ đơn giản là nhận đầu vào là một chuỗi văn bản, sau đó đưa đầu ra qua một lớp softmax để dự đoán nhãn của văn bản.

Đối với bài toán **text entailment** (suy diễn mối quan hệ văn bản), mô hình GPT sẽ nhận đầu vào câu tiền đề (premise) được nối với câu giả thuyết (hypothesis) và đưa ra đầu ra là xác suất của các nhãn như "entailment" (suy diễn), "contradiction" (mâu thuẫn) và "neutral" (trung lập).

Đối với bài toán **text similarity** (tương đồng văn bản), mô hình GPT sẽ nhận đầu vào là hai câu và nối hai câu này với thứ tự ngược nhau ("sentence A" + "sentence B" và "sentence B" + "sentence A") và đưa ra đầu ra là xác suất của các nhãn như "similar" (tương đồng) và "dissimilar" (không tương đồng).
Phương pháp này được áp dụng tương tự cho bài toán **Multiple-Choice Question Answering** (trả lời câu hỏi nhiều lựa chọn) bằng cách nối câu hỏi với các lựa chọn trả lời và đưa ra xác suất cho từng lựa chọn.

### 4.2. Các biến thể của GPT

#### GPT-2

Mô hình GPT-2 được giới thiệu vào năm 2019 trong bài báo ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

GPT-2 là phiên bản mở rộng của GPT với một số điều chỉnh nhỏ trong kiến trúc và đặc biệt là kích thước mô hình lớn hơn lên tới 1.5 tỷ tham số và được huấn luyện trên một tập dữ liệu lớn hơn lên tới 40GB văn bản.

Ngoài ra, GPT-2 còn được huấn luyện không chỉ với dữ liệu văn bản thông thường mà còn bổ sung thêm các tác vụ (task) vào trong văn bản.

Hình ảnh dưới đây được lấy từ bài báo, mô tả cách thức huấn luyện mô hình GPT-2 với tác vụ dịch thuật giữa tiếng Anh và tiếng Pháp.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/gpt2.png" style="width: 600px;"/>

#### GPT-3

Mô hình GPT-3 được giới thiệu vào năm 2020 trong bài báo ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165).

GPT-3 có kích thước mô hình khổng lồ lên đến 175 tỷ tham số và được huấn luyện trên hàng trăm tỷ token văn bản, dung lượng dữ liệu huấn luyện lên tới 570GB.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/gpt3_in_context_learning.png" style="width: 600px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/gpt3_zero_one_few_shot.png" style="width: 600px;"/>

#### GPT-3.5

Mô hình GPT-3.5 hay còn được gọi là InstructGPT được giới thiệu vào năm 2022 trong bài báo ["Training language models to follow instructions with human feedback"](https://arxiv.org/abs/2203.02155) sử dụng kiến trúc tương tự GPT-3 nhưng được cải tiến với phương pháp huấn luyện mới gọi là **Reinforcement Learning from Human Feedback (RLHF)**.
Phương pháp này cho phép mô hình học từ phản hồi của con người để cải thiện khả năng hiểu và sinh văn bản.

Hình ảnh dưới đây mô tả quá trình huấn luyện mô hình GPT-3.5 với phương pháp RLHF.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/5-transformer/gpt_3p5.png" style="width: 600px;"/>

#### GPT-4

Mô hình GPT-4 được giới thiệu vào năm 2023 trong bài báo ["GPT-4 Technical Report"](https://arxiv.org/abs/2303.08774).
Tuy nhiên, đây chỉ là một báo cáo kỹ thuật tập trung vào việc đánh giá kết quả mà GPT-4 đã đạt được và không có thông tin chi tiết về kiến trúc và cách thức hoạt động của mô hình.

## 5. Mô hình T5

## 6. Mô hình Vision Transformer (ViT)


Vision Transformer (ViT) áp dụng kiến trúc Transformer vào thị giác máy tính. Thay vì dùng CNN như trước đây, ViT chia ảnh đầu vào thành các miếng nhỏ (patch) cỡ cố định (ví dụ 16×16 pixel)
v7labs.com
v7labs.com
. Mỗi miếng ảnh được flatten thành vector và đưa qua một lớp embedding để chuyển thành vectơ biểu diễn phù hợp. 
