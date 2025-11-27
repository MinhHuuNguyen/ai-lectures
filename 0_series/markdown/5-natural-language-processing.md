---
time: 06/28/2022
title: "[SERIES] Xử lý ngôn ngữ tự nhiên Natural Language Processing"
description: Xử lý ngôn ngữ tự nhiên là một lĩnh vực con của trí tuệ nhân tạo, tập trung vào việc tương tác giữa máy tính và con người thông qua ngôn ngữ tự nhiên. NLP có rất nhiều ứng dụng trong đời sống, từ tìm kiếm thông tin đến trợ lý ảo.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/banner.jpeg
tags: [deep-learning, natural-language-processing, series]
is_highlight: false
is_published: true
---

---

### [Bài 1: Xử lý ngôn ngữ tự nhiên Natural language processing](/blog/xu-ly-ngon-ngu-tu-nhien-natural-language-processing)

Natural language processing là một lĩnh vực con của machine learning và deep learning giúp máy tính có khả năng hiểu và xử lý được thông tin dưới dạng văn bản. Ngôn ngữ là một phần không thể thiếu trong cuộc sống của con người, và việc máy tính có khả năng hiểu được ngôn ngữ tự nhiên của con người là một bước tiến lớn trong việc phát triển trí tuệ nhân tạo.

###### 1. Giới thiệu chung về lĩnh vực Natural language processing

###### 2. Các bài toán con của Natural language processing

###### 3. Biểu thức chính quy Regular expression

###### 4. Workflow dự án Natural language processing

###### 5. Mô hình ngôn ngữ lớn Large language model

---

### [Bài 2: Mạng nơ ron hồi quy Recurrent Neural Network](/blog/mang-no-ron-hoi-quy-recurrent-neural-network/)

Mạng nơ ron hồi quy (RNN) là mô hình rất phổ biến trong thời gian trước đây với những kết quả đầy hứa hẹn trên các bài toán xử lý ngôn ngữ tự nhiên (NLP). Cho dù ở thời điểm hiện tại, với sự phát triển của cơ chế Attention và các mô hình Transformer đạt kết quả cao trên các bài toán xử lý ngôn ngữ, ý tưởng về cơ chế hoạt động của các mô hình RNN vẫn đáng chú ý và được áp dụng trong một số trường hợp cụ thể như lượng dữ liệu ít hoặc tài nguyên tính toán hạn chế.

###### 1. Ý tưởng "hồi quy" trong kiến trúc mạng nơ ron

###### 2. Vấn đề phụ thuộc xa của RNN

###### 3. Kiến trúc mô hình Long short-term memory (LSTM)

###### 4. Các biến thể của RNN nói chung và LSTM nói riêng

### [Bài 3: Xử lý dữ liệu văn bản Text data processing](/blog/xu-ly-du-lieu-van-ban-text-data-processing/)

*Mình sẽ viết bài này trong thời gian tới với các nội dung như: Các kỹ thuật tokenization; Các kỹ thuật word embedding như TF-IDF, Bag of Words, Word2Vec, GloVe, BERT...; Các hàm xử lý văn bản với NLTK*

---

### [Bài 4: Cơ chế Attention Attention Mechanism](/blog/co-che-attention-attention-mechanism/)

Attention là cơ chế giúp mô hình học sâu tập trung (attend) vào các thành phần quan trọng trong dữ liệu đầu vào, tương tự như con người chú ý đến chi tiết nổi bật. Cơ chế này ra đời (2014) nhằm khắc phục hạn chế của RNN/LSTM cũ khi phải mã hóa toàn bộ chuỗi vào một vector cố định. Theo kết quả nghiên cứu, Transformer (2017) – kiến trúc dùng hoàn toàn attention – đã trở thành nền tảng của các mô hình ngôn ngữ lớn (LLMs) như GPT, BERT hiện đại. Trong thị giác máy tính, cơ chế Attention cũng được áp dụng thành công qua Vision Transformer (ViT) và các mô hình DETR.

###### 1. Mô hình seq2seq truyền thống

###### 2. Giới thiệu chung về Attention Mechanism

###### 3. Cơ chế Attention cơ bản

###### 4. Một số phiên bản của cơ chế Attention

---

### [Bài 5: Mô hình Transformer](/blog/mo-hinh-transformer/)

Mô hình seq2seq (sequence-to-sequence) là kiến trúc mạng nơ-ron được sử dụng để chuyển đổi một chuỗi đầu vào thành một chuỗi đầu ra. Tuy nhiên, mô hình seq2seq truyền thống gặp khó khăn trong việc xử lý các chuỗi dài do phụ thuộc vào RNN/LSTM. Năm 2017, Transformer ra mắt đã giải quyết vấn đề này bằng cách sử dụng cơ chế Attention hoàn toàn, cho phép mô hình học được mối quan hệ giữa các từ trong chuỗi mà không cần tuần tự.

###### 1. Mô hình Transformer

###### 2. Mô hình BERT và các biến thể

###### 3. Mô hình GPT và các biến thể

###### 4. Mô hình T5

###### 5. Mô hình Vision Transformer (ViT)

---

### [Bài 6: Bài toán nhận diện thực thể trong văn bản Named Entity Recognition NER](/blog/nhan-dien-thuc-the-named-entity-recognition-ner/)

*Mình sẽ viết bài này trong thời gian tới*

### [Bài 7: Bài toán phân tích ngữ pháp trong văn bản Part-of-Speech Tagging](/blog/phan-tich-ngu-phap-part-of-speech-tagging/)

*Mình sẽ viết bài này trong thời gian tới*

### [Bài 8: Bài toán gán tiêu đề cho ảnh Image Captioning](/blog/gian-tieu-de-cho-anh-image-captioning/)

*Mình sẽ viết bài này trong thời gian tới*
