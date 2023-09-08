---
layout: "post"
title:  "Natural language processing"
author: "Nguyễn Hữu Minh"
permalink: "/deep-learning/natural-language-processing"
parent: "Deep learning"
has_children: true
nav_order: 4
---

# Natural language processing

## 1. Giới thiệu chung về Natural language processing

Natural language processing (NLP) là một lĩnh con của machine learning và deep learning giúp máy tính có khả năng hiểu và xử lý được thông tin dưới dạng văn bản.

Các mô hình giải quyết bài toán NLP thường hoạt động trên dữ liệu dạng văn bản, cụ thể hơn, dữ liệu có thể là một câu, một từ, một đoạn văn bản hay một tập hợp các từ.
Một số mô hình deep learning thường kết hợp dữ liệu dạng văn bản với các loại dữ liệu khác để máy tính có khả năng phân tích và đưa ra câu trả lời giống với con người.

Các bài toán NLP đã thu hút được sự quan tâm từ rất lâu, tuy nhiên, những mô hình giải quyết bài toán NLP cổ điển thường khó đạt được đến độ chân thật như con người.
Lý do nằm ở tính phức tạp và trừu tượng trong ngôn ngữ của con người.

Một ví dụ điển hình là câu nói: "Các cổ động viên hôm nay thật cuồng nhiệt, họ như đốt cháy bầu không khí tại sân vận động".
Với góc nhìn con người, ta dễ dàng hiểu rằng từ "đốt cháy" ở đây mang nghĩa bóng, thể hiện cho tinh thần cổ vũ của "các cổ động viên".
Tuy nhiên, đối với máy tính, không dễ dàng để phân biệt được nghĩa của từ "đốt cháy" ở đây so với "đốt cháy" bằng lửa.

Một lý do khác dẫn đến tính phức tạp trong các bài toán NLP với các mô hình cũ là vấn đề về đa ngôn ngữ.
Trên thế giới có khoảng 200 quốc gia với rất nhiều ngôn ngữ khác nhau, mỗi ngôn ngữ lại có những đặc điểm về từ vựng khác nhau, đặc điểm về ngữ pháp khác nhau ...
Điều này là một rào cản khá lớn khi ta cố gắng xây dựng được mô hình NLP đa ngôn ngữ, và thông thường, trong các mô hình NLP, ngôn ngữ tiếng anh thường có chất lượng tốt nhất do có lượng dữ liệu nhiều.

## 2. Các bài toán natural language processing

NLP là một lĩnh vực gồm nhiều bài toán con, đa dạng về cả dữ liệu đầu vào và lời dự đoán đầu ra.

<img src="https://dezyre.gumlet.io/images/blog/machine-learning-nlp-text-classification-algorithms-and-models/Text_Classification_Machine_Learning_NLP.png" style="width: 500px;"/>

- Text classification: phân lớp văn bản là bài toán đơn giản nhất trong NLP.
Mô hình giải quyết bài toán text classification nhận đầu vào là một đoạn văn bản và trả đầu ra là lớp tương ứng với đoạn văn bản đó.
Bài toán text classification có rất nhiều ứng dụng trong việc phân tích văn bản nói chung như đánh giá email spam, đánh giá cảm xúc của người dùng thông qua feedback, đánh giá xu hướng tích cực hay tiêu cực trên mạng xã hội, kiểm tra lỗi chính tả ...

<img src="https://byteiota.com/wp-content/uploads/2021/01/POS-Tagging-800x400.jpg" style="width: 500px;"/>

- Part-of-speech tagging (PoS tagging): là bài toán khá quan trọng trong NLP tập trung vào việc phân tích ngữ pháp của đoạn văn bản.
Cụ thể, PoS tagging tập trung vào việc phân loại các từ trong đoạn văn bản là từ loại gì (chủ ngữ, động từ, giới từ, mạo từ ...)

<img src="https://www.shaip.com/wp-content/uploads/2022/02/Blog_Named-Entity-Recognition-%E2%80%93-The-Concept-Types-Applications.jpg" style="width: 500px;"/>

- Named entity recognition (NER): là bài toán phân lớp các thực thể trong văn bản.
Các mô hình NER nhận đầu vào là đoạn văn bản và trả đầu ra là danh sách các thực thể tương ứng với từng từ trong đoạn văn bản đó.
Tuỳ thuộc vào từng bộ dữ liệu mà ta sẽ có danh sách các thực thể khác nhau, tuy nhiên, một số thực thể phổ biên trong bài NER là tên người, địa điểm, thời gian, số điện thoại, email ...
Bên cạnh việc sử dụng các mô hình machine learning, ta cũng có thể sử dụng biểu thức chính quy (regular expression) để xác định một số thực thể trong văn bản.

<img src="https://i0.wp.com/turbolab.in/wp-content/uploads/2021/09/Text-Summarization-NLP.jpg" style="width: 500px;"/>

- Sequence-to-sequence (seq2seq): là bài toán quan trọng nhất trong NLP.
Một cách khái quát, các mô hình giải quyết bài toán seq2seq đều nhận đầu vào là một đoạn văn bản và trả đầu ra là một đoạn văn bản khác.
Từ đó, seq2seq có rất nhiều các bài toán con cụ thể hơn:
    - Text summarization: là bài toán tóm tắt đoạn văn bản. Các mô hình giải quyết bài toán text summarization nhận đầu vào là một đoạn văn bản đầy đủ và trả đầu ra là một đoạn văn bản ngắn hơn, tóm tắt của đoạn văn bản đầu vào. Có hai trường phái của các bài toán summarization:
        - Extractive summarization: các mô hình theo trường phái extractive chỉ sử dụng những câu những từ có trong đoạn văn bản ban đầu để tóm tắt mà không sử dụng thêm những câu từ bên ngoài
        - Abstractive summarization: các mô hình theo trường phái abstractive được phép sử dụng cả những từ ngữ bên ngoài đoạn văn bản ban đầu, miễn sao có thể tóm tắt được đoạn văn bản chính xác nhất.
    - Machine translation: là bài toán dịch tự động. Mô hình nhận đầu vào là một đoạn văn bản ở ngôn ngữ thứ nhất và trả đầu ra là kết quả dịch đoạn văn bản đó sang một ngôn ngữ khác
    - Chatbot: là bài toán rất phổ biến trong NLP giúp ta xây dựng được những hệ thống trả lời tự động. Các mô hình chatbot cổ điển thường gặp phải vấn đề trong việc trả lời chưa thực sự chân thật và vấn đề về khả năng ghi nhớ đổi với những đoạn hội thoại dài.
    - Auto completion: là bài toán tự động hoàn thành đoạn văn bản theo đoạn văn bản đầu vào. Hiện nay, các mô hình auto complete đã được tích hợp rất nhiều trong các công cụ soạn thảo văn bản.

<img src="https://opennmt.net/simple-attn.png" style="width: 500px;"/>

<img src="https://s3.cloud.cmctelecom.vn/tinhte2/2019/02/4569819_ngon-ngu-viet-chatbot.png" style="width: 500px;"/>

Ngoài ra, ta cũng có bài toán kết hợp giữa computer vision và NLP như:
- Image captioning: là bài toán nhận đầu vào là ảnh, và trả đầu ra là đoạn văn bản mô tả ảnh đầu vào.
Image captioning là bài toán đòi hỏi mô hình vừa có khả năng hiểu và lưu giữ được thông tin hình ảnh và sinh ra đoạn văn bản mô tả đúng ngữ pháp và chân thực.

<img src="https://repository-images.githubusercontent.com/83958320/8f162500-8ace-11e9-94ee-0b86d27bbc5e" style="width: 700px;"/>

## 3. Natural language processing workflow và vai trò của Text processing

Tương tự như workflow nói chung trong quá trình xây dựng mô hình deep learning, natural language processing workflow cũng trải qua một số bước như chuẩn bị dữ liệu, tiền xử lý dữ liệu, xây dựng mô hình, huấn luyện và đánh giá mô hình, kiểm thử mô hình, triển khai mô hình.

Tuy nhiên, đối với NLP, bước tiền xử lý dữ liệu đòi hỏi những thao tác đặc thù với dữ liệu dạng văn bản.
Những thao tác này được gọi chung là Text processing.

<img src="https://miro.medium.com/v2/resize:fit:1200/1*UhfwmhMN9sdfcWIbO5_tGg.jpeg" style="width: 700px;"/>

Thông thường, trong hầu hết các bài toán NLP, quá trình text processing bao gồm rất nhiều các bước như:
- Sentence tokenization: Chia một đoạn văn bản thành các câu
- Word tokenization: Chia một đoạn văn bản thành các từ
- Text lemmatization và stemming: xử lý các dạng ngữ pháp của từ (VD: "am, are, is" trở thành "be", "dog, dogs, dog’s, dogs’" trở thành "dog"). Stemming gồm các thao tác thô như loại bỏ các hậu tố (suffix) trong các từ. Lemmatization sử dụng từ điển và các phép phân tích hình thái học của từ (morphological analysis) để làm sạch dữ liệu text.
- Xử lý stop words: Stop words bao gồm các từ xuất hiện rất nhiều trong các đoạn văn bản nhưng những từ này lại không mang lại nhiều ý nghĩa. Tuy vậy, danh sách các stop words có thể khác nhau phụ thuộc vào cụ thể bộ dữ liệu và bài toán NLP.
- Word embedding: là quá trình mã hoá các từ trong đoạn văn bản thành dạng số. Khái quát hơn, ta mã hoá đoạn văn bản trở thành vector và từ đó các mô hình machine learning sẽ tính toán trên các vector này.

Tóm lại, đầu vào của quá trình text processing là dữ liệu dạng văn bản và trả đầu ra là dữ liệu đã được làm sạch dưới dạng vector mã hoá để sẵn sàng đưa vào trong mô hình machine learning.

<img src="https://www.ruder.io/content/images/size/w2000/2016/04/word_embeddings_colah.png" style="width: 700px;"/>

Một công cụ khá hữu ích trong quá trình xử lý dữ liệu text là Biểu thức chính quy (Regular expression - Regex).
Regex là phương pháp xây dựng nhóm các ký tự, ký hiệu viết ra theo quy luật tạo thành các mẫu (pattern), nó được sử dụng để tìm kiếm văn bản (text).
Cụ thể hơn, khi xây dựng Regex, ta xây dựng các luật để lọc hoặc lấy ra được những đoạn text.
Điều này giúp quá trình làm sạch dữ liệu text trở nên dễ dàng và hiệu quả hơn.

<img src="https://miro.medium.com/v2/resize:fit:1036/1*WfLCo4Ql59kxq_0frEe7xQ.png" style="width: 700px;"/>

<!-- ## 4. Text tokenization và word embedding -->


