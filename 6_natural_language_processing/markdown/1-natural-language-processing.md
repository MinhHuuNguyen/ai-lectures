---
time: 02/19/2023
title: Xử lý ngôn ngữ tự nhiên Natural language processing
description: Natural language processing là một lĩnh vực con của machine learning và deep learning giúp máy tính có khả năng hiểu và xử lý được thông tin dưới dạng văn bản. Ngôn ngữ là một phần không thể thiếu trong cuộc sống của con người, và việc máy tính có khả năng hiểu được ngôn ngữ tự nhiên của con người là một bước tiến lớn trong việc phát triển trí tuệ nhân tạo.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/banner.png
tags: [deep-learning, natural-language-processing]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về lĩnh vực Natural language processing

Natural language processing (NLP) là một lĩnh con của machine learning và deep learning giúp máy tính có khả năng hiểu và xử lý được thông tin dưới dạng văn bản.

Các mô hình giải quyết bài toán NLP thường hoạt động trên dữ liệu dạng văn bản, cụ thể hơn, dữ liệu có thể là một câu, một từ, một đoạn văn bản hay một tập hợp các từ.
Một số mô hình deep learning thường kết hợp dữ liệu dạng văn bản với các loại dữ liệu khác để máy tính có khả năng phân tích và đưa ra câu trả lời giống với con người.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/nlp.png" style="width: 500px;"/>

Các bài toán NLP đã thu hút được sự quan tâm từ rất lâu, tuy nhiên, những mô hình giải quyết bài toán NLP cổ điển thường khó đạt được đến độ chân thật như con người.
Lý do nằm ở tính phức tạp và trừu tượng trong ngôn ngữ của con người.

Một ví dụ điển hình là câu nói: "Các cổ động viên hôm nay thật cuồng nhiệt, họ như đốt cháy bầu không khí tại sân vận động".
Với góc nhìn con người, ta dễ dàng hiểu rằng từ "đốt cháy" ở đây mang nghĩa bóng, thể hiện cho tinh thần cổ vũ của "các cổ động viên".
Tuy nhiên, đối với máy tính, không dễ dàng để phân biệt được nghĩa của từ "đốt cháy" ở đây so với "đốt cháy" bằng lửa.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/fire.png" style="width: 500px;"/>

Một lý do khác dẫn đến tính phức tạp trong các bài toán NLP với các mô hình cũ là vấn đề về đa ngôn ngữ.
Trên thế giới có khoảng 200 quốc gia với rất nhiều ngôn ngữ khác nhau, mỗi ngôn ngữ lại có những đặc điểm về từ vựng khác nhau, đặc điểm về ngữ pháp khác nhau ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/multi_lingual.png" style="width: 500px;"/>

Điều này là một rào cản khá lớn khi ta cố gắng xây dựng được mô hình NLP đa ngôn ngữ, và thông thường, trong các mô hình NLP, ngôn ngữ tiếng anh thường có chất lượng tốt nhất do có lượng dữ liệu nhiều.

## 2. Các bài toán con của Natural language processing

### 2.1. Phân lớp văn bản Text classification

Là bài toán đơn giản nhất trong NLP, mô hình giải quyết nhận đầu vào là một đoạn văn bản và trả đầu ra là lớp tương ứng với đoạn văn bản đó.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/classification.png" style="width: 800px;"/>

Bài toán text classification có rất nhiều ứng dụng trong việc phân tích văn bản nói chung như đánh giá email spam, đánh giá cảm xúc của người dùng thông qua feedback, đánh giá xu hướng tích cực hay tiêu cực trên mạng xã hội, kiểm tra lỗi chính tả ...

### 2.2. Phân lớp ngữ pháp văn bản Part-of-speech tagging (PoS tagging)

Là bài toán khá quan trọng trong NLP tập trung vào việc phân tích ngữ pháp của văn bản.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/pos.png" style="width: 800px;"/>

Cụ thể, PoS tagging tập trung vào việc phân loại các từ trong đoạn văn bản là từ loại gì (chủ ngữ, động từ, giới từ, mạo từ ...)

### 2.3. Phân lớp thực thể văn bản Named entity recognition (NER)

Là bài toán phân lớp các thực thể trong văn bản, nhận đầu vào là đoạn văn bản và trả đầu ra là danh sách các thực thể tương ứng với từng từ trong đoạn văn bản đó.

Tuỳ thuộc vào từng bộ dữ liệu mà ta sẽ có danh sách các thực thể khác nhau, tuy nhiên, một số thực thể phổ biên trong bài NER là tên người, địa điểm, thời gian, số điện thoại, email ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/ner.png" style="width: 1000px;"/>

Bên cạnh việc sử dụng các mô hình machine learning, ta cũng có thể sử dụng biểu thức chính quy (regular expression) để xác định một số thực thể trong văn bản.

### 2.4. Sequence-to-sequence (seq2seq)

Là bài toán quan trọng nhất trong NLP, nhận đầu vào là một đoạn văn bản và trả đầu ra là một đoạn văn bản khác.

Đây là bài toán khó nhất trong NLP do tính phức tạp của ngôn ngữ tự nhiên, tuy nhiên, seq2seq lại có rất nhiều ứng dụng thực tiễn trong cuộc sống.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/seq2seq.png" style="width: 600px;"/>

seq2seq có rất nhiều các bài toán con cụ thể hơn.

#### 2.4.1. Tóm tắt văn bản Text summarization

Các mô hình giải quyết bài toán text summarization nhận đầu vào là một đoạn văn bản đầy đủ và trả đầu ra là một đoạn văn bản ngắn hơn, tóm tắt của đoạn văn bản đầu vào.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/summarization.png" style="width: 500px;"/>

Có hai trường phái của các bài toán summarization:
- **Extractive summarization**: các mô hình extractive chỉ sử dụng những câu những từ có trong đoạn văn bản ban đầu để tóm tắt mà không sử dụng thêm những câu từ bên ngoài
- **Abstractive summarization**: các mô hình abstractive được phép sử dụng cả những từ ngữ bên ngoài đoạn văn bản ban đầu, miễn sao có thể tóm tắt được đoạn văn bản chính xác nhất.

#### 2.4.2. Dịch máy Machine translation

Là bài toán dịch tự động, nhận đầu vào là một đoạn văn bản ở ngôn ngữ thứ nhất và trả đầu ra là kết quả dịch đoạn văn bản đó sang một ngôn ngữ khác.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/translation.png" style="width: 500px;"/>

Ở giai đoạn trước, mỗi mô hình dịch máy thường được huấn luyện trên một cặp ngôn ngữ nhất định, tuy nhiên, hiện nay, với sự phát triển của các mô hình ngôn ngữ lớn (large language model), ta có thể xây dựng được các mô hình dịch máy đa ngôn ngữ.

#### 2.4.3. Trả lời tự động Chatbot

Là bài toán rất phổ biến trong NLP giúp ta xây dựng được những hệ thống trả lời tự động.

Các mô hình chatbot cổ điển thường gặp phải vấn đề trong việc trả lời chưa thực sự chân thật và vấn đề về khả năng ghi nhớ đổi với những đoạn hội thoại dài.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/example_chatgpt.png" style="width: 1200px;"/>

Từ năm 2022, với sự phát triển của các mô hình ngôn ngữ lớn (large language model), điển hình là ChatGPT, các mô hình chatbot đã có những bước tiến vượt bậc trong việc trả lời tự động.

#### 2.4.4. Tự động hoàn thành văn bản Auto completion

Là bài toán tự động hoàn thành đoạn văn bản theo đoạn văn bản đầu vào.
Các mô hình auto completion đã được tích hợp rất nhiều trong các công cụ soạn thảo văn bản, giúp người dùng tiết kiệm thời gian và công sức trong việc soạn thảo văn bản.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/auto_completion.png" style="width: 700px;"/>

Một ví dụ rất mạnh mẽ của mô hình auto completion là các phần mềm soạn thảo lập trình như GitHub Copilot, Cursor, chỉ cần nhập một vài từ khoá, mô hình sẽ tự động hoàn thành đoạn code thậm chí cả file code hoặc nhiều file code khác nhau cho người dùng.

### 2.5. Đặt tiêu đề cho ảnh - Image captioning:

Là bài toán xây dựng một câu hoặc một đoạn văn mô tả một ảnh đầu vào.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/captioning.png" style="width: 1000px;"/>

Bài toán này thường được giải quyết bằng cách kết hợp giữa mô hình xử lý ảnh và mô hình xử lý ngôn ngữ tự nhiên (NLP).
Image captioning là bài toán đòi hỏi mô hình vừa có khả năng hiểu và lưu giữ được thông tin hình ảnh và sinh ra đoạn văn bản mô tả đúng ngữ pháp và chân thực.

## 3. Biểu thức chính quy Regular expression

Biểu thức chính quy (Regular expression - Regex) là phương pháp xây dựng nhóm các ký tự, ký hiệu viết ra theo quy luật tạo thành các mẫu (pattern), nó được sử dụng để tìm kiếm văn bản (text).

Cụ thể hơn, khi xây dựng Regex, ta xây dựng các luật để lọc hoặc tìm kiếm các chuỗi văn bản theo các mẫu đã định trước.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/regex.png" style="width: 800px;"/>

Ví dụ:
```python
import re
# Tìm kiếm các số điện thoại trong đoạn văn bản
text = "Số điện thoại của tôi là 0123456789, số điện thoại của bạn là 0987654321"
pattern = r'\d{10}'  # Mẫu tìm kiếm các chuỗi số có độ dài 10
matches = re.findall(pattern, text)
print(matches)  # Output: ['0123456789', '0987654321']
```

Một số quy tắc viết Regex:
| Ký hiệu | Ý nghĩa |
| --- | --- |
| `.` | Bất kỳ ký tự nào |
| `\d` | Bất kỳ chữ số nào (0-9) |
| `\D` | Bất kỳ ký tự nào không phải chữ số |
| `\w` | Bất kỳ ký tự chữ cái hoặc số (a-z, A-Z, 0-9, _) |
| `\W` | Bất kỳ ký tự nào không phải chữ cái hoặc số |
| `\s` | Bất kỳ ký tự khoảng trắng (space, tab, newline) |
| `\S` | Bất kỳ ký tự nào không phải khoảng trắng |
| `^` | Bắt đầu chuỗi |
| `$` | Kết thúc chuỗi |
| `*` | Lặp lại 0 hoặc nhiều lần |
| `+` | Lặp lại 1 hoặc nhiều lần |
| `?` | Lặp lại 0 hoặc 1 lần |
| `{n}` | Lặp lại đúng n lần |
| `{n,}` | Lặp lại ít nhất n lần |
| `{n,m}` | Lặp lại từ n đến m lần |
| ... | ... |

Một số Regex phổ biến và ý nghĩa:

| Regex | Ý nghĩa |
| --- | --- |
| `\d{3}-\d{3}-\d{4}` | Số điện thoại định dạng xxx-xxx-xxxx |
| `\w+@\w+\.\w+` | Địa chỉ email định dạng xxx@xxx.xxx |
| `https?://[^\s]+` | URL bắt đầu bằng http:// hoặc https:// |
| `\b[A-Z][a-z]*\b` | Từ bắt đầu bằng chữ hoa và theo sau là chữ thường |

Regex là một công cụ rất hữu ích trong quá trình xử lý dữ liệu văn bản, giúp ta dễ dàng lọc và tìm kiếm các chuỗi văn bản theo các mẫu đã định trước.

## 4. Workflow dự án Natural language processing

Tương tự như workflow nói chung trong quá trình xây dựng mô hình deep learning, natural language processing workflow cũng trải qua một số bước như **chuẩn bị dữ liệu**, **tiền xử lý dữ liệu**, **xây dựng mô hình**, **huấn luyện** và **đánh giá mô hình**, **kiểm thử mô hình**, **triển khai mô hình**.

Tuy nhiên, đối với NLP, bước tiền xử lý dữ liệu đòi hỏi những thao tác đặc thù với dữ liệu dạng văn bản.
Những thao tác này được gọi chung là **Text processing**.

Thông thường, trong hầu hết các bài toán NLP, quá trình **Text processing** bao gồm rất nhiều các bước như:

### 4.1. Tokenization

Chia một đoạn văn bản thành các đơn vị nhỏ hơn, thường là các từ hoặc câu, được gọi là token.

Cách tokenization đơn giản nhất là chia đoạn văn bản thành các từ bằng cách sử dụng dấu cách (space) làm ranh giới.
Tuy nhiên, với các ngôn ngữ khác nhau hoặc với các mô hình và các bộ dữ liệu khác nhau thì cách tokenization có thể khác nhau.
Thông thường, mỗi mô hình NLP sẽ có một bộ tokenization riêng biệt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/tokenization.png" style="width: 800px;"/>

Từ các token trong bộ dữ liệu, ta có thể xây dựng được từ điển (vocabulary) cho mô hình. Từ điển bao gồm các token và index tương ứng của token đó.

### 4.2. Lemmatization và Stemming

Lemmatization và Stemming là hai kỹ thuật xử lý ngôn ngữ tự nhiên để chuẩn hoá các từ về dạng cơ bản của chúng.

- **Stemming**: Là quá trình loại bỏ các hậu tố (suffix) của từ để đưa từ về dạng gốc.
Ví dụ, "running", "ran", "runner" sẽ được chuyển thành "run".
Stemming thường sử dụng các thuật toán đơn giản và không cần từ điển.

- **Lemmatization**: Là quá trình chuyển đổi từ về dạng cơ bản của nó dựa trên ngữ cảnh và từ điển.
Ví dụ, "better" sẽ được chuyển thành "good".
Lemmatization thường phức tạp hơn và cần từ điển để xác định dạng cơ bản của từ.

Đối với các mô hình NLP cổ điển hoặc những mô hình NLP giải quyết các bài toán đơn giản như phân loại văn bản ..., Stemming và Lemmatization thường được sử dụng để giảm thiểu số lượng từ trong từ điển và cải thiện hiệu suất của mô hình.

Tuy nhiên, với các mô hình NLP seq2seq, quá trình này thường không cần thiết do mô hình đã được huấn luyện trên một lượng dữ liệu rất lớn và có khả năng hiểu được ngữ cảnh của từ.

### 4.3. Stop words

Stop words là các từ xuất hiện rất nhiều trong các đoạn văn bản nhưng những từ này lại không mang lại nhiều ý nghĩa.

Ví dụ, các từ như "the", "is", "and", "a", "to" ... thường được coi là stop words trong tiếng anh.

Đối với các mô hình NLP cổ điển hoặc những mô hình NLP giải quyết các bài toán đơn giản như phân loại văn bản ..., việc loại bỏ stop words là cần thiết để giảm thiểu số lượng từ trong từ điển và cải thiện hiệu suất của mô hình.

Tuy nhiên, với các mô hình NLP seq2seq, quá trình này thường không cần thiết do mô hình đã được huấn luyện trên một lượng dữ liệu rất lớn và có khả năng hiểu được ngữ cảnh của từ.

### 4.4. Word embedding

Word embedding là quá trình mã hoá các từ trong đoạn văn bản thành dạng số.
Quá trình này giúp mô hình có thể hiểu được mối quan hệ giữa các từ và từ đó cải thiện hiệu suất của mô hình.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/1-natural-language-processing/embedding.png" style="width: 700px;"/>

Một số phương pháp word embedding phổ biến là: **Bag of words (BoW)**, **Term Frequency-Inverse Document Frequency (TF-IDF)**, **Word2Vec**, **GloVe**, **FastText** ...

## 5. Mô hình ngôn ngữ lớn Large language model

Mô hình Ngôn ngữ Lớn (Large Language Model – LLM) là các mô hình học sâu có quy mô cực lớn được huấn luyện trên lượng dữ liệu văn bản khổng lồ.
Các LLM thường có hàng chục đến hàng trăm tỷ tham số, học từ hàng triệu trang văn bản (toàn bộ Wikipedia, sách, web ...) để nắm bắt ngữ pháp, ý nghĩa ngôn ngữ và kiến thức thế giới, có khả năng hiểu và sinh ngôn ngữ tự nhiên.

Kiến trúc nền tảng của đa số LLM hiện đại là Transformer và sử dụng các kỹ thuật huấn luyện đặc biệt như Masked Language Modeling (MLM), Standard Language Modeling (SLM), Reinforcement Learning from Human Feedback (RLHF) ...
Kết quả là LLM học được cách dự đoán từ ngữ tiếp theo trong một ngữ cảnh, LLM có khả năng sinh ra văn bản lưu loát, mạch lạc về nhiều chủ đề.

Nội dung chi tiết mô hình Transformer và các mô hình LLM có thể được tìm thấy trong [bài viết này](/blog/mo-hinh-transformer).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/llm.png" style="width: 600px;"/>

Ví dụ: ChatGPT của OpenAI là một ứng dụng tiêu biểu của LLM.
Nó được xây dựng dựa trên dòng mô hình GPT-3.5/GPT-4, đã được huấn luyện trên lượng dữ liệu văn bản đồ sộ.
Nhờ đó, ChatGPT có thể trả lời hầu hết các câu hỏi, hỗ trợ viết thư, bài luận, làm thơ, viết mã lập trình, thảo luận nhiều chủ đề... một cách lưu loát tự nhiên.
Điểm đặc biệt là ChatGPT tương tác dưới dạng hội thoại, nhớ ngữ cảnh các câu trước đó và phản hồi như một người trợ lý thông minh.

Thành công của ChatGPT đã thúc đẩy một cuộc đua phát triển LLM trên toàn cầu: Gemini của Google, Claude của Anthropic, LLaMA của Meta, DeepSeek ...
Những mô hình này có thể sinh ra văn bản mạch lạc, trả lời câu hỏi, dịch thuật, viết mã ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/llm2slm.png" style="width: 600px;"/>

Trong thời gian gần đây, một hướng nghiên cứu mới cũng nhận được nhiều sự quan tâm là Small Language Model (SLM) - mô hình ngôn ngữ nhỏ.
SLM là các mô hình ngôn ngữ có kích thước nhỏ hơn LLM nhưng vẫn có khả năng sinh văn bản chất lượng cao.
