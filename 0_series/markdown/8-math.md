---
time: 02/16/2023
title: "[SERIES] Toán trong Machine Learning"
description: Toán trong Machine Learning là nền tảng giúp mô hình học và ra quyết định chính xác. Nó bao gồm các lĩnh vực như đại số tuyến tính, giải tích, xác suất – thống kê và tối ưu hóa. Nhờ toán học, ta hiểu được cách mô hình hoạt động, huấn luyện hiệu quả hơn và cải thiện khả năng tổng quát hóa trong các bài toán thực tế.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/0_series/images/8-math/banner.jpeg
tags: [math, series]
is_highlight: false
is_published: true
---

---

### [Bài 1: Đại số tuyến tính Linear Algebra](/blog/dai-so-tuyen-tinh-linear-algebra)

Đại số tuyến tính là nền tảng toán học cốt lõi trong machine learning, cung cấp ngôn ngữ và công cụ để biểu diễn và xử lý dữ liệu dưới dạng vector, ma trận và tensor. Các phép toán như nhân ma trận, chuẩn hóa vector hay phân rã ma trận được sử dụng trong hầu hết các thuật toán học máy, từ hồi quy tuyến tính đến mạng nơ-ron sâu. Nhờ đại số tuyến tính, việc tối ưu hóa mô hình, biểu diễn đặc trưng và xử lý dữ liệu quy mô lớn trở nên hiệu quả và chính xác hơn.

###### 1. Giới thiệu chung về Đại số tuyến tính

###### 2. Các khái niệm ma trận đặc biệt

###### 3. Các phép toán cơ bản trên vector và ma trận

###### 4. Một số ứng dụng trong Machine Learning

---

### [Bài 2: Giải tích toán học Calculus](/blog/giai-tich-toan-hoc-calculus)

Giải tích toán học (Calculus) đóng vai trò trọng yếu trong machine learning, đặc biệt trong việc tối ưu hóa mô hình. Các khái niệm như đạo hàm, vi phân và gradient được sử dụng để cập nhật tham số mô hình nhằm giảm thiểu hàm mất mát. Nhờ giải tích, thuật toán gradient descent và các biến thể của nó có thể tìm ra điểm cực tiểu của hàm mục tiêu, giúp mô hình học được từ dữ liệu. Ngoài ra, giải tích còn hỗ trợ hiểu rõ sự biến thiên của đầu ra theo đầu vào, góp phần vào việc phân tích độ nhạy và ổn định của hệ thống học máy.

###### 1. 

###### 2. 

###### 3. 

###### 4. 

---

### [Bài 3: Một số khái niệm cơ bản trong xác suất](/blog/mot-so-khai-niem-co-ban-trong-xac-suat)

Xác suất là một trong những khái niệm quan trọng nhất trong thống kê và học máy. Nó giúp chúng ta hiểu rõ hơn về cách mà các biến ngẫu nhiên tương tác với nhau và cách mà chúng ta có thể dự đoán các kết quả trong tương lai. Bài viết này sẽ giúp bạn hiểu rõ hơn về các khái niệm cơ bản trong xác suất, bao gồm biến ngẫu nhiên, không gian mẫu, biến cố, kết quả, phân phối xác suất, xác suất đồng thời, xác suất biên và xác suất điều kiện.

###### 1. Biến ngẫu nhiên và các khái niệm liên quan

###### 2. Xác suất đồng thời, xác suất biên và xác suất điều kiện

###### 3. Định lý Bayes (Bayes' theorem)

---

### [Bài 4: Các phân phối xác suất cơ bản](/blog/cac-phan-phoi-xac-suat-co-ban)

Trong machine learning, phân phối xác suất là công cụ quan trọng để mô hình hóa dữ liệu và sự không chắc chắn. Nhiều thuật toán dựa trên giả thiết rằng dữ liệu tuân theo các phân phối xác suất nhất định. Hiểu rõ các phân phối này giúp lựa chọn mô hình và giải thuật thích hợp trong các bài toán học máy.

###### 1. Tổng quan và định lý giới hạn trung tâm (Central Limit Theorem)

###### 2. Hàm khối xác suất, Hàm mật độ xác suất, Hàm phân phối xác suất

###### 3. Kullback-Leibler divergence (KL divergence)

###### 4. Phân phối chuẩn (Normal distribution)

###### 5. Phân phối đều (Uniform distribution)

###### 6. Phân phối Bernoulli (Bernoulli distribution)

---

### [Bài 5: Thống kê dữ liệu](/blog/thong-ke-du-lieu)

Thống kê dữ liệu là nền tảng quan trọng trong machine learning, giúp hiểu và mô tả đặc trưng của dữ liệu trước khi xây dựng mô hình. Thông qua các khái niệm như trung bình, phương sai, độ lệch chuẩn, phân phối xác suất và kiểm định giả thuyết, nhà nghiên cứu có thể đánh giá xu hướng, mức độ biến động và mối quan hệ giữa các biến. Việc phân tích thống kê giúp phát hiện dữ liệu ngoại lai, mất cân bằng hay nhiễu, từ đó hỗ trợ tiền xử lý và lựa chọn mô hình phù hợp. Nhờ đó, mô hình học máy đạt hiệu quả và độ chính xác cao hơn.

###### 1. 

###### 2. 

###### 3. 

###### 4. 

---

### [Bài 6: Thuật toán Maximum Likelihood Estimation (MLE) và Maximum A Posteriori (MAP)](/blog/maximum-likelihood-estimation-mle-va-maximum-a-posteriori-map)

Maximum Likelihood Estimation (MLE) và Maximum A Posteriori (MAP) là hai phương pháp thống kê quan trọng trong machine learning dùng để ước lượng tham số mô hình. MLE tìm giá trị tham số làm cực đại xác suất quan sát dữ liệu, tập trung hoàn toàn vào thông tin từ dữ liệu huấn luyện. Trong khi đó, MAP kết hợp cả dữ liệu và kiến thức tiên nghiệm (prior) thông qua định lý Bayes, cho phép ước lượng ổn định hơn khi dữ liệu hạn chế hoặc nhiễu. Cả hai phương pháp đều đóng vai trò cốt lõi trong các mô hình xác suất và suy luận Bayes hiện đại.

###### 1. 

###### 2. 

###### 3. 

###### 4. 
