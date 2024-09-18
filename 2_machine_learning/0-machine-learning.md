---
slug: machine-learning
time: "11/09/2024"
title: "Machine learning"
description: "Machine Learning (ML) là một phần của trí tuệ nhân tạo (AI) mà chúng ta dùng để xây dựng các mô hình hoặc chương trình máy tính có khả năng tự học từ dữ liệu."
author: "Nguyễn Hữu Minh"
banner_url: "https://tenten.vn/tin-tuc/wp-content/uploads/2023/08/1cG6U1qstYDijh9bPL42e-Q.jpg"
tags:
is_highlight: false
---

# Giới thiệu chung về Machine Learning

## 1. Tư tưởng của Machine Learning và Lập trình truyền thống

Machine Learning (ML) là một phần của trí tuệ nhân tạo (AI) mà chúng ta dùng để xây dựng các mô hình hoặc chương trình máy tính có khả năng tự học từ dữ liệu.
Thay vì viết cụ thể từng bước giải quyết vấn đề, ML cho phép máy tính "học" từ dữ liệu và cải thiện hiệu suất theo thời gian.

<img src="https://drive.google.com/uc?id=1bIc2ej3NKOMw9CAAiJAZ5H51fsl67PKD" style="width: 800px;"/>

ML đã thay đổi cách chúng ta giải quyết các vấn đề phức tạp, từ dự đoán thời tiết, phân loại email rác, cho đến chẩn đoán bệnh.
Nó giúp chúng ta hiểu rõ hơn về dữ liệu và tạo ra các mô hình dự đoán, tiên đoán và phân tích mà không cần viết lại mã rất nhiều lần.

## 2. Ứng dụng của Machine Learning trong thực tế

### 2.1. Autopilot - Tesla

AutoPilot của Tesla là một hệ thống lái tự động được tích hợp vào các xe điện sản xuất bởi Tesla, một công ty đổi mới trong lĩnh vực ô tô tự lái.
AutoPilot cho phép xe tự động thực hiện nhiều tác vụ lái xe, như duy trì làn đường, duyệt qua giao lộ, và tự động điều khiển tốc độ, giữ khoảng cách an toàn.

<img src="https://drive.google.com/uc?id=1nKeq1weK407tVKFb8DDf3wDVOB1U1QHk" style="width: 600px;"/>

Hệ thống AutoPilot của Tesla sử dụng nhiều cảm biến, radar, camera và máy tính để thu thập thông tin về môi trường xung quanh và hướng dẫn xe điều khiển an toàn.
Mặc dù được gọi là "lái tự động," AutoPilot vẫn cần sự giám sát của người lái và có thể yêu cầu họ can thiệp trong một số tình huống.

### 2.2. Siri - Apple

Siri là trợ lý ảo phát triển bởi Apple, được tích hợp trên các thiết bị của họ như iPhone, iPad, Mac và các sản phẩm khác.
Siri hoạt động dựa trên trí tuệ nhân tạo để hiểu và thực hiện các lệnh và câu hỏi của người dùng thông qua giọng nói.

<img src="https://drive.google.com/uc?id=16M_8nJ6CdrdGNHu2Hd3kCaaA6ZVerfnW" style="width: 600px;"/>

Siri có khả năng thực hiện nhiều nhiệm vụ, bao gồm gửi tin nhắn, thực hiện cuộc gọi, kiểm tra thời tiết, dự báo giao thông, tìm kiếm thông tin trên mạng, mở ứng dụng, đặt báo thức, và thậm chí điều khiển các thiết bị trong nhà thông qua HomeKit.

Với các phiên bản cập nhật liên tục, Siri ngày càng được cải tiến về khả năng hiểu và phản hồi tự nhiên, từ việc nhận dạng giọng nói đến khả năng hiểu ngữ cảnh và mục đích của người dùng.
Siri đã trở thành một phần quan trọng trong việc tương tác với các thiết bị của Apple, mang lại sự tiện ích và tương tác trực quan cho người dùng.

### 2.3. AlphaGo - Google Deepmind

AlphaGo là một hệ thống trí tuệ nhân tạo phát triển bởi Google DeepMind, được thiết kế để chơi cờ vây. Nó đã gây tiếng vang lớn trong cộng đồng khoa học máy tính và cờ vây khi năm 2016, AlphaGo đã đánh bại Lee Sedol - một trong những kỳ thủ hàng đầu thế giới - trong một trận đấu cờ vây trực tiếp.

<img src="https://drive.google.com/uc?id=1Uo5mjLBRjJrkON-XVWJtaZXvLj08gaS9" style="width: 600px;"/>

AlphaGo không chỉ đơn thuần là một chương trình máy tính chơi cờ vây, mà còn ứng dụng kỹ thuật học sâu và học tăng cường để phân tích và đưa ra quyết định trong trò chơi.
Hệ thống này đã đạt được một khả năng đánh cờ đáng kinh ngạc, thậm chí khi đối mặt với những nước đi sáng tạo và phức tạp mà trước đây được xem là khó khăn đối với máy tính.

Sự thành công của AlphaGo đã thể hiện sự tiến bộ đáng kinh ngạc trong lĩnh vực trí tuệ nhân tạo và Machine Learning, đồng thời mở ra nhiều cơ hội mới trong việc áp dụng trí tuệ nhân tạo vào các lĩnh vực khác nhau.

### 2.4. Social Credit Score - China government


Hệ thống điểm tín dụng xã hội (Social Credit Score) của Chính phủ Trung Quốc là một chương trình theo dõi và đánh giá hành vi của công dân và doanh nghiệp dựa trên nhiều tiêu chí khác nhau.
Mục tiêu chính của chương trình là tạo ra một hệ thống xác định mức độ đáng tin cậy của các cá nhân và tổ chức trong xã hội, từ đó ảnh hưởng đến quyền lợi và ưu đãi mà họ có được.

<img src="https://drive.google.com/uc?id=1qqQDrQcxRiUT3jB0Y-iqYxJq0RzL_j_b" style="width: 600px;"/>

Hệ thống này sử dụng thông tin từ nhiều nguồn khác nhau như hành vi mua sắm, thanh toán tài chính, việc thực hiện nghĩa vụ công dân, và hoạt động trực tuyến để tính toán điểm tín dụng của mỗi cá nhân và doanh nghiệp.
Những người có điểm cao có thể được hưởng ưu đãi như vay tiền dễ dàng hơn, du lịch dễ dàng hơn, và nhiều quyền lợi khác.
Ngược lại, những người có điểm thấp có thể gặp khó khăn trong việc nhận vay hoặc thậm chí bị hạn chế trong việc di chuyển và hoạt động kinh doanh.

Hệ thống điểm tín dụng xã hội của Trung Quốc đã nhận nhiều ý kiến trái chiều, với một số người cho rằng nó có thể đảm bảo tính trật tự và đạo đức trong xã hội, trong khi những người khác lo ngại về việc xâm phạm quyền riêng tư và nguy cơ rơi vào việc kiểm soát quá mức từ phía chính quyền.

## 3. Workflow để xây dựng được mô hình Machine Learning

### Bước 1. Thu thập dữ liệu
Dữ liệu là chìa khóa cho ML.
Dữ liệu cần được thu thập, làm sạch và chuẩn hóa để làm việc hiệu quả.

### Bước 2. Chọn mô hình
Dựa trên bài toán, chúng ta chọn một mô hình ML thích hợp: Decision Trees, Neural Networks, Support Vector Machines, v.v.

### Bước 3. Huấn luyện mô hình
Chúng ta sẽ cung cấp dữ liệu cho mô hình, để nó học cách thực hiện dự đoán chính xác.

### Bước 4. Đánh giá và điều chỉnh
Mô hình cần phải được đánh giá bằng cách sử dụng dữ liệu kiểm tra hoặc kỹ thuật cross-validation.
Nếu hiệu suất chưa tốt, chúng ta điều chỉnh tham số hoặc chọn lại mô hình.

### Bước 5. Dự đoán và triển khai
Cuối cùng, mô hình được sử dụng để dự đoán kết quả cho dữ liệu mới và triển khai trong môi trường thực tế.

<img src="https://drive.google.com/uc?id=1zwu90nvOm7UiQtQjcLZRdgQjm-sscUbv" style="width: 1000px;"/>

## 4. Các nhóm mô hình Machine Learning

<img src="https://drive.google.com/uc?id=1WfBsHYF-F-U4zIh_NtKTIg2eIHMmWUru" style="width: 1200px;"/>

### 4.1. Nhóm mô hình Học có giám sát - Supervised learning

Trong loại này, chúng ta cung cấp cho mô hình các cặp dữ liệu đầu vào và kết quả tương ứng.
Mô hình sẽ học cách ánh xạ từ dữ liệu vào kết quả và sau đó có thể dự đoán kết quả cho dữ liệu mới.

<img src="https://drive.google.com/uc?id=1j3KzbxZ-SX2QVr8uwWkqw0LbeG232a7W" style="width: 800px;"/>

#### 4.1.1. Nhóm mô hình giải quyết bài toán Phân lớp - Classification

Bài toán phân loại (Classification) là một trong những bài toán quan trọng trong lĩnh vực Machine Learning.
Trong bài toán này, mục tiêu là xây dựng một mô hình có khả năng dự đoán lớp hoặc nhãn của dữ liệu mới dựa trên dữ liệu đã biết từ trước.
Cụ thể, mô hình được huấn luyện từ một tập dữ liệu mẫu với thông tin đã biết về các lớp hoặc nhãn, và sau đó được sử dụng để dự đoán lớp hoặc nhãn cho dữ liệu không biết.

<img src="https://drive.google.com/uc?id=10ZB7TCfsq9iZCxiE89aEEuATuA0XSTnz" style="width: 500px;"/>

<img src="https://drive.google.com/uc?id=1CNXniN2LoQtShMhh45I2qS5EqFGox1KN" style="width: 500px;"/>

<img src="https://drive.google.com/uc?id=1UGUG1t6JMMREJy0pGIAR0rOXPsHsuNjH" style="width: 1000px;"/>

Bài toán phân loại yêu cầu xây dựng mô hình có khả năng tự học cách phân biệt các đặc trưng quan trọng của từng lớp và áp dụng những hiểu biết này để đưa ra dự đoán chính xác với dữ liệu mới.

#### 4.1.2. Nhóm mô hình giải quyết bài toán Hồi quy - Regression

Bài toán hồi quy (Regression) là một trong những bài toán quan trọng trong lĩnh vực Machine Learning.
Mục tiêu chính của bài toán này là dự đoán một giá trị liên tục dựa trên các dữ liệu đầu vào đã biết trước.
Thay vì dự đoán lớp hoặc nhãn như trong bài toán phân loại, bài toán hồi quy tập trung vào việc tìm mối quan hệ giữa các biến đầu vào và giá trị đầu ra.

<img src="https://drive.google.com/uc?id=1fbxwkdJIq0zjw0VITS_P7tZaoSPKnpCx" style="width: 500px;"/>

<img src="https://drive.google.com/uc?id=1IftmRIdBaEaIJ0wpEfzVsA-o5UrDyGi8" style="width: 500px;"/>

<img src="https://drive.google.com/uc?id=1fJnIu5BHIS8qRBmBWRdXpzoqW1wzmn_A" style="width: 500px;"/>

<img src="https://drive.google.com/uc?id=1m-kLE6lYBUNNLC9pMvPS6tY9-JfwI9ea" style="width: 1000px;"/>

Để giải quyết bài toán hồi quy, chúng ta xây dựng một mô hình có khả năng tìm ra mối quan hệ giữa các biến đầu vào và giá trị đầu ra.

### 4.2. Nhóm mô hình Học không giám sát - Unsupervised learning

Trong loại này, chúng ta không cung cấp kết quả cho mô hình.
Thay vào đó, mô hình tự tìm kiếm mẫu, cấu trúc hoặc nhóm trong dữ liệu.

<img src="https://drive.google.com/uc?id=1SgrIRFHo6Ok_AvRvll90b7OEin1E78GZ" style="width: 800px;"/>

#### 4.2.1. Nhóm mô hình giải quyết bài toán Phân cụm - Clustering

Bài toán phân cụm (Clustering) là một khía cạnh quan trọng trong lĩnh vực Machine Learning.
Mục tiêu chính của bài toán này là tự động nhóm các điểm dữ liệu có đặc trưng tương tự vào các nhóm hoặc cụm khác nhau.
Trong bài toán này, chúng ta không biết trước lớp hoặc nhãn của các điểm dữ liệu, mà chỉ tìm cách xác định sự tương đồng giữa chúng.

<img src="https://drive.google.com/uc?id=1rkD_QoO1ZL7qy1birBVPeZZH0CA79KcD" style="width: 500px;"/>

<img src="https://drive.google.com/uc?id=1kSDvfyimsbftX1_Jnd7tiRhAFrbcy-VA" style="width: 500px;"/>

Tóm lại, bài toán phân cụm là quá trình tìm cách nhóm các điểm dữ liệu tương tự lại với nhau, dựa vào các đặc trưng chung mà không cần biết trước thông tin về lớp hoặc nhãn.

#### 4.2.2. Nhóm mô hình giải quyết bài toán Giảm chiều dữ liệu - Dimension Reduction

Bài toán giảm chiều dữ liệu (Dimension Reduction) là một phần quan trọng trong Machine Learning, hướng tới việc giảm số lượng biến đầu vào trong dữ liệu mà vẫn giữ lại thông tin quan trọng.
Mục tiêu chính của bài toán này là giảm chiều của dữ liệu mà vẫn duy trì tính chất quan trọng và giảm thiểu sự phức tạp.

<img src="https://drive.google.com/uc?id=1ugl1mHyv3WJzrdubby7XXUH8S0wFGB2g" style="width: 500px;"/>

<img src="https://drive.google.com/uc?id=1pguwbNIoVh6fdMgat5xv54KVWmAfXHL2" style="width: 500px;"/>

### 4.3. Nhóm mô hình Học tăng cường - Reinforcement learning

Loại này làm việc dựa trên hệ thống thưởng và phạt.
Mô hình học từ các tương tác với môi trường và cố gắng tối ưu hóa việc ra quyết định để đạt được mục tiêu.

<img src="https://drive.google.com/uc?id=1l_EdOkXuzZIXRSLIB6awOftxktRZpHlA" style="width: 500px;"/>

Tóm lại, bài toán giảm chiều dữ liệu nhằm mục tiêu giảm số chiều của dữ liệu mà vẫn duy trì thông tin quan trọng, từ đó giúp cải thiện hiệu suất và tốc độ xử lý trong các tác vụ Machine Learning.

## 5. Thư viện Sklearn

Thư viện Scikit-learn (hay sklearn) là một trong những thư viện Machine Learning phổ biến và mạnh mẽ dành cho ngôn ngữ lập trình Python.
Scikit-learn cung cấp một loạt các công cụ và thuật toán cho việc xây dựng và đào tạo các mô hình Machine Learning một cách dễ dàng và hiệu quả.

Một số đặc điểm chính của thư viện Scikit-learn:
- **Dễ sử dụng:** Scikit-learn thiết kế với mục tiêu đơn giản hóa quá trình xây dựng mô hình. Cú pháp đơn giản và tài liệu phong phú giúp người dùng nhanh chóng làm quen và sử dụng thư viện.
- **Thuật toán đa dạng:** Scikit-learn cung cấp nhiều loại thuật toán Machine Learning như học có giám sát, học không giám sát, phân loại, hồi quy, phân cụm, và giảm chiều dữ liệu. Điều này cho phép người dùng lựa chọn và thử nghiệm các phương pháp khác nhau dựa trên bài toán cụ thể.
- **Tích hợp tốt:** Scikit-learn tích hợp với các thư viện Python khác như NumPy, pandas và Matplotlib, giúp người dùng dễ dàng làm việc với dữ liệu và hiển thị kết quả.
- **Công cụ đánh giá và tinh chỉnh mô hình:** Thư viện này cung cấp các công cụ để đánh giá hiệu suất của mô hình, tối ưu hóa tham số và thậm chí tự động tìm kiếm siêu tham số để cải thiện kết quả.
- **Hướng dẫn chi tiết:** Scikit-learn đi kèm với tài liệu phong phú, bao gồm ví dụ minh họa và hướng dẫn sử dụng, giúp người dùng hiểu rõ cách sử dụng các chức năng và thuật toán.

Tóm lại, Scikit-learn là một thư viện quan trọng trong cộng đồng Machine Learning của Python, mang đến sự tiện lợi và hiệu quả trong việc xây dựng các mô hình và tác vụ liên quan đến dữ liệu.

Ta có thể cài thư viên Sklearn vào môi trường conda thông qua câu lệnh

``` bash
conda install -c anaconda scikit-learn -y
```

<img src="https://drive.google.com/uc?id=1XvS74lDRR3nnTfF_UmDK8qOVoBvgyJQd" style="width: 1200px;"/>

<img src="https://drive.google.com/uc?id=1thkqLLZSeg6miG5wWRGxUMRa7wIQr5-T" style="width: 1200px;"/>
