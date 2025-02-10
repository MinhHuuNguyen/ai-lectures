---
time: 11/09/2024
title: Các nhóm bài toán và mô hình Machine Learning
description: Machine learning được chia thành nhiều nhóm bài toán và mô hình khác nhau, mỗi nhóm giải quyết một loại bài toán cụ thể. Bài viết này sẽ giới thiệu về các nhóm bài toán và mô hình Machine Learning phổ biến và quy trình đơn giản nhất để xây dựng mô hình ML.
banner_url: 
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Các nhóm chính của bài toán và mô hình Machine Learning

Machine learning được chia thành một số nhóm chính dựa trên cách mà mô hình học từ dữ liệu và cách chúng được sử dụng để giải quyết các bài toán cụ thể.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/problems_algorithms.jpeg" style="width: 1200px;"/>

- **Học có giám sát (Supervised learning):** Trong loại này, mô hình được huấn luyện trên dữ liệu đã được gán nhãn. Mục tiêu là học cách ánh xạ từ dữ liệu đầu vào đến kết quả tương ứng.
- **Học không giám sát (Unsupervised learning):** Trong loại này, mô hình học từ dữ liệu không được gán nhãn. Mục tiêu là tìm cấu trúc ẩn trong dữ liệu mà không cần biết trước thông tin về lớp hoặc nhãn.
- **Học tăng cường (Reinforcement learning):** Trong loại này, mô hình học từ tương tác với môi trường và cố gắng tối ưu hóa việc ra quyết định để đạt được mục tiêu.
- **Học bán giám sát (Semi-supervised learning) và Học tự giám sát (Self-supervised learning):** Loại này kết hợp giữa học có giám sát và học không giám sát, hoặc học từ dữ liệu không được gán nhãn một cách tự động.

## 2. Nhóm Học có giám sát - Supervised learning

Trong loại này, chúng ta cung cấp cho mô hình các cặp dữ liệu đầu vào và kết quả tương ứng.
Mô hình sẽ học cách ánh xạ từ dữ liệu vào kết quả và sau đó có thể dự đoán kết quả cho dữ liệu mới.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/supervised.jpeg" style="width: 1200px;"/>

### 2.1. Bài toán Phân loại - Classification

Bài toán phân loại (Classification) là một trong những bài toán quan trọng trong lĩnh vực Machine Learning.
Trong bài toán này, mục tiêu là xây dựng một mô hình có khả năng dự đoán lớp hoặc nhãn của dữ liệu mới dựa trên dữ liệu đã biết từ trước.
Cụ thể, mô hình được huấn luyện từ một tập dữ liệu mẫu với thông tin đã biết về các lớp hoặc nhãn, và sau đó được sử dụng để dự đoán lớp hoặc nhãn cho dữ liệu không biết.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/classification.jpeg" style="width: 1200px;"/>

Bài toán phân loại yêu cầu xây dựng mô hình có khả năng tự học cách phân biệt các đặc trưng quan trọng của từng lớp và áp dụng những hiểu biết này để đưa ra dự đoán chính xác với dữ liệu mới.

### 2.2. Bài toán Hồi quy - Regression

Bài toán hồi quy (Regression) là một trong những bài toán quan trọng trong lĩnh vực Machine Learning.
Mục tiêu chính của bài toán này là dự đoán một giá trị liên tục dựa trên các dữ liệu đầu vào đã biết trước.
Thay vì dự đoán lớp hoặc nhãn như trong bài toán phân loại, bài toán hồi quy tập trung vào việc tìm mối quan hệ giữa các biến đầu vào và giá trị đầu ra.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/regression.jpeg" style="width: 1200px;"/>

Để giải quyết bài toán hồi quy, chúng ta xây dựng một mô hình có khả năng tìm ra mối quan hệ giữa các biến đầu vào và giá trị đầu ra.

## 3. Nhóm Học không giám sát - Unsupervised learning

Trong loại này, chúng ta không cung cấp kết quả cho mô hình.
Thay vào đó, mô hình tự tìm kiếm mẫu, cấu trúc hoặc nhóm trong dữ liệu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/unsupervised.jpeg" style="width: 1200px;"/>

### 3.1. Bài toán Phân cụm - Clustering

Bài toán phân cụm (Clustering) là một khía cạnh quan trọng trong lĩnh vực Machine Learning.
Mục tiêu chính của bài toán này là tự động nhóm các điểm dữ liệu có đặc trưng tương tự vào các nhóm hoặc cụm khác nhau.
Trong bài toán này, chúng ta không biết trước lớp hoặc nhãn của các điểm dữ liệu, mà chỉ tìm cách xác định sự tương đồng giữa chúng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/clustering.jpeg" style="width: 1200px;"/>

Tóm lại, bài toán phân cụm là quá trình tìm cách nhóm các điểm dữ liệu tương tự lại với nhau, dựa vào các đặc trưng chung mà không cần biết trước thông tin về lớp hoặc nhãn.

### 3.2. Bài toán Giảm chiều dữ liệu - Dimension Reduction

Bài toán giảm chiều dữ liệu (Dimension Reduction) là một phần quan trọng trong Machine Learning, hướng tới việc giảm số lượng biến đầu vào trong dữ liệu mà vẫn giữ lại thông tin quan trọng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/dimension-reduction.jpeg" style="width: 1200px;"/>

Mục tiêu chính của bài toán này là giảm chiều của dữ liệu mà vẫn duy trì tính chất quan trọng và giảm thiểu sự phức tạp.

## 4. Nhóm Học tăng cường - Reinforcement learning

Loại này làm việc dựa trên hệ thống thưởng và phạt.
Mô hình học từ các tương tác với môi trường và cố gắng tối ưu hóa việc ra quyết định để đạt được mục tiêu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/reinforcement.jpeg" style="width: 1200px;"/>

Tóm lại, bài toán giảm chiều dữ liệu nhằm mục tiêu giảm số chiều của dữ liệu mà vẫn duy trì thông tin quan trọng, từ đó giúp cải thiện hiệu suất và tốc độ xử lý trong các tác vụ Machine Learning.

## 5. Nhóm Học bán giám sát - Semi-supervised learning và Học tự giám sát - Self-supervised learning

Loại này kết hợp giữa học có giám sát và học không giám sát, hoặc học từ dữ liệu không được gán nhãn một cách tự động.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/semi-self-supervised.jpeg" style="width: 1200px;"/>

Tóm lại, học bán giám sát và học tự giám sát là những phương pháp kết hợp giữa học có giám sát và học không giám sát, giúp cải thiện hiệu suất và khả năng tự học của mô hình trong khi giảm thiểu chi phí thu thập, xử lý và gán nhãn dữ liệu.

## 6. Workflow để xây dựng được mô hình Machine Learning

Để xây dựng một mô hình Machine Learning hiệu quả, chúng ta cần tuân thủ một quy trình làm việc cụ thể.
Quy trình này bao gồm các bước chính sau:
- Bước 1: **Thu thập dữ liệu**: Bước này bao gồm Data Extraction - Trích xuất dữ liệu từ nhiều nguồn khác nhau và Data Collection - Thu thập và lưu trữ vào trong các kho dữ liệu hoặc cơ sở dữ liệu.
- Bước 2: **Xử lý dữ liệu**: Dữ liệu sau khi được lưu trữ có thể được xử lý với các thao tác như:
    - Data labelling - Gán nhãn dữ liệu: Dữ liệu được đưa vào trong các Annotation Tool - Công cụ gán nhãn dữ liệu để con người gán nhãn.
    - Data analysis - Phân tích dữ liệu: Dữ liệu được phân tích để hiểu rõ hơn về đặc trưng và cấu trúc của dữ liệu.
    - Data preprocessing - Tiền xử lý dữ liệu: Dữ liệu được tiền xử lý để chuẩn hóa, làm sạch và chuẩn bị cho quá trình huấn luyện mô hình.
    Dữ liệu sau khi được tiền xử lý sẽ được chia thành các tập dữ liệu con như training set, validation set và test set để sử dụng trong quá trình training và evaluate mô hình.
- Bước 3: **Huấn luyện mô hình**: Ta lự chọn một mô hình phù hợp và huấn luyện mô hình trên tập dữ liệu đã được tiền xử lý ở trên.
    - Trong giai đoạn huấn luyện - training phase, mô hình sẽ được học trên bộ dữ liệu train và được đánh giá trên bộ dữ liệu validation để điều chỉnh tham số, tối ưu hoá độ chính xác.
    - Thông thường, sau khi train và validate, mô hình sẽ được đánh giá trên bộ dữ liệu test để đánh giá lần cuối về hiệu suất của mô hình.
    - Quá trình huấn luyện mô hình có thể được lặp lại nhiều lần để cải thiện hiệu suất của mô hình, cũng như việc ta có thể lựa chọn nhiều mô hình khác nhau trong giai đoạn này.
- Bước 4: **Dự đoán và triển khai**: Trong số các mô hình đã được huấn luyện ở trên, ta chọn một mô hình tốt nhất và sử dụng nó để dự đoán kết quả cho dữ liệu mới và triển khai mô hình trong môi trường thực tế.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/workflow.jpeg" style="width: 1200px;"/>

Trong thực tế, quy trình xây dựng mô hình Machine Learning có thể phức tạp hơn và yêu cầu nhiều công việc chi tiết hơn, nhưng quy trình trên là một hướng dẫn cơ bản để bắt đầu với Machine Learning.

## 7. Thư viện Scikit-learn

Thư viện Scikit-learn (hay Sklearn) là một trong những thư viện Machine Learning phổ biến và mạnh mẽ dành cho ngôn ngữ lập trình Python.
Scikit-learn cung cấp một loạt các công cụ và thuật toán cho việc xây dựng và đào tạo các mô hình Machine Learning một cách dễ dàng và hiệu quả.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/sklearn_map.png" style="width: 1200px;"/>

Một số đặc điểm chính của thư viện Scikit-learn:
- **Dễ sử dụng:** Scikit-learn thiết kế với mục tiêu đơn giản hóa quá trình xây dựng mô hình. Cú pháp đơn giản và tài liệu phong phú giúp người dùng nhanh chóng làm quen và sử dụng thư viện.
- **Thuật toán đa dạng:** Scikit-learn cung cấp nhiều loại thuật toán Machine Learning như học có giám sát, học không giám sát, phân loại, hồi quy, phân cụm, và giảm chiều dữ liệu. Điều này cho phép người dùng lựa chọn và thử nghiệm các phương pháp khác nhau dựa trên bài toán cụ thể.
- **Tích hợp tốt:** Scikit-learn tích hợp với các thư viện Python khác như NumPy, pandas và Matplotlib, giúp người dùng dễ dàng làm việc với dữ liệu và hiển thị kết quả.
- **Công cụ đánh giá và tinh chỉnh mô hình:** Thư viện này cung cấp các công cụ để đánh giá hiệu suất của mô hình, tối ưu hóa tham số và thậm chí tự động tìm kiếm siêu tham số để cải thiện kết quả.
- **Hướng dẫn chi tiết:** Scikit-learn đi kèm với tài liệu phong phú, bao gồm ví dụ minh họa và hướng dẫn sử dụng, giúp người dùng hiểu rõ cách sử dụng các chức năng và thuật toán.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/1-machine-learning-introduction/sklearn_cheatsheet.webp" style="width: 1200px;"/>

Tóm lại, Scikit-learn là một thư viện quan trọng trong cộng đồng Machine Learning của Python, mang đến sự tiện lợi và hiệu quả trong việc xây dựng các mô hình và tác vụ liên quan đến dữ liệu.
