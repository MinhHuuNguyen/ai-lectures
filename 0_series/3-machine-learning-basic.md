---
time: 07/05/2022
title: "[SERIES] Machine Learning cơ bản"
description: Machine Learning là một lĩnh vực nghiên cứu trong trí tuệ nhân tạo, mà mục tiêu là phát triển các kỹ thuật giúp máy tính học từ dữ liệu. Bài viết này sẽ tổng hợp danh sách một số kiến thức cơ bản nhất về Machine Learning như các thuật toán ML cơ bản, cách chia dữ liệu, cách đánh giá mô hình, cách tinh chỉnh mô hình ...
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/ai_vs_ml_vs_dl.jpeg
tags: [machine-learning, series]
is_highlight: false
is_published: true
---

---

### [Bài 0: Giới thiệu chung về Trí tuệ nhân tạo Artificial Intelligence](/blog/gioi-thieu-chung-ve-tri-tue-nhan-tao-artificial-intelligence)

Trí tuệ nhân tạo - Artificial Intelligence là một lĩnh vực nghiên cứu và ứng dụng các phương pháp máy tính để mô phỏng và mở rộng khả năng tư duy của con người. Trong những năm gần đây, AI đã phát triển mạnh mẽ và đóng góp rất nhiều ứng dụng từ hỗ trợ con người đến tự động hóa các công việc. Bài giới thiệu này sẽ giới thiệu chung về Trí tuệ nhân tạo - AI, Máy học - Machine Learning và Học sâu - Deep Learning.

###### 1. Một số ví dụ nổi tiếng về Trí tuệ nhân tạo

###### 2. Trí tuệ nhân tạo - Artificial Intelligence, Máy học - Machine Learning và Học sâu - Deep Learning

###### 3. So sánh giữa Lập trình truyền thống và Machine Learning

###### 4. Xây dựng Trí tuệ nhân tạo như thế nào?

---

### [Bài 1: Các nhóm bài toán và mô hình Machine Learning](/blog/cac-nhom-bai-toan-va-mo-hinh-machine-learning)

Machine learning được chia thành nhiều nhóm bài toán và mô hình khác nhau, mỗi nhóm giải quyết một loại bài toán cụ thể. Bài viết này sẽ giới thiệu về các nhóm bài toán và mô hình Machine Learning phổ biến và quy trình đơn giản nhất để xây dựng mô hình ML.

###### 1. Các nhóm chính của bài toán và mô hình Machine Learning

###### 2. Nhóm Học có giám sát - Supervised learning

###### 3. Nhóm Học không giám sát - Unsupervised learning

###### 4. Nhóm Học tăng cường - Reinforcement learning

###### 5. Nhóm Học bán giám sát - Semi-supervised learning và Học tự giám sát - Self-supervised learning

###### 6. Workflow để xây dựng được mô hình Machine Learning

###### 7. Thư viện Scikit-learn

---

### [Bài 2: Mô hình hồi quy tuyến tính Linear Regression](/blog/mo-hinh-hoi-quy-tuyen-tinh-linear-regression)

Mô hình Linear Regression là một trong những mô hình đơn giản nhất trong các Machine Learning. Mô hình Linear Regression thường được sử dụng để dự đoán giá trị của một biến liên tục dựa trên một hoặc nhiều biến đầu vào.

###### 1. Bài toán dự đoán giá nhà

###### 2. Kiến trúc mô hình

###### 3. Huấn luyện mô hình

###### 4. Loss function của mô hình Linear Regression

###### 5. Phương pháp tối ưu mô hình

###### 6. Chỉ số đánh giá mô hình

---

### [Bài 3: Thuật toán tối ưu Gradient Descent](/blog/thuat-toan-toi-uu-gradient-descent)

Trong Machine Learning, ta thường dùng phép toán đạo hàm để tối ưu hàm loss. Tuy nhiên, trong đa số các trường hợp, việc tính đạo hàm của hàm loss là không thể hoặc rất khó khăn, đặc biệt khi kiến trúc mô hình phức tạp và bộ dữ liệu lớn. Trong bài viết này, chúng ta sẽ tìm hiểu về thuật toán tối ưu Gradient Descent, một trong những phương pháp phổ biến nhất để tối ưu hàm loss trong Machine Learning.

###### 1. Nhắc lại về bài toán Khảo sát hàm số

###### 2. Ý tưởng của thuật toán Gradient Descent

###### 3. Ảnh hưởng của các tham số trong thuật toán Gradient Descent

---

### [Bài 4: Mô hình hồi quy logistic Logistic Regression](/blog/mo-hinh-hoi-quy-logistic-logistic-regression)

Mô hình hồi quy logistic Logistic regression
description: Mô hình Linear regression là mô hình đơn giản để giải quyết bài toán regression, còn đối với bài toán classification, ta có mô hình Logistic regression. Mô hình Logistic regression có thể giải quyết bài toán phân lớp nhị phân (binary classification), bài toán phân lớp nhiều label (multi-label classification) và bài toán phân lớp nhiều lớp (multi-class classification).

###### 1. Bài toán phân lớp nhị phân - Binary classification

###### 2. Ý tưởng chung của logistic regression

###### 3. Hàm kích hoạt Sigmoid

###### 4. Bài toán phân lớp nhiều label - Multi-label classification

###### 5. Bài toán phân lớp nhiều lớp - Multi-class classification

###### 6. Hàm kích hoạt Softmax

###### 7. Hàm kích hoạt Tanh

---

### [Bài 5: Metrics đánh giá cho bài toán classification](/blog/metrics-danh-gia-cho-bai-toan-classification)

Bài toán classification là một trong những bài toán phổ biến nhất trong machine learning. Để đánh giá được chất lượng của mô hình sao cho chính xác và khách quan nhất, ta cần xây dựng bộ các metrics đánh giá cho bài toán classification. Có nhiều metrics khác nhau cho bài toán classification, các chỉ số này có những điểm mạnh và điểm yếu riêng. Trong bài viết này, ta sẽ cùng nhau tìm hiểu về các metrics đánh giá cho bài toán classification.

###### 1. Chỉ số Accuracy

###### 2. Bài toán về sản phẩm tốt và sản phẩm lỗi trong hàng hoá

###### 3. Confusion matrix

###### 4. Precision - Recall - F score - Specificity

###### 5. Confusion matrix trong bài toán multi-class classification

---

### [Bài 6: Hiện tượng Overfit và Underfit](/blog/hien-tuong-overfit-va-underfit)

Trong quá trình huấn luyện mô hình machine learning, ta thường gặp phải hiện tượng overfit và underfit. Hai hiện tượng này khiến cho việc huấn luyện mô hình gặp nhiều khó khăn và gây ra sự sai sót trong quá trình đánh giá mô hình.

###### 1. Tương quan giữa dữ liệu và mô hình trong quá trình huấn luyện

###### 2. Hiện tượng Underfitting

###### 3. Hiện tượng Overfitting

###### 4. Kỹ thuật Regularization

###### 5. Vai trò của bộ dữ liệu validation

---

### [Bài 7: Mô hình K-Nearest Neighbors (KNN)](/blog/mo-hinh-k-nearest-neighbors-knn)

KNN là một trong những mô hình machine learning đơn giản và dễ hiểu nhất, có thể được sử dụng cho cả bài toán phân loại và hồi quy.

###### 1. Tổng quan

###### 2. Các bước của thuật toán

###### 3. Công thức tính khoảng cách

###### 4. Ưu điểm và nhược điểm của KNN

###### 5. Các biến thể nâng cấp của KNN

---

### [Bài 8: Mô hình Support Vector Machine (SVM)](/blog/mo-hinh-support-vector-machine-svm)

SVM là mô hình machine learning dựa vào khoảng cách giữa các điểm dữ liệu và đường phân lớp để tìm ra được đường phân lớp tốt nhất. SVM thường là mô hình phân lớp có độ chính xác cao hơn so với mô hình Logistic Regression (mạng nơ ron với 1 layer).

###### 1. Tổng quan

###### 2. Công thức tính khoảng cách từ 1 điểm

###### 3. Mô hình SVM - Hard Margin SVM

###### 4. Tối ưu mô hình Hard Margin SVM

###### 5. Mô hình Soft Margin SVM

###### 6. Mô hình Kernel SVM

---

### [Bài 9: Mô hình Naive Bayes Classification](/blog/mo-hinh-naive-bayes-classification)

Naive Bayes là một mô hình machine learning điển hình, đại diện cho các mô hình dựa vào xác suất thống kê. Mô hình này được sử dụng rộng rãi trong các bài toán phân loại, đặc biệt là phân loại văn bản

###### 1. Tổng quan

###### 2. Một số kiến thức trong xác suất thống kê

###### 3. Mô hình Naive Bayes Classification

###### 4. Ví dụ minh hoạ

###### 5. Ưu và nhược điểm của Naive Bayes Classification

---

### [Bài 10: Mô hình Decision Tree](/blog/mo-hinh-decision-tree)

Decision Tree là một trong những mô hình học máy khá cổ điển nhưng vẫn được sử dụng rất nhiều trong thực tế. Mô hình này có thể được sử dụng cho cả bài toán phân loại và hồi quy. Hơn nữa, Decision Tree cũng là một trong những mô hình dễ hiểu và dễ giải thích nhất trong các mô hình học máy.

###### 1. Tổng quan

###### 2. Các khái niệm trong Decision Tree

###### 3. Hàm Entropy

###### 4. Decision Tree với hàm Entropy

###### 5. Ví dụ minh hoạ

###### 6. Decision Tree với bài toán hồi quy

###### 7. Ưu và nhược điểm của mô hình

---

### [Bài 11: Mô hình K-Means Clustering](/blog/mo-hinh-k-means-clustering)

Bên cạnh các mô hình học có giám sát, mô hình học không giám sát cũng đóng một vai trò quan trọng trong Machine Learning. Trong bài viết này, chúng ta sẽ tìm hiểu về mô hình phân cụm K-means Clustering, mô hình giúp phân chia dữ liệu thành các cụm dựa trên đặc trưng của chúng.

###### 1. Tổng quan

###### 2. Các bước của thuật toán

###### 3. Công thức tính khoảng cách

###### 4. Tối ưu trong mô hình K-means

###### 5. Phương pháp lựa chọn số lượng cụm

###### 6. Ưu và nhược điểm của mô hình

###### 7. Các biến thể nâng cấp của mô hình

---

### [Bài 12: Mô hình DBSCAN](/blog/mo-hinh-dbscan)

Khác với K-means Clustering, mô hình phân cụm DBSCAN Clustering không yêu cầu số lượng cụm cần phân chia trước. Trong bài viết này, chúng ta sẽ tìm hiểu về mô hình DBSCAN Clustering, mô hình giúp phân chia dữ liệu thành các cụm dựa trên mật độ của chúng.

###### 1. Tổng quan

###### 2. Các khái niệm được định nghĩa trong mô hình DBSCAN

###### 3. Các bước của thuật toán

###### 4. Ưu và nhược điểm của mô hình

###### 5. Mẹo lựa chọn các tham số đầu vào cho mô hình

###### 6. Các biến thể nâng cấp của mô hình

---

### [Bài 13: Mô hình PCA](/blog/mo-hinh-pca)

Làm việc trực tiếp trên dữ liệu có số chiều cao gây ra khó khăn cả về việc lưu trữ và tốc độ tính toán. Do đó, giảm chiều dữ liệu là một bài toán có tính ứng dụng cao trong Machine Learning, giúp lưu trữ và xử lý dữ liệu với hiệu năng tốt hơn. PCA là mô hình giảm chiều dữ liệu đại diện cho nhóm các mô hình tuyến tính, dựa vào các phép toán trên ma trận để giảm chiều dữ liệu.

###### 1. Tổng quan

###### 2. Giá trị riêng và vector riêng (Eigenvalues và Eigenvectors)

###### 3. Ma trận hiệp phương sai (Covariance matrix)

###### 4. Các bước của thuật toán

###### 5. Ví dụ minh hoạ

###### 6. Ưu và nhược điểm của mô hình

###### 7. Các biến thể nâng cấp của mô hình

---

### [Bài 14: Mô hình tSNE](/blog/mo-hinh-tsne)

Các feature vectors trong các bài toán machine learning thực tế có thể có số chiều rất lớn và số lượng các điểm dữ liệu cũng lớn dần theo thời gian. Điều này có thể được gọi là Curse of Dimensionality, Lời nguyền của số chiều. Trong các thuật toán giảm chiều dữ liệu, t-SNE là một đại diện nổi bật cho phương pháp giảm chiều dữ liệu phi tuyến tính.

###### 1. Tổng quan

###### 2. Công thức tính khoảng cách KL divergence

###### 3. Các bước của thuật toán

###### 4. Ưu và nhược điểm của mô hình

###### 5. Các biến thể nâng cấp của mô hình

---

### [Bài 15: Mô hình mạng nơ ron đơn giản Neural network](/blog/mo-hinh-mang-no-ron-don-gian-neural-network)

Mô hình mạng nơ ron đơn giản Neural network là một mô hình tính toán lấy cảm hứng từ cấu trúc và hoạt động của bộ não con người. Mô hình mạng nơ ron đơn giản là nền tảng cho sự phát triển của các mô hình mạng nơ ron phức tạp hơn được sử dụng trong các mô hình Trí tuệ nhân tạo nổi tiếng hiện nay.

###### 1. Tổng quan

###### 2. Mối quan hệ giữa hàm XOR và mạng nơ ron

###### 3. Kiến trúc và các lớp trong mạng nơ ron

###### 4. Huấn luyện mạng nơ ron

---

### [Bài 16: Mô hình LDA](/blog/mo-hinh-lda)

*Mình sẽ viết bài này trong thời gian tới*

---

### [Bài 17: Kỹ thuật Ensemble Learning](/blog/ky-thuat-ensemble-learning)

*Mình sẽ viết bài này trong thời gian tới*

###### 1. Ý tưởng chung của Ensemble Learning

###### 2. Phân loại kỹ thuật Ensemble Learning

###### 3. Mô hình Random Forest

###### 4. Mô hình GBM và XGBoost
