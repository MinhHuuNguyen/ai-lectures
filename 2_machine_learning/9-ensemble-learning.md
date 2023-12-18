---
layout: "post"
title:  "Ensemble Learning"
author: "Nguyễn Hữu Minh"
permalink: "/machine-learning/ensemble-learning"
parent: "Machine learning"
nav_order: 8
---

# Ensemble Learning

# 1. Ý tưởng chung:

- Ensemble Learning là một kỹ thuật học máy, kết hợp nhiều mô hình học máy đơn giản để tạo thành một mô hình học máy mạnh hơn.
- Ensemble Learning thường được sử dụng trong các bài toán phân loại.
- Ensemble Learning có thể được sử dụng với các mô hình học máy khác nhau, hoặc với cùng một mô hình học máy nhưng với các siêu tham số khác nhau.

Các ưu điểm của ensemble learning bao gồm:
- Tăng độ chính xác:
Ensemble thường cung cấp kết quả tốt hơn so với mô hình đơn lẻ bằng cách giảm thiểu tình trạng overfitting và cải thiện khả năng tổng hợp thông tin từ nhiều nguồn.
- Ứng dụng linh hoạt:
Các phương pháp ensemble có thể áp dụng cho nhiều loại mô hình khác nhau, giúp tận dụng sức mạnh của các thuật toán khác nhau.

Tuy nhiên, cũng có một số điểm cần lưu ý như tăng chi phí tính toán do cần huấn luyện và dự đoán từ nhiều mô hình, cũng như đôi khi khó khăn trong việc diễn giải kết quả dự đoán.

# 2. Các phương pháp Ensemble Learning:

## 2.1. Bagging (Bootstrap Aggregating):

Bagging xây dựng một lượng lớn các models (thường là cùng loại) trên những subsamples khác nhau từ tập training dataset một cách song song nhằm đưa ra dự đoán tốt hơn.

Bagging là quá trình tạo ra nhiều tập con dữ liệu con khác nhau từ tập dữ liệu huấn luyện bằng cách sử dụng kỹ thuật lấy mẫu có hoàn lại (bootstrap).
Sau đó, mỗi mô hình được huấn luyện trên một tập con dữ liệu này.
Cuối cùng, kết quả của tất cả các mô hình được kết hợp bằng cách lấy trung bình (đối với regression) hoặc bằng cách bình chọn (đối với classification).

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20230731175958/Bagging-classifier.png" style="width: 800px;"/>

Điểm yếu của Bagging:
- Các model trong Bagging đều là học một cách riêng rẽ, không liên quan hay ảnh hưởng gì đến nhau, điều này trong một số trường hợp có thể dẫn đến kết quả tệ khi các model có thể học cùng ra 1 kết quả.
Chúng ta không thể kiểm soát được hướng phát triển của các model con thêm vào bagging
- Chúng ta mong đợi các model yếu của thể hỗ trợ lẫn nhau, học được từ nhau để tránh đi vào các sai lầm của model trước đó.
Đây là điều Bagging không làm được

### 2.2. Boosting:

Boosting xây dựng một lượng lớn các models (thường là cùng loại), tuy nhiên, quá trình huấn luyện trong phương pháp này diễn ra tuần tự theo chuỗi (sequence).

Trái ngược với bagging, boosting tập trung vào việc xây dựng các mô hình theo cách tuần tự, với mỗi mô hình cố gắng sửa lỗi của mô hình trước đó.
Các mô hình yếu (weak learners) được kết hợp để tạo thành một mô hình mạnh mẽ.

AdaBoost (Adaptive Boosting) và Gradient Boosting là các thuật toán boosting phổ biến.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20210707140911/Boosting.png" style="width: 800px;"/>

Đối với AdaBoost, các bước thực hiện như sau:
- **Bước 1:** Khởi tạo trọng số cho tất cả các điểm dữ liệu huấn luyện, thường là 1/n với n là số lượng điểm dữ liệu huấn luyện.
- **Bước 2:** Vòng lặp xây dựng mô hình:
    - Mô hình đầu tiên được huấn luyện trên tập dữ liệu huấn luyện với trọng số đã được khởi tạo.
    - Trọng số của mỗi điểm dữ liệu được cập nhật dựa trên kết quả dự đoán của mô hình đầu tiên (trọng số được tính thông qua sai số dự đoán).
    - Các mô hình tiếp theo được huấn luyện trên tập dữ liệu huấn luyện với trọng số đã được cập nhật.
    - Quá trình lặp lại cho đến khi đạt được số lượng mô hình yêu cầu.

AdaBoost có thể được áp dụng mà không cần dựa vào việc đánh trọng số lại các điểm dữ liệu, thay vào đó, chúng ta có thể re-sample để lấy dữ liệu train cho các model tiếp theo dựa vào xác suất được xác định bới các trọng số.

<img src="https://i.imgur.com/HDCbnL7.png" style="width: 800px;"/>

Đối với Gradient Boosting, các bước thực hiện như sau:
- **Bước 1:** Khởi tạo giá trị pseudo residuals (sai số giả) bằng nhau cho tất cả các điểm dữ liệu huấn luyện.
- **Bước 2:** Vòng lặp xây dựng mô hình:
    - Mô hình đầu tiên được huấn luyện để dự đoán giá trị pseudo residuals đã khởi tạo.
    - Tính toán lại giá trị residuals bằng cách lấy sự chênh lệch label và giá trị dự đoán của mô hình đầu tiên.
    - Các mô hình tiếp theo được huấn luyện để dự đoán giá trị residuals đã tính toán.
    - Quá trình lặp lại cho đến khi đạt được số lượng mô hình yêu cầu.

<img src="https://i.imgur.com/ZE8bBTA.png" style="width: 800px;"/>

Điểm yếu của Boosting:
- Boosting là một quá trình tuần tự, không thể xử lí song song, do đó, thời gian train mô hình có thể tương đối lâu.

### 2.3. Stacking:

Stacking (stacked generalization) là một phương pháp ensemble learning khác, nó cũng nhằm mục đích kết hợp sức mạnh của nhiều mô hình để tạo ra một mô hình mạnh mẽ hơn.
Tuy nhiên, stacking thường tiếp cận một cách linh hoạt hơn bằng cách sử dụng một mô hình cấp cao để học cách kết hợp các đầu ra của các mô hình cấp thấp.

Quy trình của stacking thường như sau:
- **Chia dữ liệu:**
Tập dữ liệu được chia thành hai hoặc nhiều tập con không chồng lấp.
Thông thường, một tập dữ liệu con được sử dụng để huấn luyện các mô hình cấp thấp và một tập dữ liệu con khác được sử dụng để huấn luyện mô hình cấp cao.
- **Huấn luyện mô hình cấp thấp:**
Các mô hình cấp thấp (base models) được huấn luyện trên một tập dữ liệu con.
- **Tạo đầu ra dự đoán:**
Các mô hình cấp thấp được sử dụng để tạo ra các dự đoán trên tập dữ liệu kiểm thử hoặc validation.
- **Huấn luyện mô hình cấp cao:**
Mô hình cấp cao (meta-model hoặc blender) được huấn luyện bằng cách sử dụng đầu ra dự đoán từ các mô hình cấp thấp như là đặc trưng đầu vào.
Mô hình này học cách kết hợp các dự đoán từ các mô hình cấp thấp để tối ưu hóa hiệu suất.
- **Dự đoán mới:**
Khi mô hình cấp cao đã được huấn luyện, nó có thể được sử dụng để dự đoán trên dữ liệu mới.

Các ưu điểm của stacking bao gồm:
- **Tính linh hoạt cao:**
Có thể sử dụng nhiều loại mô hình cấp thấp khác nhau, giúp tận dụng sức mạnh của các thuật toán đa dạng.
- **Hiệu suất cao:**
Stacking thường cung cấp hiệu suất tốt hơn so với việc sử dụng các mô hình cấp thấp độc lập.

Tuy nhiên, stacking cũng đôi khi đòi hỏi nhiều dữ liệu hơn và tốn thời gian và công sức hơn so với một số phương pháp ensemble khác.

<img src="https://www.researchgate.net/profile/Nipaporn-Chanamarn/publication/308368870/figure/fig4/AS:408666126733315@1474445005322/The-concept-diagram-of-stacking-ensemble-learning-32.png" style="width: 600px;"/>

# 3. Một số mô hình nổi bật:

## 3.1. Random Forest:

Random Forest là một thuật toán ensemble learning phổ biến, kết hợp nhiều cây quyết định (decision trees) để tạo thành một mô hình mạnh mẽ hơn.

Cụ thể, Random Forest sử dụng phương pháp "bagging" (bootstrap aggregating) để tạo ra các tập dữ liệu con ngẫu nhiên từ bộ dữ liệu huấn luyện, và sau đó xây dựng nhiều cây quyết định độc lập trên các tập dữ liệu con này.

Kết quả của Random Forest được xác định thông qua quá trình bình chọn (voting) giữa các cây quyết định thành viên.

Một số điểm quan trọng về Random Forest:
- Bộ dữ liệu ngẫu nhiên (Random Data):
Random Forest sử dụng phương pháp bagging (bootstrap aggregating) để tạo ra các tập con dữ liệu ngẫu nhiên từ bộ dữ liệu huấn luyện bằng cách lấy mẫu với hoàn lại.
Điều này giúp đảm bảo tính đa dạng và tránh overfitting.
- Chọn ngẫu nhiên các đặc trưng (Random Feature Selection):
Khi xây dựng mỗi cây quyết định trong Random Forest, chỉ một số lượng ngẫu nhiên các đặc trưng được chọn để xây dựng cây.
Điều này giúp mô hình trở nên linh hoạt hơn và chống lại việc một số đặc trưng quan trọng bị bỏ qua.
- Số lượng cây (Number of Trees):
Random Forest bao gồm một tập hợp các cây quyết định.
Số lượng cây này có thể được xác định bởi người dùng.
Nhiều cây hơn thường dẫn đến hiệu suất tốt hơn, nhưng đồng thời cũng tăng độ phức tạp tính toán.
- Bình chọn (Voting):
Khi thực hiện dự đoán, mỗi cây trong Random Forest đưa ra một dự đoán và cuối cùng, dự đoán cuối cùng là kết quả của việc bình chọn của tất cả các cây.
- Độ quan trọng của đặc trưng:
Random Forest cung cấp một phương pháp để đo lường độ quan trọng của từng đặc trưng trong quá trình huấn luyện.
Điều này giúp hiểu rõ hơn về tầm quan trọng của từng yếu tố đối với mô hình.

<img src="https://tikz.net/janosh/random-forest.png" style="width: 800px;"/>

## 3.2. Light GBM và XGBoost:

### 3.2.1. XGBoost:

XGBoost (Extreme Gradient Boosting) là một thuật toán gradient boosting phổ biến, được phát triển bởi Tianqi Chen.

Một số đặc điểm nổi bật của XGBoost:
- **Regularization:**
XGBoost có thể được sử dụng để giảm overfitting thông qua các phương pháp regularization như thành phần gamma (min_split_loss), thành phần lambda (min_gain_to_split) và thành phần alpha (reg_alpha)
- **Phương Pháp Chia Nhánh:**
XGBoost sử dụng phương pháp chia nhánh "pre-sorted" để tăng tốc quá trình đào tạo.
Dữ liệu được sắp xếp trước để giảm thời gian cần thiết cho việc tìm kiếm các điểm chia nhánh tốt nhất.
- **Xử Lý Dữ Liệu Imbalanced:**
XGBoost có thể được sử dụng để xử lý các bộ dữ liệu không cân bằng thông qua các tham số scale_pos_weight và max_delta_step.
- **Early Stopping:**
XGBoost có thể được sử dụng để ngừng sớm quá trình đào tạo thông qua tham số early_stopping_rounds.
- **Hỗ Trợ GPU:**
XGBoost có thể được sử dụng để tăng tốc quá trình đào tạo thông qua GPU.

<img src="https://i.imgur.com/tixTkYI.png" style="width: 800px;"/>

### 3.2.2. Light GBM:

Light GBM (Light Gradient Boosting Machine) là một thuật toán gradient boosting phổ biến, được phát triển bởi Microsoft.

Một số đặc điểm nổi bật của Light GBM:
- **Cấu Trúc Histogram:**
LightGBM sử dụng cấu trúc histogram để đồng thời giảm kích thước dữ liệu và tăng tốc quá trình đào tạo.
Thay vì sắp xếp dữ liệu theo giá trị, LightGBM phân chia dữ liệu thành các histogram và chỉ sử dụng các giá trị xấp xỉ để đào tạo cây quyết định.
- **Leaf-wise Tree:**
LightGBM xây dựng cây theo kiểu leaf-wise (tương tự với best-first), tức là nó chọn lá cây có sự giảm lỗi lớn nhất trước, điều này giúp giảm độ sâu của cây và tăng tốc quá trình học.
- **Tối Ưu Hóa Song Song (Parallelization):**
LightGBM hỗ trợ tối ưu hóa song song thông qua các tham số num_threads và device_type, trên cả CPU và GPU.
- **Regularization:**
LightGBM cung cấp các tham số để kiểm soát overfitting như max_depth, min_child_samples, và lambda (regularization term).
Điều này giúp người dùng có thể tinh chỉnh mô hình để đạt được sự cân bằng giữa độ chính xác và khả năng tổng quát hóa của mô hình.
