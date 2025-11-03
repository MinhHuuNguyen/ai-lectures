---
time: 07/30/2022
title: Metrics đánh giá cho bài toán classification
description: Bài toán classification là một trong những bài toán phổ biến nhất trong machine learning. Để đánh giá được chất lượng của mô hình sao cho chính xác và khách quan nhất, ta cần xây dựng bộ các metrics đánh giá cho bài toán classification. Có nhiều metrics khác nhau cho bài toán classification, các chỉ số này có những điểm mạnh và điểm yếu riêng. Trong bài viết này, ta sẽ cùng nhau tìm hiểu về các metrics đánh giá cho bài toán classification.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/banner.jpeg
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Chỉ số Accuracy

Accuracy (độ chính xác) là một chỉ số đánh giá hiệu suất của mô hình phân loại trong Machine Learning. Nó đo lường tỷ lệ dự đoán đúng trên tổng số dự đoán.
Nói cách khác, Accuracy cho biết bao nhiêu phần trăm dự đoán của mô hình là đúng.

$$
\text{accuracy} = \frac{\text{number of true predictions}}{\text{number of predictions}}
$$

Accuracy được sử dụng đồng thời trên cả các mô hình giải quyết bài toán multi-label classification và multi-class classification.

Ưu điểm của Accuracy là tính đơn giản và dễ hiểu. Nó cung cấp một cái nhìn tổng quan về hiệu suất của mô hình mà không cần phải xem xét các yếu tố khác, phù hợp khi làm việc với các bộ dữ liệu cân bằng giữa các lớp dữ liệu.
Đối với những bộ dữ liệu mất cân bằng, accuracy bộc lộ điểm yếu khi không thể đánh giá khách quan được chất lượng của mô hình.

## 2. Bài toán về sản phẩm tốt và sản phẩm lỗi trong hàng hoá

Giả sử, ta cần phải đánh giá một mô hình phân loại sản phẩm tốt và sản phẩm lỗi trong hàng hoá.

**Ví dụ 1:** Ta có bộ dữ liệu đánh giá **cân bằng** gồm 100 sản phẩm, trong đó có 55 sản phẩm tốt và 45 sản phẩm lỗi.
- Trong 45 sản phẩm lỗi, mô hình dự đoán **đúng** 40 sản phẩm là lỗi và **sai** 5 sản phẩm là tốt.
- Trong 55 sản phẩm tốt, mô hình dự đoán **đúng** 50 sản phẩm là tốt và **sai** 5 sản phẩm là lỗi.
- Lúc này, ta có accuracy được tính như sau
$$ \text{accuracy} = \frac{50 + 40}{100} = 90\% $$
- Với chỉ số accuracy = 90%, ta có thể thấy rằng mô hình này có độ chính xác tương đối cao và **có thể đánh giá đây là một mô hình tốt**.

**Ví dụ 2:** Ta có bộ dữ liệu đánh giá **cân bằng** gồm 100 sản phẩm, trong đó có 55 sản phẩm tốt và 45 sản phẩm lỗi.
- Trong 45 sản phẩm lỗi, mô hình dự đoán **đúng** 5 sản phẩm là lỗi và **sai** 40 sản phẩm là tốt.
- Trong 55 sản phẩm tốt, mô hình dự đoán **đúng** 50 sản phẩm là tốt và **sai** 5 sản phẩm là lỗi.
- Lúc này, ta có accuracy được tính như sau:
$$ \text{accuracy} = \frac{50 + 5}{100} = 55\% $$
- Với chỉ số accuracy = 55%, ta có thể thấy rằng mô hình này có độ chính xác thấp và **đây là một mô hình kém**.

**Ví dụ 3:** Ta có bộ dữ liệu đánh giá **không cân bằng** gồm 100 sản phẩm, trong đó có 90 sản phẩm tốt và 10 sản phẩm lỗi.
- Trong 10 sản phẩm lỗi, mô hình dự đoán **đúng** 5 sản phẩm là lỗi và **sai** 5 sản phẩm là tốt.
- Trong 90 sản phẩm tốt, mô hình dự đoán **đúng** 85 sản phẩm là tốt và **sai** 5 sản phẩm là lỗi.
- Lúc này, ta có accuracy được tính như sau:
$$ \text{accuracy} = \frac{85 + 5}{100} = 90\% $$
- Với chỉ số accuracy = 90%, ta có thể thấy rằng mô hình này có độ chính xác tương đối cao **nhưng ta không thể đánh giá được mô hình này có tốt hay không**.

**Ví dụ 4:** Ta có bộ dữ liệu đánh giá **không cân bằng** gồm 100 sản phẩm, trong đó có 90 sản phẩm tốt và 10 sản phẩm lỗi.
- Trong 10 sản phẩm lỗi, mô hình dự đoán **đúng** 8 sản phẩm là lỗi và **sai** 2 sản phẩm là tốt.
- Trong 90 sản phẩm tốt, mô hình dự đoán **đúng** 85 sản phẩm là tốt và **sai** 5 sản phẩm là lỗi.
- Lúc này, ta có accuracy được tính như sau:
$$ \text{accuracy} = \frac{85 + 8}{100} = 93\% $$
- Với chỉ số accuracy = 93%, ta có thể thấy rằng mô hình này có độ chính xác tương đối cao và **có thể đánh giá đây là một mô hình tốt**.

Với các ví dụ trên, ta có thể thấy rằng accuracy không thể đánh giá được chính xác chất lượng của mô hình trong trường hợp bộ dữ liệu mất cân bằng giữa các lớp dữ liệu.
Đối với các bài toán phân loại mà số lượng các lớp dữ liệu không cân bằng, accuracy có thể dẫn đến những đánh giá sai lệch về chất lượng của mô hình.

Lúc này, ta cần phải sử dụng các chỉ số đánh giá khác để có thể đánh giá được chất lượng của mô hình một cách chính xác và khách quan hơn.

## 3. Confusion matrix

Trước khi đến với các metrics đánh giá giúp giải quyết vấn đề mà accuracy gặp phải, ta đến với một công cụ trực quan hoá kết quả của mô hình classification rất hữu ích, đó là confusion matrix.

Confusion matrix là công cụ giúp trực quan hoá cả lời dự đoán của mô hình và label đúng của điểm dữ liệu đó.
Confusion matrix là ma trận vuông gồm các cột là các lời dự đoán của mô hình và các hàng là các label đúng của điểm dữ liệu đó hoặc ngược lại.
Đối với bài toán binary classification, confusion matrix là một ma trận có kích thước $2 \times 2$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_definition.jpeg" style="width: 500px;"/>

Từ đó, confusion matrix tạo ra 4 giá trị: True Positive (TP), False Positive (FP), True Negative (TN) và False Negative (FN).
- TP là số luợng các điểm dữ liệu mà mô hình dự đoán **đúng là lớp positive** tương ứng với label của chúng là postive.
- TN là số luợng các điểm dữ liệu mà mô hình dự đoán **đúng là lớp negative** tương ứng với label của chúng là negative.
- FP là số luợng các điểm dữ liệu mà mô hình dự đoán **sai là lớp positive** nhưng với label của chúng là negative.
- FN là số luợng các điểm dữ liệu mà mô hình dự đoán **sai là lớp negative** nhưng với label của chúng là postive.
Với các giá trị như trên, hiển nhiên, ta luôn mong muốn hai giá trị TP và TN lớn và hai giá trị FP và FN nhỏ.

Xét các ví dụ trên, giả sử ta coi lớp positive là lớp **sản phẩm lỗi** và lớp negative là lớp **sản phẩm tốt**, ta có thể tính toán được confusion matrix như sau:

**Ví dụ 1:** Ta có TP = 40, TN = 50, FP = 5, FN = 5.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_example_1.jpeg" style="width: 500px;"/>

**Ví dụ 2:** Ta có TP = 5, TN = 50, FP = 40, FN = 5.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_example_2.jpeg" style="width: 500px;"/>

**Ví dụ 3:** Ta có TP = 5, TN = 85, FP = 5, FN = 5.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_example_3.jpeg" style="width: 500px;"/>

**Ví dụ 4:** Ta có TP = 8, TN = 85, FP = 2, FN = 5.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_example_4.jpeg" style="width: 500px;"/>

Với các giá trị TP, TN, FP và FN, ta có công thức của accuracy như sau:
$$ \text{accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_accuracy.jpeg" style="width: 500px;"/>

## 4. Recall - Precision - F score - Specificity

Từ confusion matrix, ta có thể tính toán được các chỉ số đánh giá khác nhau cho mô hình classification.

### 4.1. Recall

Recall được tính bằng việc lấy số lượng dự đoán **đúng là lớp positive** của mô hình chia cho **tổng số điểm dữ liệu thực sự là postive**.

$$
\text{recall} = \frac{\text{TP}}{\text{TP + FN}}
$$

Với chỉ số recall, ta có thể tính được trên ví dụ 3 và ví dụ 4 như sau:

**- Ví dụ 3:**
$$ \text{recall} = \frac{5}{5 + 5} = \frac{5}{10} = 50\% $$

**- Ví dụ 4:**
$$ \text{recall} = \frac{8}{8 + 2} = \frac{8}{10} = 80\% $$

Từ đây, ta có thể khẳng định rằng mô hình trong ví dụ 4 là mô hình khá tốt trong khi mô hình trong ví dụ 3 là mô hình kém.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_recall.jpeg" style="width: 500px;"/>

**Ví dụ 5:** Ta có bộ dữ liệu đánh giá **không cân bằng** gồm 100 sản phẩm, trong đó có 90 sản phẩm tốt và 10 sản phẩm lỗi.
- Trong 10 sản phẩm lỗi, mô hình dự đoán **đúng** 10 sản phẩm là lỗi và **sai** 0 sản phẩm là tốt.
- Trong 90 sản phẩm tốt, mô hình dự đoán **đúng** 20 sản phẩm là tốt và **sai** 70 sản phẩm là lỗi.
- Lúc này, ta có các giá trị TP = 10, TN = 20, FP = 70, FN = 0 và ta tính được recall như sau:
$$ \text{recall} = \frac{10}{10 + 0} = \frac{10}{10} = 100\% $$
- Với chỉ số recall = 100%, ta có thể thấy rằng mô hình này có độ chính xác tương đối cao **nhưng ta không thể đánh giá mô hình này tốt được**.

### 4.2. Precision

Precision được tính bằng việc lấy số lượng dự đoán **đúng là lớp positive** của mô hình chia cho **tổng số dự đoán positive** của mô hình.

$$
\text{precision} = \frac{\text{TP}}{\text{TP + FP}}
$$

Với chỉ số precision, ta có thể tính được trên ví dụ 5 như sau:
$$ \text{precision} = \frac{10}{10 + 70} = \frac{10}{80} = 12.5\% $$
Và với chỉ số precision = 12.5%, ta có thể thấy rằng mô hình này có độ chính xác thấp và **đây là một mô hình kém**.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_precision.jpeg" style="width: 500px;"/>

Các chỉ số đánh giá ra đời nhằm mục đích đánh giá và so sánh các mô hình machine learning với nhau, để chọn ra mô hình có độ chính xác cao nhất.
Tuy nhiên, việc quan sát đồng thời cả precision và recall thường gây ra khó khăn, liệu rằng với một mô hình có precision cao hơn nhưng recall thấp hơn và một mô hình có recall cao hơn nhưng precision thấp hơn thì mô hình nào tốt hơn?

Để giải quyết vấn đề này, ta cần một chỉ số đánh giá khác giúp kết hợp được cả precision và recall lại với nhau.

### 4.3. F1 score - F beta score

Ta có $\text{F1}$ score là chỉ số giúp kết hợp được precision và recall.

$\text{F1}$ được tính bằng sự kết hợp của cả giá trị precision và giá trị recall, do đó, $\text{F1}$ chỉ cao khi cả precision và recall đều cao, còn bất kỳ một trong hai chỉ số thấp thì $\text{F1}$ sẽ thấp.

$$
\text{F1} = \frac{2 * \text{precision} * \text{recall}}{\text{precision} + \text{recall}}
$$

Tuy nhiên, trong một số trường hợp, mặc dù ta muốn quan sát đồng thời cả precision và recall, nhưng ta lại ưu tiên precision hơn một chút hoặc ưu tiên recall hơn một chút.
Ta có thể sử dụng dạng khái quát của $\text{F1}$ là $\text{F}_\beta$.

$$
\text{F}_\beta = \frac{(1 + \beta^2) * \text{precision} * \text{recall}}{\beta^2 * \text{precision} + \text{recall}}
$$

Trong đó, $\beta$ là giá trị do ta lựa chọn nhằm cân đối giữa việc ưu tiên precision hay ưu tiên recall.
- Với trường hợp ta muốn ưu tiên precision, ta lựa chọn $0 < \beta < 1$. $\beta$ càng nhỏ, càng ưu tiên precision.
- Với trường hợp ta muốn ưu tiên recall, ta lựa chọn $1 < \beta < \infty$. $\beta$ càng lớn, càng ưu tiên recall.
- Với trường hợp cân bằng giữa precision và recall, ta chọn $\beta = 1$, ta có chỉ số $\text{F1}$.

### 4.4. Specificity

Một chỉ số tương tự như recall, nhưng hoạt động với lớp negative, đó là specificity.
Tuy nhiên, chỉ số này ít được sử dụng trong thực tế.

$$
\text{specificity} = \frac{\text{TN}}{\text{TN + FP}}
$$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_specificity.jpeg" style="width: 500px;"/>

## 5. Confusion matrix trong bài toán multi-class classification

Tương tự đối với bài toán multi-class classification, confusion matrix là một ma trận vuông có kích thước $n \times n$, trong đó $n$ là số lớp dữ liệu trong bài toán phân loại.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_3class.jpeg" style="width: 500px;"/>

Với việc xây dựng được confusion matrix, ta cũng có thể tính toán được các chỉ số đánh giá cho mô hình như accuracy, recall, precision, F1 score và specificity.

- **Accuracy**

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_3class_accuracy.jpeg" style="width: 500px;"/>

- **Recall với class A là lớp positive**

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_3class_recall_a.jpeg" style="width: 500px;"/>

- **Precision với class A là lớp positive**

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/5-classification-metrics/cm_3class_precision_a.jpeg" style="width: 500px;"/>

Trong thư viện scikit-learn, 

<!-- ## 6. Chỉ số ROC - AUC

ROC (Receiver Operating Characteristic) là một biểu đồ thể hiện mối quan hệ giữa tỷ lệ dương tính thực sự (True Positive Rate - TPR) và tỷ lệ âm tính giả (False Positive Rate - FPR) của một mô hình phân loại nhị phân.

TPR là tỷ lệ dự đoán đúng của mô hình trên các điểm dữ liệu thực sự là positive, được tính bằng công thức sau:
 -->
