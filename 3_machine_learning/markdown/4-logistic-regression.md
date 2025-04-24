---
time: 07/26/2022
title: Mô hình hồi quy logistic Logistic regression
description: Mô hình Linear regression là mô hình đơn giản để giải quyết bài toán regression, còn đối với bài toán classification, ta có mô hình Logistic regression. Mô hình Logistic regression có thể giải quyết bài toán phân lớp nhị phân (binary classification), bài toán phân lớp nhiều label (multi-label classification) và bài toán phân lớp nhiều lớp (multi-class classification).
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/classification_banner.webp
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Bài toán phân lớp nhị phân - Binary classification

Bài toán phân lớp là một trong những bài toán phổ biến nhất trong Machine Learning.
Dạng phổ biến của bài toán phân lớp là bài toán phân lớp nhị phân (binary classification).

Bài toán phân lớp nhị phân là bài toán mà mô hình đầu vào là một điểm dữ liệu và trả đầu ra là một giá trị, từ giá trị này, ta có thể phân loại điểm dữ liệu đó thuộc lớp nào trong hai lớp dữ liệu khác nhau.
- Một trong hai lớp dữ liệu được gọi là lớp positive (được đại diện bởi số 1).
Lớp còn lại được gọi là lớp negative (được đại diện bởi số 0).
Ta có thể quy ước một trong hai lớp dữ liệu là lớp positive và lớp còn lại là lớp negative.
- Giá trị đầu ra của mô hình sẽ là một giá trị nằm trong khoảng $[0, 1]$.
Giá trị này có thể được hiểu như xác suất mà điểm dữ liệu thuộc lớp positive.
- Ta chọn một giá trị nào đó để làm ngưỡng tự tin (confidence threshold).
Thông thường, giá trị này là 0.5.
Nếu giá trị đầu ra của mô hình lớn hơn ngưỡng này, ta sẽ phân loại điểm dữ liệu đó thuộc lớp positive, ngược lại, ta sẽ phân loại điểm dữ liệu đó thuộc lớp negative.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/dog_cat_binary_cls.jpeg" style="width: 400px;"/>

Ví dụ: Xét bài toán phân lớp nhị phân với dữ liệu ảnh và hai lớp dữ liệu là chó và mèo.
- Ta có thể quy ước lớp chó là lớp positive (label được mã hoá là số 1) và lớp mèo là lớp negative (label được mã hoá là số 0).
- Lúc này, ta sẽ xây dựng mô hình để nhận đầu vào là một hình ảnh, trả đầu ra là một giá trị nằm trong khoảng $[0, 1]$.
Giả sử, với một hình ảnh nào đó, mô hình trả đầu ra là 0.8, ta có thể hiểu rằng xác suất mà hình ảnh đó là hình ảnh của chó là 0.8 và xác suất mà hình ảnh đó là hình ảnh của mèo là 0.2.
- Giả sử, ta chọn ngưỡng tự tin là 0.5, lúc này, với giá trị đầu ra là 0.8, ta sẽ phân loại hình ảnh đó thuộc lớp chó (lớp positive).
Một giả sử khác, nếu ta chọn ngưỡng tự tin là 0.9, lúc này, với giá trị đầu ra là 0.8, ta sẽ phân loại hình ảnh đó thuộc lớp mèo (lớp negative).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/binary_cls.png" style="width: 800px;"/>

## 2. Ý tưởng chung của logistic regression

Mô hình Linear regression sử dụng phép biến đổi tuyến tính, trả đầu ra nằm trong khoảng $[- \infty, \infty]$ phù hợp cho việc giải quyết bài toán regression.

$$ \hat{y} = WX $$

Kế thừa điều này, mô hình Logistic regression biến đổi đầu ra của phép biến đổi tuyến tính sao cho có thể sử dụng được nó để giải quyết bài toán classification.
Cụ thể, Logistic regression áp dụng một hàm số $f$ lên đầu ra của phép biến đổi tuyến tính để thu được đầu ra như mong muốn.
Các hàm số $f$ này được gọi là các logistic activation function.

$$ \hat{y} = f(WX) $$

Đối với từng bài toán phân lớp khác nhau, ta sẽ có các hàm số $f$ khác nhau.
Tuy nhiên, nhìn chung, các hàm số này đều có một số đặc điểm chung như sau:
- Là các hàm phi tuyến (non-linear function).
Hàm số phi tuyến là hàm số mà đồ thị của nó không phải là một đường thẳng.
- Nhận đầu vào là một giá trị nằm trong khoảng $[- \infty, \infty]$ hoặc một vector chứa các giá trị nằm trong khoảng $[- \infty, \infty]$ (kết quả của phép biến đổi tuyến tính).
- Trả đầu ra nằm trong khoảng $[0, 1]$ hoặc $[-1, 1]$ tuỳ thuộc vào hàm số $f$.

## 3. Hàm kích hoạt Sigmoid

### 3.1. Công thức của hàm Sigmoid

Hàm Sigmoid là một làm số phi tuyến nhận đầu vào là bất kỳ giá trị nào trong khoảng $[- \infty, \infty]$ và trả đầu ra nằm trong khoảng trong khoảng $[0, 1]$.
Từ đó, hàm Sigmoid là một logistic activation function phù hợp để giải quyết bài toán phân lớp nhị phân (binary classification).

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Giá trị đầu ra của hàm Sigmoid có thể được hiểu như giá trị xác suất mà điểm dữ liệu thuộc lớp positive (được đại diện bởi số 1).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/sigmoid.png" style="width: 600px;"/>

Ưu điểm của hàm Sigmoid là nó có đạo hàm dễ tính toán.
Điều này giúp ta có thể tối ưu mô hình logistic regression bằng các thuật toán tối ưu như gradient descent.

Nhược điểm của hàm Sigmoid là hàm có gradient rất nhỏ khi đầu vào là các giá trị rất lớn lớn hoặc rất nhỏ.
Điều này dẫn đến hiện tượng vanishing gradient, làm cho việc tối ưu mô hình trở nên khó khăn hơn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/binary_cls_with_sigmoid.png" style="width: 1000px;"/>

### 3.2. Hàm loss Binary Cross Entropy

Giả sử, ta sử dụng bộ dữ liệu gồm có m phần tử $X = [x^1, x^2, \dots, x^i, \dots, x^m]$.
Xét điểm dữ liệu $x^i$, ta có $\sigma(W x^i)$ là xác suất mà mô hình dự đoán điểm dữ liệu $x^i$ thuộc lớp số 1 và $1 - \sigma(W x^i)$ là xác suất mà điểm dữ liệu $x^i$ thuộc lớp số 0.

$$ P(y^i = 1 | x^i, W) = \sigma(W x^i) = \hat{y}^i $$
$$ P(y^i = 0 | x^i, W) = 1 - \sigma(W x^i) = 1 - \hat{y}^i $$

Kết hợp hai công thức trên, ta thu được:

$$ P(y^i| x^i, W) = (\hat{y}^i)^{y^i} (1 - \hat{y}^i)^{(1 - y^i)} $$

Tính toán trên toàn bộ bộ dữ liệu, ta có:
$$ P(y|X, W) = \prod_{i=1}^m P(y^i| x^i, W) = \prod_{i=1}^m (\hat{y}^i)^{y^i} (1 - \hat{y}^i)^{(1 - y^i)} $$
$$ P(y|X, W) = (\hat{y})^{y} (1 - \hat{y})^{(1 - y)} $$

Đến đây, mô hình cần học ra được giá trị $W$ sao cho giá trị xác suất $P(y|X, W)$ là lớn nhất và tương đương, giá trị xác suất $- P(y|X, W)$ là nhỏ nhất.

$$ {W}^{*} = \arg\max_{W} P(y|X, W) = \arg\min_{W} - P(y|X, W) $$

Với các giá trị xác suất đều nhỏ hơn 1, nên việc sử dụng phép nhân rất nhiều các giá trị nhỏ sẽ gây ra hiện tượng sai số trong máy tính.
Cụ thể, sau rất nhiều phép nhân các giá trị nhỏ, giá trị $P(y|X, W)$ sẽ rất nhỏ và xấp xỉ bằng 0, máy tính sẽ tự động làm tròn giá trị này về 0, dẫn đến việc không thể tối ưu được mô hình.

Ta có một giải pháp là sử dụng hàm logarit để biến đổi từ phép nhân thành phép cộng.
Hàm logarit là một hàm số đồng biến, do đó, việc tối ưu mô hình với hàm logarit không khác gì so với việc tối ưu mô hình với hàm xác suất.

Do đó, việc cực tiểu hoá giá trị $- P(y|X, W)$ tương đương với việc cực tiểu hoá $- \log{P(y|X, W)}$.

$$ {W}^{*} = \arg\min_{W} - \log{P(y|X, W)} $$

Từ đó, ta xây dựng được hàm loss cho mô hình logistic regression.
Hàm loss này được gọi là hàm loss **Binary Cross Entropy**.

$$ \mathcal{L}(W) = - \log{P(y|X, W)} $$
$$ \mathcal{L}(W) = - \log{(\hat{y}^{y} (1 - \hat{y})^{(1 - y)})} $$
$$ \mathcal{L}(W) = - (\log{\hat{y}^{y}} + \log{(1 - \hat{y})^{(1 - y)}}) $$
$$ \mathcal{L}(W) = - (y \log \hat{y} + (1 - y) \log(1 - \hat{y})) $$
$$ \mathcal{L}(W) = - \sum_i^m (y^i \log \hat{y}^i + (1 - y^i) \log(1 - \hat{y}^i)) $$

trong đó:
- $\mathcal{L}(W)$ là giá trị loss của mô hình logistic regression với trọng số $W$.
- $W$ là trọng số của mô hình logistic regression.
- $X$ là đầu vào của mô hình logistic regression.
- $y^i$ là nhãn của điểm dữ liệu $x^i$.
- $\hat{y}^i$ là giá trị đầu ra của mô hình với đầu vào là điểm dữ liệu $x^i$.
- $m$ là số lượng điểm dữ liệu trong bộ dữ liệu.

### 3.3. Tối ưu mô hình với hàm Sigmoid và hàm loss Binary Cross Entropy

Xét bộ dữ liệu đầu vào $X = [x^1, x^2, \dots, x^i, \dots, x^m]$.

$$ z = WX $$
$$ \hat{y} = \sigma(z) $$
$$ loss = \mathcal{L}(\hat{y})  $$

Để cực tiểu hoá giá trị hàm loss, ta sử dụng thuật toán Gradient Descent.

$$ \frac{\partial \mathcal{L}}{\partial {W}}  = \frac{\partial \mathcal{L}}{\partial {\hat{y}}} \cdot \frac{\partial \hat{y}}{\partial {z}} \cdot \frac{\partial z}{\partial {W}} $$

trong đó:
- $\frac{\partial \mathcal{L}}{\partial {W}}$ là đạo hàm của hàm loss theo trọng số $W$.
- $\frac{\partial \mathcal{L}}{\partial {\hat{y}}}$ là đạo hàm của hàm loss theo đầu ra của mô hình $\hat{y}$, chính là đạo hàm của hàm binary cross entropy.
- $\frac{\partial \hat{y}}{\partial {z}}$ là đạo hàm của đầu ra của mô hình theo đầu vào $z$, chính là đạo hàm của hàm Sigmoid.
- $\frac{\partial z}{\partial {W}}$ là đạo hàm của đầu vào $z$ theo trọng số $W$, chính là đạo hàm của phép biến đổi tuyến tính.

Đạo hàm của hàm loss binary cross entropy được tính như sau

$$ \frac{\partial \mathcal{L}}{\partial {\hat{y}}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} $$

Đạo hàm của hàm Sigmoid được tính như sau

$$ \frac{\partial \sigma}{\partial {z}} = \hat{y}(1 - \hat{y}) $$

Đạo hàm của hàm biến đổi tuyến tính được tính như sau

$$ \frac{\partial z}{\partial {W}} = X $$

Từ đó, ta có đạo hàm của hàm loss theo trọng số $W$ như sau

$$ \frac{\partial \mathcal{L}}{\partial {W}} = (\hat{y} - y)X $$

Đến đây, ta có thể áp dụng thuật toán gradient descent để tối ưu $W$ như bình thường.

## 4. Bài toán phân lớp nhiều label - Multi-label classification

Nếu như mô hình Logistic regression nói trên giúp giải quyết bài toán phân lớp nhị phân.
Hay trong ví dụ cụ thể về bài toán phân lớp ảnh chó hay mèo, mô hình logistic regression nói trên giúp ta trả lời câu hỏi "Trong ảnh có hình ảnh của chó hay không có hình ảnh của chó (nghĩa là có hình ảnh của mèo)???".

Tuy nhiên, trong một số trường hợp, với dữ liệu đầu vào có nhiều thông tin hơn, hay nói cách khác, ***dữ liệu đầu vào có thể có nhiều label***, ta phải giải quyết bài toán phân lớp nhiều label (multi-label classification).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/multi_label_cls.png" style="width: 800px;"/>

Lúc này mô hình logistic regression cần trả lời giúp ta câu hỏi "Trong ảnh có hình ảnh của chó hay không có hình ảnh của chó?, có hình ảnh của mèo hay không có hình ảnh của mèo? có hình ảnh của gà hay không có hình ảnh của gà? ..."
Do đó, cùng lúc, mô hình logistic regression cần giải quyết nhiều bài toán binary classification.

Thay vì phép biến đổi tuyến tính $WX$ cho đầu ra là một giá trị và ta dùng giá trị đó làm đầu vào cho hàm Sigmoid, trong trường hợp này, mô hình logistic regression thực hiện phép biến đổi tuyến tính $WX$ cho đầu ra là một vector và ta áp dụng hàm Sigmoid lên từng phần tử của vector.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/multi_label_cls_with_sigmoid.png" style="width: 1000px;"/>

Đến đây, ta thu được một dãy các giá trị xác suất mà mỗi giá trị lần lượt tương ứng với lời dự đoán của mô hình logistic regression trên từng câu hỏi "có hình ảnh của chó hay không có hình ảnh của chó?, có hình ảnh của mèo hay không có hình ảnh của mèo? có hình ảnh của gà hay không có hình ảnh của gà? ..."

Việc tính giá trị loss của hàm binary cross entropy trong bài toán multi-label classification không quá khác so với trong bài toán binary classification. Sau khi tính loss với từng lời dự đoán của mô hình trên từng câu hỏi, thông thường, ta sẽ lấy tổng hoặc lấy trung bình các giá trị loss này để thu được giá trị loss cuối cùng.

Cụ thể, với bài toán multi-label classification gồm K lớp, ta tính toán giá trị loss cuối cùng bằng cách lấy trung bình các giá trị loss của từng lớp.

$$ \mathcal{L}(\hat{y}) = \frac{1}{k} \sum_{k}^{K} \mathcal{L}(\hat{y}_k) = \frac{1}{k} \sum_{k}^{K} \mathcal{L}(\sigma (z_k)) = \frac{1}{k} \sum_{k}^{K} \mathcal{L}(\sigma (W_kX))$$

Từ đó, việc tính đạo hàm cũng sẽ khác một chút

$$ \frac{\partial \mathcal{L}}{\partial {W}} = \frac{1}{K} \sum_{k}^{K} \frac{\partial \mathcal{L}}{\partial {\hat{y}_k}} \cdot \frac{\partial {\hat{y}_k}}{\partial {z_k}} \cdot \frac{\partial z_k}{\partial {W_k}} $$

trong đó:
- $\frac{\partial \mathcal{L}}{\partial {W}}$ là đạo hàm của hàm loss theo trọng số $W$.
- $\frac{\partial \mathcal{L}}{\partial {\hat{y}_k}}$ là đạo hàm của hàm loss theo đầu ra của mô hình $\hat{y}_k$, chính là đạo hàm của hàm binary cross entropy.
- $\frac{\partial \hat{y}_k}{\partial {z_k}}$ là đạo hàm của đầu ra của mô hình theo đầu vào $z_k$, chính là đạo hàm của hàm Sigmoid.
- $\frac{\partial z_k}{\partial {W_k}}$ là đạo hàm của đầu vào $z_k$ theo trọng số $W_k$, chính là đạo hàm của phép biến đổi tuyến tính.

Đến đây, ta vẫn áp dụng thuật toán gradient descent để tối ưu $W$ như bình thường.

## 5. Bài toán phân lớp nhiều lớp - Multi-class classification

Có một trường hợp giống với multi-label classification là ta làm việc với nhiều lớp dữ liệu khác nhau, nhưng khác so với bài toán multi-label classification là mỗi ***dữ liệu đầu vào chỉ có thể có một label***, ta gọi bài toán này là bài toán multi-class classification.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/multi_class_cls.png" style="width: 800px;"/>

Trong bài toán này, mô hình logistic regression cần trả lời giúp ta câu hỏi "Trong ảnh có hình ảnh của chó hay của mèo hay của gà?".
Lúc này, phép biến đổi tuyến tính $WX$ cho đầu ra là một vector và ta sử dụng hàm kích hoạt nhận đầu vào là cả vector này và trả đầu ra là một vector mới với các giá trị nằm trong khoảng $[0, 1]$ và tổng các giá trị trong vector này bằng 1.
Ta có thể hiểu các giá trị trong vector này là xác suất mà điểm dữ liệu thuộc lớp số 1, lớp số 2, lớp số 3, ... tương ứng với các lớp dữ liệu khác nhau.

Thông thường, ta sẽ lựa chọn lớp nào có xác suất lớn nhất trong vector này làm lớp mà mô hình dự đoán cho điểm dữ liệu đầu vào.
Tuy nhiên, trong một số trường hợp, ta có thể lựa chọn lớp nào có xác suất lớn hơn ngưỡng tự tin (confidence threshold) mà ta đã quy ước trước đó.

Ví dụ: Xét bài toán phân lớp ảnh với 3 lớp dữ liệu là chó, mèo và gà.
- Ta có thể quy ước lớp chó là lớp số 1 (label được mã hoá là số 0), lớp mèo là lớp số 2 (label được mã hoá là số 1) và lớp gà là lớp số 3 (label được mã hoá là số 2).
- Lúc này, ta sẽ xây dựng mô hình để nhận đầu vào là một hình ảnh, trả đầu ra là một vector gồm 3 giá trị nằm trong khoảng $[0, 1]$.
Giả sử, với một hình ảnh nào đó, mô hình trả đầu ra là [0.8, 0.1, 0.1], ta có thể hiểu rằng xác suất mà hình ảnh đó là hình ảnh của chó là 0.8 và xác suất mà hình ảnh đó là hình ảnh của mèo là 0.1 và xác suất mà hình ảnh đó là hình ảnh của gà là 0.1.
Tương tự, với một hình ảnh khác, mô hình trả đầu ra là [0.2, 0.7, 0.1], ta có thể hiểu rằng xác suất mà hình ảnh đó là hình ảnh của chó là 0.2 và xác suất mà hình ảnh đó là hình ảnh của mèo là 0.7 và xác suất mà hình ảnh đó là hình ảnh của gà là 0.1.
- Thông thường, với giá trị đầu ra là [0.8, 0.1, 0.1], ta sẽ phân loại hình ảnh đó thuộc lớp chó (lớp số 1).
Với giá trị đầu ra là [0.2, 0.7, 0.1], ta sẽ phân loại hình ảnh đó thuộc lớp mèo (lớp số 2).

## 6. Hàm kích hoạt Softmax

Mô hình logistic regression giải quyết bài toán multi-class classification có thể được gọi với tên gọi khác là Softmax Regression vì ta sẽ sử dụng hàm Softmax thay thế cho hàm Sigmoid ở vị trí của một logistic activation function.

### 6.1. Công thức của hàm Softmax

Hàm Softmax là một làm số phi tuyến nhận đầu vào là một vector gồm bất kỳ giá trị nào trong khoảng $[- \infty, \infty]$ và trả đầu ra là một vector mới gồm các giá trị nằm trong khoảng $[0, 1]$ và tổng các giá trị trong vector này bằng 1.
Từ đó, hàm Softmax là một logistic activation function phù hợp để giải quyết bài toán phân lớp nhiều lớp (multi-class classification).

$$ \hat{y} = Softmax(z) = \left[\frac{e^{z_1}}{\sum_{i=1}^K e^{z_i}}, \frac{e^{z_2}}{\sum_{i=1}^K e^{z_i}}, \dots, \frac{e^{z_K}}{\sum_{i=1}^K e^{z_i}}\right] $$

Từng giá trị trên vector đầu ra của hàm Softmax có thể được hiểu như giá trị xác suất mà điểm dữ liệu thuộc lớp số 1, lớp số 2, lớp số 3, ... tương ứng với các lớp dữ liệu khác nhau.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/softmax.png" style="width: 600px;"/>

Ưu điểm của hàm Softmax là nó có đạo hàm dễ tính toán.
Điều này giúp ta có thể tối ưu mô hình logistic regression bằng các thuật toán tối ưu như gradient descent.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/multi_class_cls_with_softmax.png" style="width: 1000px;"/>

### 6.2. Hàm loss Categorical Cross Entropy

Với việc thay đổi logistic activation function, ta cần một hàm loss khác để tính toán giá trị loss trên tất cả các lớp trong bộ dữ liệu.
Hàm loss này được gọi là hàm loss **Categorical Cross Entropy** và ta cũng có thể hiểu, Categorical Cross Entropy là một phiên bản khái quát hơn của Binary cross entropy.

Giả sử, ta sử dụng bộ dữ liệu gồm có m phần tử $X = [x^1, x^2, \dots, x^i, \dots, x^m]$ và K lớp trong bộ dữ liệu.
Xét điểm dữ liệu $x^i$, với hàm Softmax, ta thu được vector gồm các giá trị xác suất mà điểm dữ liệu $x^i$ thuộc lớp số 1, lớp số 2, lớp số 3, ... tương ứng với các lớp dữ liệu khác nhau.

$$ P(y^i = k | x^i, W) = Softmax(W_kX) = \hat{y}^i_k $$

$$ P(y^i | x^i, W) = \prod_{k=1}^K P(y^i = k | x^i, W) = \prod_{k=1}^K \hat{y}^i_k $$

Tính toán trên toàn bộ bộ dữ liệu, ta có:

$$ P(y|X, W) = \prod_{i=1}^m P(y^i | x^i, W) = \prod_{i=1}^m \prod_{k=1}^K \hat{y}^i_k $$
$$ P(y|X, W) = \prod_{i=1}^m \prod_{k=1}^K (\hat{y}^i_k)^{y^i_k} $$

Tương tự, ta đi tìm giá trị $W$ sao cho giá trị xác suất $P(y|X, W)$ là lớn nhất và tương đương, giá trị xác suất $- P(y|X, W)$ là nhỏ nhất.

$$ W^{*} = \arg\max_{W} P(y|X, W) = \arg\min_{W} - P(y|X, W) $$

Cũng tương tự, ta sử dụng hàm logarit để biến đổi từ phép nhân thành phép cộng.
Từ đó, ta thu được hàm loss Categorical cross entropy.
$$ \mathcal{L}(W) = - \log{P(y|X, W)} $$
$$ \mathcal{L}(W) = - \log{(\prod_{i=1}^m \prod_{k=1}^K (\hat{y}^i_k)^{y^i_k})} $$
$$ \mathcal{L}(W) = - \sum_{i=1}^m \sum_{k=1}^K y^i_k \log{\hat{y}^i_k} $$

trong đó:
- $\mathcal{L}(W)$ là giá trị loss của mô hình logistic regression với trọng số $W$.
- $W$ là trọng số của mô hình logistic regression.
- $X$ là đầu vào của mô hình logistic regression.
- $y^i_k$ là nhãn của điểm dữ liệu $x^i$ với lớp số k.
- $\hat{y}^i_k$ là giá trị đầu ra của mô hình với đầu vào là điểm dữ liệu $x^i$ và lớp số k.
- $m$ là số lượng điểm dữ liệu trong bộ dữ liệu.
- $K$ là số lượng lớp trong bộ dữ liệu.

So sánh với hàm loss binary cross entropy:

$$ \mathcal{L}(W) = - \sum_{i=1}^m (y^i \log \hat{y}^i + (1 - y^i) \log(1 - \hat{y}^i)) $$

ta thấy rằng, hàm loss binary cross entropy là một trường hợp đặc biệt của hàm loss categorical cross entropy với K = 2.
Với K = 2, ta có thể hiểu rằng, hàm loss binary cross entropy là hàm loss categorical cross entropy với 2 lớp dữ liệu là lớp số 1 (lớp positive) và lớp số 2 (lớp negative).

### 6.3. Tối ưu mô hình với hàm Softmax và hàm loss Categorical Cross Entropy

Với việc ta thấy rằng hàm loss categorical cross entropy là một trường hợp đặc biệt của hàm loss binary cross entropy với K = 2, ta có thể áp dụng các công thức tính toán tương tự như trong bài toán phân lớp nhị phân.

Áp dụng chain rule, ta có
$$ \frac{\partial \mathcal{L}}{\partial {W}}  = \frac{\partial \mathcal{L}}{\partial {\hat{y}}} \cdot \frac{\partial \hat{y}}{\partial {z}} \cdot \frac{\partial z}{\partial {W}} $$

trong đó:
- $\frac{\partial \mathcal{L}}{\partial {W}}$ là đạo hàm của hàm loss theo trọng số $W$.
- $\frac{\partial \mathcal{L}}{\partial {\hat{y}}}$ là đạo hàm của hàm loss theo đầu ra của mô hình $\hat{y}$, chính là đạo hàm của hàm categorical cross entropy.
- $\frac{\partial \hat{y}}{\partial {z}}$ là đạo hàm của đầu ra của mô hình theo đầu vào $z$, chính là đạo hàm của hàm Softmax.
- $\frac{\partial z}{\partial {W}}$ là đạo hàm của đầu vào $z$ theo trọng số $W$, chính là đạo hàm của phép biến đổi tuyến tính.

Đạo hàm của hàm loss categorical cross entropy được tính như sau

$$ \frac{\partial \mathcal{L}}{\partial {\hat{y}}} = - \frac{y}{\hat{y}} $$

Đạo hàm của hàm Softmax được tính với từng phần tử như sau

$$ \frac{\partial \hat{y}_k}{\partial {z_j}} = \hat{y}_k \cdot (\delta_{kj} - \hat{y}_j) $$

trong đó:
- $\delta_{kj}$ là hàm delta Kronecker, với $k = j$, $\delta_{kj} = 1$ và ngược lại, $\delta_{kj} = 0$.
- $k$ là chỉ số của lớp dữ liệu.
- $j$ là chỉ số của lớp dữ liệu.
- $\hat{y}_k$ là giá trị đầu ra của mô hình với đầu vào là điểm dữ liệu $x^i$ và lớp số k.
- $\hat{y}_j$ là giá trị đầu ra của mô hình với đầu vào là điểm dữ liệu $x^i$ và lớp số j.

### 6.4. Sử dụng hàm Softmax trong bài toán phân lớp nhị phân

Trong bài toán phân lớp nhị phân, ta vẫn có thể sử dụng hàm Softmax thay cho hàm Sigmoid.

Từ đó, bài toán phân lớp nhị phân sẽ trở thành bài toán phân lớp nhiều lớp với 2 lớp dữ liệu là lớp số 1 (lớp positive) và lớp số 2 (lớp negative).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/binary_cls_with_softmax.png" style="width: 1000px;"/>

## 7. Hàm kích hoạt Tanh

Hàm Tanh nhận đầu vào là bất kỳ giá trị nào trong khoảng $[- \infty, \infty]$ nhưng khác với Sigmoid, Tanh trả đầu ra nằm trong khoảng trong khoảng $[-1, 1]$.

$$ \text{Tanh}(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-logistic-regression/tanh.png" style="width: 600px;"/>

Ta có thể dễ dàng biến đổi khoảng giá trị đầu ra của hàm Tanh từ $[-1, 1]$ về giống như Sigmoid $[0, 1]$ thông qua công thức

$$ \text{Tanh}(z) = 2\sigma(2z) - 1 $$
