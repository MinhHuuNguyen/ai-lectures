---
time:
title: Mô hình Linear Regression
description: Mô hình Linear Regression là một trong những mô hình đơn giản nhất trong các Machine Learning. Mô hình Linear Regression thường được sử dụng để dự đoán giá trị của một biến liên tục dựa trên một hoặc nhiều biến đầu vào.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/2-linear-regression/house_price.png
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Bài toán dự đoán giá nhà

### 1.1. Dựa vào diện tích

Một trong những bài toán phổ biến nhất của Linear Regression là dự đoán giá nhà dựa vào diện tích.
Trong bài toán này, chúng ta sẽ xây dựng mô hình Machine Learning dự đoán giá nhà dựa vào diện tích của căn nhà.

Để làm được điều này, ta cần thu thập một bộ dữ liệu gồm diện tích của các căn nhà và giá nhà tương ứng.

| #     | House Size | House Price |
|-------|------------|-------------|
| 1     | 81         | 405         |
| 2     | 122        | 632         |
| 3     | 44         | 251         |
| 4     | 136        | 723         |
| 5     | 101        | 510         |
| 6     | 90         | 496         |
| 7     | 50         | 322         |
| 8     | 132        | 697         |
| 9     | 112        | 575         |
| 10    | 116        | 736         |

Theo dữ liệu thực tế, giá nhà thường tăng theo diện tích của căn nhà, nghĩa là diện tích nhà càng lớn thì giá nhà càng cao và ngược lại.
Tuy điều này không hoàn toàn đúng với tất cả các điểm dữ liệu, nhưng nếu xét một cách khái quát trên toàn bộ dữ liệu, ta có thể thấy mối quan hệ này khá rõ ràng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/2-linear-regression/house_price.png" style="width: 800px;"/>

Từ đó, việc xây dựng một mô hình Linear Regression để dự đoán giá nhà dựa vào diện tích cụ thể là việc tìm một hàm số nhận đầu vào là diện tích nhà và trả về giá nhà tương ứng.
Với mỗi một giá trị diện tích nhà, mô hình sẽ tính toán ra được giá nhà tương ứng.

$$ pred\_house\_price = linear\_regression(house\_size) $$

Ví dụ:
$$ 405 = linear\_regression(81) $$
$$ 632 = linear\_regression(122) $$


### 1.2. Dựa vào nhiều tiêu chí khác nhau

Ngoài diện tích, giá nhà còn phụ thuộc vào nhiều yếu tố khác như vị trí, tiện ích, hướng nhà, năm xây dựng, ...
Lúc này, bộ dữ liệu mà chúng ta sử dụng sẽ có nhiều thông tin hơn

| #     | House Size | Location | Utility | Direction | Year Built | House Price |
|-------|------------|----------|---------|-----------|------------|-------------|
| 1     | 132        | HN       | Good    | East      | 1998       | 689         |
| 2     | 81         | TB       | Normal  | West      | 2005       | 405         |
| 3     | 122        | HCM      | Normal  | South     | 2010       | 632         |
| 4     | 44         | HP       | Good    | North     | 2000       | 251         |
| 5     | 136        | HN       | Bad     | West      | 2020       | 723         |
| 6     | 101        | HCM      | Normal  | East      | 2018       | 510         |
| 7     | 90         | ĐN       | Good    | North     | 1988       | 496         |
| 8     | 50         | TB       | Bad     | South     | 1995       | 322         |
| 9     | 132        | ĐN       | Normal  | West      | 1990       | 697         |
| 10    | 112        | HP       | Good    | East      | 1997       | 575         |


Trong trường hợp này, mô hình Linear Regression sẽ nhận nhiều đầu vào hơn, mỗi đầu vào tương ứng với một yếu tố cụ thể.
Lúc này, mô hình Linear Regression sẽ được mô tả bởi một hàm số như sau:

$$ pred\_house\_price = linear\_regression(house\_size, location, utility, direction, year\_built, ...) $$

Ví dụ:
$$ 689 = linear\_regression(132, HN, Good, East, 1998, ...) $$
$$ 405 = linear\_regression(81, TB, Normal, West, 2005, ...) $$

## 2. Kiến trúc mô hình

Mô hình Linear Regression có dạng khái quát như sau:

$$ \hat{y} = w_1x_1 + w_2x_2 + ... + w_nx_n + b $$
Trong đó:
- $\hat{y}$ là giá trị dự đoán của mô hình
- $w_1, w_2, ..., w_n$ là các trọng số tương ứng với các biến đầu vào $x_1, x_2, ..., x_n$
- $x_1, x_2, ..., x_n$ là các biến đầu vào
- $b$ là giá trị bias

Nếu áp dụng công thức trên cho bài toán dự đoán giá nhà dựa vào diện tích, ta có

$$ house\_price = w_1 \cdot house\_size + b $$

Tương tự, đối với bài toán dự đoán giá nhà dựa vào nhiều tiêu chí khác nhau, mô hình Linear Regression sẽ có dạng như sau:

$$ house\_price = w_1 \cdot house\_size + w_2 \cdot location + w_3 \cdot utility + w_4 \cdot direction + w_5 \cdot year\_built + b $$

Áp dụng công thức trên cho bộ dữ liệu mẫu, ta có thể tính toán ra giá trị Predicted Price tương ứng với mỗi điểm dữ liệu.

| #  | House Size | Location | Utility | Direction | Year Built | House Price | Predicted Price |
|----|------------|----------|---------|-----------|------------|-------------|-----------------|
| 1  | 132        | HN       | Good    | East      | 1998       | 689         | 880.2           |
| 2  | 81         | TB       | Normal  | West      | 2005       | 405         | 636.7           |
| 3  | 122        | HCM      | Normal  | South     | 2010       | 632         | 865.4           |
| 4  | 44         | HP       | Good    | North     | 2000       | 251         | 428.8           |
| 5  | 136        | HN       | Bad     | West      | 2020       | 723         | 969.2           |
| 6  | 101        | HCM      | Normal  | East      | 2018       | 510         | 781.0           |
| 7  | 90         | ĐN       | Good    | North     | 1988       | 496         | 630.8           |
| 8  | 50         | TB       | Bad     | South     | 1995       | 322         | 444.5           |
| 9  | 132        | ĐN       | Normal  | West      | 1990       | 697         | 855.4           |
| 10 | 112        | HP       | Good    | East      | 1997       | 575         | 773.1           |


## 3. Huấn luyện mô hình

Với mỗi điểm dữ liệu, áp dụng công thức Linear Regression, ta sẽ có giá trị Predicted Price tương ứng.
Tuy nhiên, ta mong muốn rằng giá trị Predicted Price càng gần với giá trị House Price thực tế càng tốt.

Từ đó, mục tiêu của quá trình huấn luyện mô hình Linear Regression là tìm ra các trọng số $w_1, w_2, ..., w_n$ và bias $b$ sao cho giá trị Predicted Price càng gần với giá trị House Price thực tế càng tốt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/2-linear-regression/house_price_pred_vs_label.png" style="width: 800px;"/>

Giống như con người, mô hình Linear Regression cũng học từ dữ liệu thông qua việc tính toán sai số giữa giá trị Predicted Price và giá trị House Price thực tế.
Về cơ bản, sai số này càng nhỏ thì mô hình càng tốt.

## 4. Loss function của mô hình Linear Regression

Ta cần một hàm số để đo đạc sự sai khác giữa giá trị dự đoán của mô hình và giá trị thực tế từ trong bộ dữ liệu.
Trong trường hợp này là đánh giá mức độ sai số giữa giá trị Predicted Price và giá trị House Price thực tế.

Hàm số này thường được gọi là Loss function, được ký hiệu là $L$.

$$ loss = L(Predicted Price, House Price) $$

### 4.1. Mean Squared Error (MSE)

Trong trường hợp của mô hình Linear Regression, Loss function thường được chọn là Mean Squared Error (MSE).

$$ loss = MSE(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 $$

Trong đó:
- $y_i$ là giá trị House Price thực tế
- $\hat{y}_i$ là giá trị Predicted Price
- $n$ là số lượng điểm dữ liệu

| #  | House Size | Location | Utility | Direction | Year Built | House Price | Predicted Price | MSE Loss |
|----|------------|----------|---------|-----------|------------|-------------|-----------------| -------- |
| 1  | 132        | HN       | Good    | East      | 1998       | 689         | 880.2           | 36557.44 |
| 2  | 81         | TB       | Normal  | West      | 2005       | 405         | 636.7           | 53684.89 |
| 3  | 122        | HCM      | Normal  | South     | 2010       | 632         | 865.4           | 54475.56 |
| 4  | 44         | HP       | Good    | North     | 2000       | 251         | 428.8           | 31612.84 |
| 5  | 136        | HN       | Bad     | West      | 2020       | 723         | 969.2           | 60614.44 |
| 6  | 101        | HCM      | Normal  | East      | 2018       | 510         | 781.0           | 73441.00 |
| 7  | 90         | ĐN       | Good    | North     | 1988       | 496         | 630.8           | 18171.04 |
| 8  | 50         | TB       | Bad     | South     | 1995       | 322         | 444.5           | 15006.25 |
| 9  | 132        | ĐN       | Normal  | West      | 1990       | 697         | 855.4           | 25090.56 |
| 10 | 112        | HP       | Good    | East      | 1997       | 575         | 773.1           | 39243.61 |

Từ bảng trên, ta có thể tính toán được giá trị loss của mô hình với toàn bộ dữ liệu bằng cách lấy trung bình của các giá trị loss của từng điểm dữ liệu.

$$ loss = 40,789.76 $$

MSE tập trung phạt mạnh những điểm dữ liệu có sai số lớn, do đó, mô hình khi được huấn luyện bằng hàm loss này sẽ thường có xu hướng có nhiều giá trị dự đoán với sai số nhỏ.

Ngoài ra, do MSE sử dụng bình phương sai số, nên nó cũng nhạy cảm với các điểm dữ liệu có giá trị dự đoán lệch lớn so với giá trị thực tế.
Điều này có thể gây ra một số vấn đề liên quan đến tràn số (overflow) nếu giá trị dự đoán và giá trị thực tế quá lớn.

### 4.2. Root Mean Squared Error (RMSE)

Để giảm thiểu vấn đề tràn số, ta có thể sử dụng Root Mean Squared Error (RMSE) thay vì MSE.

$$ loss = RMSE(\hat{y}, y) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2} $$

RMSE đơn giản là căn bậc hai của MSE, giúp giảm thiểu vấn đề tràn số và giữ nguyên tính chất phạt mạnh những điểm dữ liệu có sai số lớn của MSE.

### 4.3. Mean Absolute Error (MAE)

Ngoài MSE và RMSE, còn có một hàm loss khác cũng khá hay được sử dụng trong các bài toán Regression là Mean Absolute Error (MAE).
MAE đo lường trung bình giá trị tuyệt đối của sai số giữa giá trị dự đoán và giá trị thực tế.

$$ loss = MAE(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i| $$

Trong đó:
- $y_i$ là giá trị House Price thực tế
- $\hat{y}_i$ là giá trị Predicted Price
- $n$ là số lượng điểm dữ liệu

MAE không nhạy cảm với các điểm dữ liệu có giá trị dự đoán lệch lớn so với giá trị thực tế như MSE, do đó, mô hình khi được huấn luyện bằng hàm loss này sẽ thường có xu hướng có thể có rất ít các giá trị dự đoán lệch, nhưng nếu lệch có thể lệch lớn.

## 5. Phương pháp tối ưu mô hình

Việc tối ưu mô hình Linear Regression là việc tìm ra các trọng số $w_1, w_2, ..., w_n$ và bias $b$ sao cho giá trị loss của mô hình là nhỏ nhất.
Để làm được điều này, ta cần sử dụng một phương pháp tối ưu quen thuộc là sử dụng phép toán Đạo hàm.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/2-linear-regression/house_price_before_after.png" style="width: 800px;"/>

Để ký hiệu toán học được đơn giản, ta sẽ sử dụng biến $w$ để ký hiệu tập hợp các trọng số $w_1, w_2, ..., w_n$ và bias $b$.
Nghĩa là $w = [w_1, w_2, ..., w_n, b]$.

$$ loss = L(\hat{y}, y) = L(wx, y) = L(y_i, w_1x_1 + w_2x_2 + ... + w_nx_n + b, y) $$

Đối với hàm loss MSE, ta có công thức độ lệch trên toàn bộ bộ dữ liệu thu gọn sau:
$$ L(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 = \frac{1}{n} (\hat{y} - y)^2 $$

Từ kiến thức về hàm số đã được học từ trung học phổ thông, ta thực hiện đạo hàm hàm loss MSE theo trọng số $w$.

$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial w} = 2(\hat{y} - y) \frac{\partial \hat{y}}{\partial w} = 2(y - wx) \frac{\partial wx}{\partial w} = 2(y - wx)x^{T}$$

và giải phương trình đạo hàm bằng 0, ta sẽ tìm ra được các giá trị cực trị của hàm loss.
$$ 2(y - wx)x^{T} = 0 $$
$$ yx^{T} - wxx^{T} = 0 $$
$$ yx^{T} = wxx^{T} $$
$$ w = (xx^{T})^{-1}yx^{T} $$

Với cách làm trên, ta có thể tính toán được trọng số $w$ tối ưu của mô hình Linear Regression sao cho giá trị loss MSE là nhỏ nhất.
Tuy nhiên, cách làm như vậy có hai điểm yếu rất lớn:
- Độ phức tạp tính toán: việc tính toán ma trận nghịch đảo $(xx^{T})^{-1}$ có độ phức tạp tính toán rất lớn, đặc biệt khi số lượng điểm dữ liệu lớn.
- Độ ổn định: việc tính toán ma trận nghịch đảo $(xx^{T})^{-1}$ có thể dẫn đến vấn đề không ổn định nếu ma trận $xx^{T}$ không khả nghịch.

Để giải quyết vấn đề này, ta sử dụng một phương pháp tối ưu khác là Gradient Descent.
Bài viết tiếp theo sẽ giới thiệu về phương pháp tối ưu Gradient Descent.

## 6. Chỉ số đánh giá mô hình

Để đánh giá chất lượng của mô hình Linear Regression nói chung và các mô hình regression, bên cạnh các giá trị loss như MSE, MAE, ta còn sử dụng một số chỉ số đánh giá khác.

### 6.1. R-squared (R2 - Coefficient of determination)

R-squared là một chỉ số đánh giá mô hình phổ biến được sử dụng trong các bài toán Regression.
R-squared đo lường mức độ biến thiên của giá trị dự đoán so với giá trị thực tế.

$$
R^2 = 1 -  \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n} (\hat{y}_i - y_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

Trong đó:
- $SS_{res}$ (Sum of Squared Residuals) là tổng bình phương sai số giữa giá trị dự đoán và giá trị thực tế
- $SS_{tot}$ (Total Sum of Squares) là tổng bình phương sai số giữa giá trị thực tế và giá trị trung bình của giá trị thực tế
- $y_i$ là giá trị thực tế
- $\hat{y}_i$ là giá trị dự đoán
- $\bar{y}$ là giá trị trung bình của giá trị thực tế
- $n$ là số lượng điểm dữ liệu

R-squared có giá trị nằm trong khoảng $[0, 1]$, với giá trị càng gần 1 thì mô hình càng tốt.

### 6.2. Adjusted R-squared

Adjusted R-squared là một biến thể của R-squared, được sử dụng để đánh giá mô hình Linear Regression khi số lượng biến đầu vào lớn.
Adjusted R-squared sẽ phạt mô hình nếu số lượng biến đầu vào tăng mà không cải thiện chất lượng dự đoán.

$$
Adjusted R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - k - 1}
$$

Trong đó:
- $R^2$ là giá trị R-squared
- $n$ là số lượng điểm dữ liệu trong tập dữ liệu
- $k$ là số lượng biến đầu vào, tương ứng với số lượng trọng số $w$ trong mô hình Linear Regression

Adjusted R-squared có giá trị nằm trong khoảng $[0, 1]$, với giá trị càng gần 1 thì mô hình càng tốt.

Việc tính toán Adjusted R-squared giúp chúng ta đánh giá mô hình Linear Regression một cách chính xác hơn, đặc biệt khi số lượng biến đầu vào lớn.

Nếu giá trị $n$ và $k$ tăng nhưng giá trị Adjusted R-squared không tăng hoặc giảm, ta có thể rút ra kết luận rằng mô hình không cải thiện chất lượng dự đoán.
