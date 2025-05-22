---
time: 01/10/2023
title: Các phân phối xác suất
description: Trong machine learning, phân phối xác suất là công cụ quan trọng để mô hình hóa dữ liệu và sự không chắc chắn. Nhiều thuật toán dựa trên giả thiết rằng dữ liệu tuân theo các phân phối xác suất nhất định. Hiểu rõ các phân phối này giúp lựa chọn mô hình và giải thuật thích hợp trong các bài toán học máy.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/2-probability/banner.png
tags: [math]
is_highlight: false
is_published: true
---

## 1. Tổng quan

Trong machine learning, phân phối xác suất là công cụ quan trọng để mô hình hóa dữ liệu và sự không chắc chắn.
Nhiều thuật toán dựa trên giả thiết rằng dữ liệu tuân theo các phân phối xác suất nhất định.
Hiểu rõ các phân phối này giúp lựa chọn mô hình và giải thuật thích hợp trong các bài toán học máy.

## 2. Định lý giới hạn trung tâm (Central Limit Theorem)

Định lý giới hạn trung tâm (Central Limit Theorem - CLT) là một trong những định lý quan trọng nhất trong thống kê và xác suất.

**Phát biểu định lý:** Nếu bạn lấy nhiều mẫu ngẫu nhiên có cùng kích thước từ một tổng thể (population) bất kỳ có trung bình $\mu$ và phương sai $\sigma^2$ hữu hạn, thì phân phối của trung bình mẫu (sample mean) sẽ tiến gần đến phân phối chuẩn (normal distribution) khi kích thước mẫu đủ lớn, bất kể phân phối gốc của tổng thể là gì.

**Ý nghĩa thực tiễn:** Giúp dùng phân phối chuẩn để ước lượng hoặc kiểm định giả thuyết cho dữ liệu không chuẩn (miễn là kích thước mẫu đủ lớn, thường là $n \geq 30$). 
Điều này rất hữu ích trong thống kê và học máy, vì nhiều phương pháp dựa trên giả định phân phối chuẩn.

## 3. Hàm khối xác suất, Hàm mật độ xác suất, Hàm phân phối xác suất

### 3.3. Hàm khối xác suất (PMF - Probability Mass Function)

Hàm khối xác suất (PMF - Probability Mass Function) là một hàm xác định xác suất của các biến ngẫu nhiên rời rạc.
Nói cách khác, hàm khối xác suất giúp ta tính toán được xác suất mà một biến ngẫu nhiên rời rạc nhận giá trị rời rạc nào đó.

Với biến ngẫu nhiên rời rạc $X$, hàm khối xác suất $p(x)$ được định nghĩa như sau:
$$ p(x_i) = P(X = x_i)$$
trong đó:
- $P(X = x_i)$ là xác suất để biến ngẫu nhiên $X$ nhận giá trị $x_i$.
- $p(x_i)$ là hàm khối xác suất (PMF) tại giá trị $x_i$.

Tính chất:
- $0 \leq p(x_i) \leq 1$
- $\sum_{i=1}^n p(x_i) = 1$

### 3.2. Hàm mật độ xác suất (PDF - Probability Density Function)

Hàm mật độ xác suất (PDF - Probability Density Function) là một hàm xác định xác suất của các biến ngẫu nhiên liên tục.
Nói cách khác, hàm mật độ xác suất giúp ta tính toán được xác suất mà một biến ngẫu nhiên liên tục nhận giá trị trong một khoảng giá trị nào đó.

Với biến ngẫu nhiên liên tục $X$, hàm mật độ xác suất $p(x)$ được định nghĩa như sau:
$$ P(a \leq X \leq b) = \int_a^b p(x) dx $$
$$ p(x) = \frac{d}{dx} P(X \leq x) $$

Tính chất:
- $0 \leq p(x) \leq 1$
- $\int_{-\infty}^{\infty} p(x) dx = 1$

### 3.1. Hàm phân phối xác suất (CDF - Cumulative Distribution Function)

Hàm phân phối xác suất (CDF - Cumulative Distribution Function) là một hàm xác định xác suất của các biến ngẫu nhiên rời rạc hoặc liên tục, CDF cho biết xác suất để biến ngẫu nhiên nhận giá trị nhỏ hơn hoặc bằng một giá trị $x$ nào đó.

Với biến ngẫu nhiên $X$, hàm phân phối xác suất $F(x)$ được định nghĩa như sau:
$$ F(x) = P(X \leq x) $$

Tính chất:
- Là hàm đơn điệu không giảm.
- $lim_{x \to -\infty} F(x) = 0$ và $lim_{x \to +\infty} F(x) = 1$

Với biến ngẫu nhiên rời rạc, CDF có thể được tính bằng cách cộng dồn các giá trị của PMF:
$$ F(x) = \sum_{x_i \leq x} p(x_i) $$

Với biến ngẫu nhiên liên tục, CDF có thể được tính bằng cách lấy tích phân của PDF:
$$ F(x) = \int_{-\infty}^{x} p(t) dt $$

#### So sánh PMF, PDF và CDF

- Hàm khối xác suất (PMF) là xác suất của các biến ngẫu nhiên rời rạc với các giá trị rời rạc cụ thể.
- Hàm mật độ xác suất (PDF) là xác suất của các biến ngẫu nhiên liên tục với các giá trị trong một khoảng nào đó.
- Hàm phân phối xác suất (CDF) là xác suất của các biến ngẫu nhiên rời rạc hoặc liên tục với các giá trị nhỏ hơn hoặc bằng một giá trị nào đó.
- CDF có thể được tính bằng cách cộng dồn PMF hoặc tích phân PDF.

## 4. Kullback-Leibler divergence (KL divergence)

KL divergence (viết đầy đủ là Kullback–Leibler divergence) là một khái niệm trong lý thuyết thông tin và xác suất, dùng để đo mức độ khác biệt giữa hai phân phối xác suất. 

KL divergence không phải là một khoảng cách thực sự (như khoảng cách Euclidean), vì nó không thỏa mãn tính đối xứng mà là một thước đo độ mất mát thông tin khi ta dùng phân phối $Q$ để xấp xỉ phân phối $P$.

$$ D_{KL}(P || Q) \neq D_{KL}(Q || P) $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/3-distribution/kl_divergence.png" style="width: 800px;"/>

Giả sử $P(x)$ và $Q(x)$ là hai phân phối xác suất trên cùng một tập biến ngẫu nhiên rời rạc $x$, KL divergence từ $P$ đến $Q$ được định nghĩa là:

$$ D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} $$

hoặc với biến ngẫu nhiên liên tục:

$$ D_{KL}(P || Q) = \int_{-\infty}^{\infty} P(x) \log \frac{P(x)}{Q(x)} dx $$

trong đó:
- $P(x)$ là phân phối thực tế
- $Q(x)$ là phân phối xấp xỉ
- $D_{KL}(P || Q)$ là KL divergence từ $P$ đến $Q$
- $D_{KL}(P || Q) \geq 0$ với $D_{KL}(P || Q) = 0$ khi và chỉ khi $P = Q$.

## 5. Phân phối chuẩn

Phân phối chuẩn (hay Normal/Gaussian distribution) là phân phối liên tục cơ bản và rất phổ biến.
Phân phối chuẩn là phân phối mô tả biến ngẫu nhiên liên tục nhận giá trị thực $x \in (-\infty, \infty)$

### 5.1. Phân phối chuẩn một biến (Univariate Normal Distribution)

Phân phối chuẩn một biến được mô tả bởi hai tham số:
- Kỳ vọng (mean), ký hiệu là $\mu$
- Phương sai (variance), ký hiệu là $\sigma^2$
- Hoặc Độ lệch chuẩn (standard deviation) $\sigma$

Phân phối chuẩn được ký hiệu là $P(x) = \text{Norm}_x [\mu, \sigma^2]$.
trong đó:
- $\mu$ thể hiện vị trí đỉnh, nơi có xác suất cao nhất
- $\sigma$ thể hiện độ rộng của phân phối.
    - $\sigma$ lớn đồng nghĩa với phân phối có đầu ra biến đổi mạnh
    - $\sigma$ nhỏ đồng nghĩa với phân phối có đầu ra ổn định.

Hàm mật độ xác suất của phân phối chuẩn một biến là:

$$ P(x) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp \left( -\frac{(x - \mu)^2}{2\sigma^2}\right) $$

#### Tính toán tham số của phân phối chuẩn một biến

Ta có một bộ dữ liệu $X = \{x_1, x_2, ..., x_N\}$ với $N$ mẫu.
Giả sử rằng $X$ tuân theo phân phối chuẩn một biến với các tham số $\mu$ và $\sigma^2$.
Ta có thể ước lượng các tham số này như sau:
- Kỳ vọng $\mu$ được ước lượng bằng:
$$ \hat{\mu} = \frac{1}{N} \sum_{i=1}^N x_i $$
- Phương sai $\sigma^2$ được ước lượng bằng:
$$ \hat{\sigma}^2 = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \hat{\mu})^2 $$
- Độ lệch chuẩn $\sigma$ được ước lượng bằng:
$$ \hat{\sigma} = \sqrt{\hat{\sigma}^2} $$

### 5.2. Phân phối chuẩn nhiều biến (Multivariate normal distribution)

Phân phối chuẩn nhiều biến là dạng tổng quát của phân phối chuẩn một biến, được sử dụng để mô tả biến ngẫu nhiên liên tục nhiều chiều.

Ta xét biến ngẫu nhiên $D$ chiều, phân phối chuẩn nhiều biến được mô tả bởi hai tham số:
- Vector kỳ vọng (mean vector) $\mu \in R^D$.
- Ma trận hiệp phương sai (covariance matrix) $\Sigma \in \mathbb{S}_{++}^D$ là một ma trận đối xứng xác định dương.

Ma trận hiệp phương sai là một ma trận vuông, trong đó:
- Các phần tử nằm trên đường chéo chính lần lượt là phương sai của từng biến.
- Các phần từ còn lại (không nằm trên đường chéo) là các hiệp phương sai của đôi một hai biến ngẫu nhiên khác nhau trong tập hợp.

Hiệp phương sai là độ đo sự biến thiên cùng nhau của hai biến ngẫu nhiên (phân biệt với phương sai - đo mức độ biến thiên của một biến).
Giá trị hiệp phương sai nằm trong khoảng từ $(-\infty, \infty)$ trong đó:
- Giá trị dương biểu thị rằng cả hai biến chuyển động theo cùng một hướng
- Giá trị âm biểu thị rằng cả hai biến chuyển động ngược chiều nhau
- Giá trị bằng không biểu thị hai biến không có tương quan với nhau.

#### Vì sao ma trận hiệp phương sai lại là ma trận đối xứng xác định dương?
Ma trận hiệp phương sai **luôn đối xứng qua đường chéo chính** và các phần từ trên đường chéo chính luôn dương nên **các trị riêng chính của chúng luôn dương** và ma trận hiệp phương sai là xác định dương.

Hàm mật độ xác suất của phân phối chuẩn nhiều biến là:

$$ P(x) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp \left(\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right) $$

#### Tính toán tham số của phân phối chuẩn nhiều biến

Ta có một bộ dữ liệu $X = \{x_1, x_2, ..., x_N\}$ với $N$ mẫu.
Trong đó, mỗi mẫu $x_i$ là một vector chiều $x_i \in R^D$.
Giả sử rằng $X$ tuân theo phân phối chuẩn nhiều biến với các tham số $\mu$ và $\Sigma$.
Ta có thể ước lượng các tham số này như sau:
- Vector kỳ vọng $\mu$ được ước lượng bằng:
$$ \hat{\mu} = \frac{1}{N} \sum_{i=1}^N x_i $$
- Ma trận hiệp phương sai $\Sigma$ được ước lượng bằng:
$$ \hat{\Sigma} = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \hat{\mu})(x_i - \hat{\mu})^T $$
- Độ lệch chuẩn $\sigma$ được ước lượng bằng:
$$ \hat{\sigma} = \sqrt{\hat{\Sigma}} $$

## 6. Phân phối đều (Uniform distribution)

Trái ngược với phân phối chuẩn, phân phối biểu diễn biến ngẫu nhiên có những giá trị có xác suất xuất hiện cao hơn các giá trị khác, phân phối đều được sử dụng để mô tả biến ngẫu nhiên liên tục có xác suất nhận các giá trị trong một khoảng xác định là như nhau.

Phân phối đều được mô tả bởi hai tham số:
- Tham số $a$ là giá trị nhỏ nhất trong khoảng
- Tham số $b$ là giá trị lớn nhất trong khoảng

Phân phối đều được ký hiệu là $P(x) = \text{Unif}_x [a, b]$.

Hàm mật độ xác suất của phân phối đều là:
$$ P(x) = \begin{cases} \frac{1}{b - a} & \text{if } a \leq x \leq b \\ 0 & \text{otherwise} \end{cases} $$

Phân phối đều thường được sử dụng khi không có thông tin thiên lệch nào, tức tất cả giá trị trong khoảng $[a,b]$ đều khả dụng như nhau.
Nói cách khác, khi ta cần tạo sự ngẫu nhiên mà không ưu tiên giá trị nào, phân phối này là lựa chọn hợp lý.

Ví dụ: Ta có thể dùng phân phối đều để khởi tạo trọng số ban đầu (random initialization) trong mạng nơ-ron

Ví dụ: Ta có thể dùng phân phối đều để sinh mẫu dữ liệu ngẫu nhiên đều trong khoảng nhất định. 

## 7. Phân phối Bernoulli (Bernoulli distribution)

Phân phối Bernoulli là phân phối xác suất rời rạc cơ bản dành cho biến ngẫu nhiên nhị phân: nó mô tả trường hợp khi đầu ra chỉ nhận một trong hai giá trị $x ∈ {0, 1}$.

Phân phối Bernoulli được mô tả bằng một tham số $p \in [0, 1]$.
Nếu biến ngẫu nhiên $X$ tuân theo phân phối Bernoulli với tham số $p$ được ký hiệu là 
$X\sim \mathrm{Bernoulli}(p)$

có nghĩa là:
- $P(X = 1) = p$
- $P(X = 0) = 1 - P(X = 1) = 1 - p$

hoặc

$$ P(X = k) = \begin{cases} 1 - p & \text{if } k = 0 \\ p & \text{if } k = 1 \end{cases} $$

Hai đẳng thức này thường được viết gọn lại trở thành hàm khối xác suất:

$$ P(X = k) = p^k (1 - p)^{1 - k} $$
với $k \in \{0, 1\}$.

Ví dụ: Tung một đồng xu có xác suất ra mặt ngửa là $p$ là một phép thử Bernoulli.

Phân phối Bernoulli được ứng dụng nhiều trong machine learning khi mô hình hóa nhãn nhị phân (nói cách khác, được sử dụng trong bài toán phân lớp nhị phân).

Ví dụ: Trong bài toán binary classification của mô hình logistic regression, xác suất dự đoán nhãn positive được hiểu là $p$, và kết quả nhãn có thể xem như một biến Bernoulli với xác suất thành công đó.
