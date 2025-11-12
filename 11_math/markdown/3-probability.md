---
time: 12/19/2022
title: Một số khái niệm cơ bản trong xác suất
description: Xác suất là một trong những khái niệm quan trọng nhất trong thống kê và học máy. Nó giúp chúng ta hiểu rõ hơn về cách mà các biến ngẫu nhiên tương tác với nhau và cách mà chúng ta có thể dự đoán các kết quả trong tương lai. Bài viết này sẽ giúp bạn hiểu rõ hơn về các khái niệm cơ bản trong xác suất, bao gồm biến ngẫu nhiên, không gian mẫu, biến cố, kết quả, phân phối xác suất, xác suất đồng thời, xác suất biên và xác suất điều kiện.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/banner.jpeg
tags: [math]
is_highlight: false
is_published: true
---

## 1. Biến ngẫu nhiên và các khái niệm liên quan

Một ví dụ thực tế thường được dùng để giải thích các khái niệm trong xác suất là ví dụ tung xúc xắc, ở đây ta sẽ sử dụng ví dụ này để minh hoạ các khái niệm.

**Biến ngẫu nhiên (random variable) là đại lượng dùng để đại diện cho những giá trị ngẫu nhiên không xác định.**
Trong ví dụ về tung xúc xắc, biến ngẫu nhiên là số chấm thu được ở mặt trên của xúc xắc.

**Không gian mẫu (sample space) là tập hợp tất cả các giá trị mà một biến ngẫu nhiên có thể nhận.**
Trong ví dụ về tung xúc xắc, không gian mẫu là tập hợp tất cả các khả năng mà mặt trên của xúc xắc có thể nhận, cụ thể là 1 chấm, 2 chấm, 3 chấm, 4 chấm, 5 chấm, và 6 chấm.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/3-probability/random_variable_sample_space.jpeg" style="width: 600px;"/>

**Biến cố (event) là một sự kiện xảy ra giúp ta nhận được một kết quả của biến ngẫu nhiên.**
Trong ví dụ về tung xúc xắc, biến cố là sự kiện ta tung xúc xắc.

**Kết quả (outcome) là giá trị mà biến ngẫu nhiên nhận được sau khi biến cố xảy ra.**
Trong ví dụ về tung xúc xắc, sau khi tung xúc xắc, ta thu được kết quả là 1 chấm chả hạn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/3-probability/event_outcome.jpeg" style="width: 600px;"/>

Khi ta thực hiện lặp đi lặp lại nhiều lần, ta có thể thu được nhiều kết quả khác nhau đối với cùng biến ngẫu nhiên, sẽ có những kết quả xuất hiện nhiều lần hơn các kết quả khác, sẽ có những kết quả xuất hiện ít lần hơn các lần khác, sẽ có những kết quả có số lần xuất hiện xấp xỉ nhau.

**Thông tin về tần suất xuất hiện của các kết quả đầu ra được gọi là phân phối xác suất (probability distribution) $p(x)$ của biến ngẫu nhiên.**

Một biến ngẫu nhiên bất kỳ có thể là biến ngẫu nhiên rời rạc (discrete random variable) hoặc biến ngẫu nhiên liên tục liên tục (continuous random variable).

**Biến ngẫu nhiên rời rạc là biến ngẫu nhiên mà không gian mẫu của nó là một tập hợp hữu hạn đếm được các giá trị.**
Trong ví dụ trên về việc tung xúc xắc, ta đang xét đến một biến ngẫu nhiên rời rạc với không gian mẫu là {1 chấm, 2 chấm, 3 chấm, 4 chấm, 5 chấm, và 6 chấm}.

Mỗi kết quả đầu ra trong không gian mẫu sẽ có một giá trị xác suất tương ứng, giá trị xác suất này không âm và có tổng bằng 1.

$$ \sum_{x} p(x) = 1 $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/3-probability/discrete_continuous_random_variable.jpeg" style="width: 600px;"/>

**Biến ngẫu nhiên liên tục là biến ngẫu nhiên mà không gian mẫu của nó là một khoảng giá trị cho trước, hay nói cách khác, không gian mẫu của nó là một tập hợp con của tập số thực.**
Khoảng giá trị này có thể là hữu hạn (ví dụ: thời gian làm bài thi của một học sinh, nằm trong khoảng từ 0 phút đến 180 phút) hoặc là vô hạn (ví dụ: thời gian di chuyển từ điểm A đến điểm B của một chiếc xe).

Xác suất để nhận biến ngẫu nhiên liên tục nhận đầu ra chính xác bằng một giá trị nào đó thường được coi là bằng 0.
Thay vào đó, ta sẽ tính toán xác suất để biến ngẫu nhiên nhận đầu ra nằm trong một khoảng giá trị nào đó, và được mô tả bởi hàm mật đô xác suất (probability density function).
Hàm mật độ xác suất luôn cho giá trị dương, và tích phân của nó trên toàn miền phải bằng 1.

$$ \int p(x)dx = 1 $$

Nhằm đơn giản hoá phần ký hiệu, hàm mật độ xác suất của một biến ngẫu nhiên liên tục $x$ cũng được ký hiệu là $p(x)$.

## 2. Hàm khối xác suất, Hàm mật độ xác suất, Hàm phân phối xác suất

### 2.3. Hàm khối xác suất (PMF - Probability Mass Function)

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

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf) minh họa phân phối xác suất rời rạc và phân phối xác suất liên tục.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/3-probability/discrete_continuous_distribution.jpeg" style="width: 900px;"/>

### 2.2. Hàm mật độ xác suất (PDF - Probability Density Function)

Hàm mật độ xác suất (PDF - Probability Density Function) là một hàm xác định xác suất của các biến ngẫu nhiên liên tục.
Nói cách khác, hàm mật độ xác suất giúp ta tính toán được xác suất mà một biến ngẫu nhiên liên tục nhận giá trị trong một khoảng giá trị nào đó.

Với biến ngẫu nhiên liên tục $X$, hàm mật độ xác suất $p(x)$ được định nghĩa như sau:
$$ P(a \leq X \leq b) = \int_a^b p(x) dx $$
$$ p(x) = \frac{d}{dx} P(X \leq x) $$

Tính chất:
- $0 \leq p(x) \leq 1$
- $\int_{-\infty}^{\infty} p(x) dx = 1$

### 2.1. Hàm phân phối xác suất (CDF - Cumulative Distribution Function)

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

## 3. Xác suất đồng thời, xác suất biên và xác suất điều kiện

Trong thực tế, thông thường ta không chỉ quan sát một biến ngẫu nhiên, ta thường quan sát cùng lúc hai hoặc nhiều biến ngẫu nhiên khác nhau, từ đó, xuất hiện các khái niệm mới về xác suất khi quan sát nhiều biến ngẫu nhiên.

### 3.1. Xác suất đồng thời (Joint probability)

Xét hai biến ngẫu nhiên $x$ và $y$, khi ta quan sát hai biến ngẫu nhiên này, ta sẽ thấy những cặp giá trị $x$ và $y$ xuất hiện nhiều hơn các cặp giá trị khác, hoặc những cặp giá trị $x$ và $y$ xuất hiện ít hơn các cặp giá trị khác.
Điều này được biển diễn thông qua xác suất xảy ra đồng thời của $x$ và $y$, ký hiệu là $p(x, y)$.

Ví dụ: Ta xét $x$ là biến ngẫu nhiên về học lực của học sinh, $y$ là biến ngẫu nhiên về hạnh kiểm của học sinh.
Ta có, $p(x=học lực giỏi, y=hạnh kiểm khá)$ là xác suất mà học sinh vừa nhận học lực giỏi vừa nhận hạnh kiểm khá.

Biến ngẫu nhiên $x$ và $y$ có thể gồm hai biến ngẫu nhiên rời rạc, hai biến ngẫu nhiên liên tục hoặc một biến ngẫu nhiên rời rạc một biến ngẫu nhiên liên tục.

Với hai biến ngẫu nhiên rời rạc $\sum_{x, y} p(x, y) = 1$.

Với hai biến ngẫu nhiên liên tục $\int p(x, y) dx dy = 1$.

Với $x$ là biến ngẫu nhiên rời rạc, $y$ là biến ngẫu nhiên liên tục $\sum_{x} \int p(x, y) dy = \int \left(\sum_{x} p(x, y) \right)dy = 1$.

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf) minh họa xác suất đồng thời của hai biến ngẫu nhiên rời rạc.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/3-probability/discrete_joint_distribution.jpeg" style="width: 400px;"/>

### 3.2. Xác suất biên (Marginal probability)

Từ xác suất đồng thời của nhiều biến ngẫu nhiên, ta có thể xác định được phân bố xác suất của từng biến bằng cách lấy tổng (với biến ngẫu nhiên rời rạc) hoặc tích phân (với biến ngẫu nhiên liên tục) theo tất cả các biến còn lại.
Quá trình này được gọi là marginalization, xác suất mà ta thu được từ quá trình này được gọi là marginal probability.

Với hai biến ngẫu nhiên rời rạc $p(x) = \sum_{y}p(x, y), p(y) = \sum_{x}p(x, y)$.

Với hai biến ngẫu nhiên liên tục $p(x) = \int p(x, y)dy, p(y) = \int p(x, y)dx$.

Với nhiều biến ngẫu nhiên rời rạc:

$$ p(x) = \sum_{y, z, w}p(x, y, z, w) $$
$$ p(x, y) = \sum_{z, w}p(x, y, z, w) $$

Trong một số tài liệu, để đơn giản hoá việc ký hiệu toán học, mặc dù sử dụng biến ngẫu nhiên liên tục, nhưng ta vẫn có thể sử dụng ký hiệu $\sum$ thay vì $\int$.

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf) minh họa xác suất biên và xác suất điều kiện của hai biến ngẫu nhiên.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/3-probability/marginal_conditional_distribution.jpeg" style="width: 800px;"/>

### 3.3. Xác suất điều kiện (Conditional probability)

Xác suất điều kiện là xác suất của một (hoặc nhiều) biến ngẫu nhiên với điều kiện một (hoặc nhiều) biến ngẫu nhiên khác nhận giá trị nào đó cụ thể.
Xác suất có điều kiện của biến ngẫu nhiên $x$ biết rằng biến ngẫu nhiên $y$ có giá trị $y*$ được ký hiệu là $p(x|y=y^*)$.

Ta có thể tính $p(x|y=y^*)$ thông qua xác suất đồng thời như sau:

$$ p(x|y = y^*) = \frac{p(x, y = y^*)}{\sum_{x} p(x, y = y^*)} = \frac{p(x, y = y^*)}{p(y = y^*)}  $$

trong đó:
- $p(x, y = y^*)$ là xác suất đồng thời của mỗi giá trị của biến ngẫu nhiên $x$ với giá trị của biến ngẫu nhiên $y = y^*$.
- $\sum_{x} p(x, y = y^*) = p(y = y^*)$ là xác suất biên của biến ngẫu nhiên y.

Ta có thể viết gọn công thức trên bằng việc không cần chỉ rõ giá trị $y = y^*$:

$$ p(x|y) = \frac{p(x, y)}{p(y)} $$

tương tự, ta có:

$$ p(y|x) = \frac{p(y, x)}{p(x)} $$

Từ hai công thức rút gọn trên, ta có mối quan hệ giữa xác suất điều kiện, xác suất biên và xác suất đồng thời:

$$ p(x, y) = p(x|y)p(y) = p(y|x)p(x) $$

## 4. Định lý Bayes (Bayes' theorem)

Định lý Bayes là một trong những định lý quan trọng nhất trong xác suất thống kê, được đặt tên theo nhà toán học người Anh Thomas Bayes.

Định lý Bayes cho phép ta tính xác suất điều kiện của một biến ngẫu nhiên với điều kiện một biến ngẫu nhiên khác đã biết giá trị.
Định lý Bayes được phát biểu như sau:

$$ p(y|x) = \frac{p(x|y)p(y)}{p(x)} $$

Trong đó:
- $p(y|x)$ là xác suất điều kiện của biến ngẫu nhiên $y$ với điều kiện biến ngẫu nhiên $x$ đã biết giá trị, được gọi là **posterior**.
- $p(x|y)$ là xác suất điều kiện của biến ngẫu nhiên $x$ với điều kiện biến ngẫu nhiên $y$ đã biết giá trị, được gọi là **likelihood**.
- $p(y)$ là xác suất biên của biến ngẫu nhiên $y$, được gọi là **prior**.
- $p(x)$ là xác suất biên của biến ngẫu nhiên $x$, được gọi là **evidence**.

Định lý Bayes được sử dụng rất nhiều, trong hai thuật toán nền tảng cho nhiều mô hình Machine learning là Maximum Likelihood Estimation (MLE) và Maximum A Posteriori Estimation (MAP).
Để hiểu rõ hơn về hai thuật toán này, bạn có thể tham khảo bài viết về [Thuật toán Maximum Likelihood Estimation (MLE) và Maximum A Posteriori (MAP)](/blog/maximum-likelihood-estimation-mle-va-maximum-a-posteriori-map).
