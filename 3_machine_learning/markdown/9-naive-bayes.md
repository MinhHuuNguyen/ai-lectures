---
time: 08/09/2022
title: Mô hình Naive Bayes Classification
description: Naive Bayes là một mô hình machine learning điển hình, đại diện cho các mô hình dựa vào xác suất thống kê. Mô hình này được sử dụng rộng rãi trong các bài toán phân loại, đặc biệt là phân loại văn bản
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/9-naive-bayes/banner.jpeg
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Tổng quan

Naive Bayes là một trong số những mô hình machine learning dựa nhiều vào kiến thức toán, cụ thể là các kiến thức về xác suất thống kê.

Nếu như đối với các mô hình phân lớp như Logistic Regression, giá trị output của mô hình **ta có thể hiểu như là giá trị xác suất**, thì đối với mô hình Naive Bayes, giá trị output của mô hình **là xác suất thật sự** vì ta sử dụng các công thức xác suất để tính toán.

Mô hình này được sử dụng rộng rãi trong các bài toán phân loại, đặc biệt là phân loại văn bản.

## 2. Một số kiến thức trong xác suất thống kê

Tham khảo một số khái niệm về xác suất thống kê trong bài viết [này](/blog/mot-so-khai-niem-co-ban-trong-xac-suat).

Mô hình Naive Bayes Classification là một mô hình phân loại dựa vào định lý Bayes.
Định lý Bayes được phát biểu như sau:

$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$

trong đó:
- $P(A|B)$ là xác suất điều kiện (conditional probability), xác xuất xảy ra của sự kiện A khi biết rằng sự kiện B đã xảy ra.
- $P(B|A)$ là xác suất điều kiện (conditional probability), xác suất xảy ra của sự kiện B khi biết rằng sự kiện A đã xảy ra.
- $P(A)$ là xác suất biên (marginal probability), xác suất xảy ra của sự kiện A.
- $P(B)$ là xác suất biên (marginal probability), xác suất xảy ra của sự kiện B.

## 3. Mô hình Naive Bayes Classification

Từ công thức Bayes ở trên, ta có thể viết lại với các ký hiệu quen thuộc trong machine learning và đối chiếu với các khái niệm trong mô hình classification như sau:

$$ P(y|x) = \frac{P(x|y)P(y)}{P(x)} $$
trong đó:
- $x$ là dữ liệu đầu vào mà mô hình cần phân lớp.
- $y$ là một trong số các lớp mà mô hình cần phân điểm dữ liệu đầu vào đó vào.
- $P(y|x)$ là xác suất mà lớp $y$ xảy ra với điều kiện dữ liệu đầu vào $x$.
- $P(x|y)$ là phân phối xác suất của dữ liệu đầu vào $x$ thuộc về lớp $y$.
- $P(y)$ là xác suất mà lớp $y$ xảy ra.
- $P(x)$ là phân phối xác suất của dữ liệu đầu vào $x$ nói chung.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/9-naive-bayes/spam_vs_not_spam.jpeg" style="width: 600px;"/>

Ví dụ: Xét bài toán phân lớp các email thành 2 lớp: spam và không spam. Từ bộ dữ liệu train, ta có thể hiểu các khái niệm trên như sau:
- $x$ là một email bất kỳ mà mô hình cần phân lớp.
- $y$ là một trong 2 lớp mà mô hình cần phân lớp email đó vào: spam hoặc không spam.
    - Xét $y$ là **lớp spam**:
        - $P(y|x)$ là xác suất mà email $x$ là **lớp spam**.
        - $P(x|y)$ là phân phối xác suất của dữ liệu đầu vào $x$ thuộc về **lớp spam**.
        - $P(y)$ là xác suất mà **lớp spam** xảy ra.
        - $P(x)$ là phân phối xác suất của dữ liệu đầu vào $x$ nói chung.
    - Xét $y$ là **lớp không spam**:
        - $P(y|x)$ là xác suất mà email $x$ là **lớp không spam**.
        - $P(x|y)$ là phân phối xác suất của dữ liệu đầu vào $x$ thuộc về **lớp không spam**.
        - $P(y)$ là xác suất mà **lớp không spam** xảy ra.
        - $P(x)$ là phân phối xác suất của dữ liệu đầu vào $x$ nói chung.

Mục tiêu của mô hình Naive Bayes là sử dụng các công thức xác suất để tính toán các xác suất điều kiện $P(y|x)$ cho tất cả các lớp $y$ mà mô hình cần phân lớp dữ liệu đầu vào $x$ vào.
Sau đó, mô hình sẽ chọn lớp có xác suất điều kiện lớn nhất để phân lớp dữ liệu đầu vào $x$.

$$ y_{pred} = \arg\max_{y} P(y|x) $$

Từ ví dụ trên, ta thấy, việc tìm giá trị lớn nhất trong các giá trị xác suất điều kiện $P(y=spam|x)$ hay $P(y=not\ spam|x)$, thực chất, không phụ thuộc vào $P(x)$, vì $P(x)$ là một hằng số không thay đổi trong bài toán này.
Do đó, ta có thể bỏ qua $P(x)$ trong công thức Bayes ở trên.

Từ đó, ta có thể viết lại công thức Bayes như sau:

$$ P(y|x) \propto P(x|y)P(y) $$

trong đó, "$\propto$" có nghĩa là "tỉ lệ thuận với".

Từ công thức trên, ta có thể thấy rằng, để tính toán xác suất điều kiện $P(y|x)$, mô hình cần 2 thành phần:
- $P(x|y)$: phân phối xác suất của dữ liệu đầu vào $x$ thuộc về lớp $y$.
- $P(y)$: xác suất mà lớp $y$ xảy ra.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/9-naive-bayes/calculate_py.jpeg" style="width: 600px;"/>

Đối với thành phần $P(y)$, ta có thể tính toán được khá dễ dàng từ bộ dữ liệu train.
Ta tính toán xác suất mà lớp $y$ xảy ra bằng cách đếm số lượng các mẫu dữ liệu thuộc về lớp $y$ trong bộ dữ liệu train, chia cho tổng số lượng mẫu dữ liệu trong bộ dữ liệu train.

$$ P(y) = \frac{N(y)}{N} $$
trong đó:
- $N(y)$ là số lượng mẫu dữ liệu thuộc về lớp $y$ trong bộ dữ liệu train.
- $N$ là tổng số lượng mẫu dữ liệu trong bộ dữ liệu train.

Ví dụ: Trong bộ dữ liệu train có 1000 mẫu dữ liệu, trong đó có 300 mẫu thuộc về lớp spam và 700 mẫu thuộc về lớp không spam.
- $P(y=spam) = \frac{300}{1000} = 0.3$
- $P(y=not\ spam) = \frac{700}{1000} = 0.7$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/9-naive-bayes/calculate_pxy.jpeg" style="width: 600px;"/>

Đối với thành phần $P(x|y)$, giá trị này khó tính toán hơn vì $x$ là một vector có thể có nhiều chiều (hay dữ liệu đầu vào có nhiều đặc trưng).

Giả sử, dữ liệu đầu vào $x$ có $n$ đặc trưng, $x = (x_1, x_2, \ldots, x_n)$.
Lúc này, để tính toán $P(x|y)$, ta có công thức như sau:

$$ P(x|y) = P(x_1, x_2, \ldots, x_n|y) $$

Tuy nhiên, công thức trên rất khó tính toán vì $x_1, x_2, \ldots, x_n$ có thể có mối quan hệ với nhau.

Để đơn giản hoá công thức trên, mô hình Naive Bayes giả định rằng các đặc trưng trong dữ liệu đầu vào $x$ là độc lập với nhau (independent).
Điều này có nghĩa là, mô hình giả định rằng, sự xuất hiện của một đặc trưng trong dữ liệu đầu vào $x$ không ảnh hưởng đến sự xuất hiện của các đặc trưng khác trong dữ liệu đầu vào $x$.
Điều này có thể không đúng trong thực tế, nhưng mô hình Naive Bayes vẫn cho kết quả khá tốt trong nhiều bài toán phân loại.

Do đó, ta có thể viết lại công thức trên như sau:

$$ P(x|y) = P(x_1|y)P(x_2|y)\ldots P(x_n|y) $$

trong đó, các giá trị $P(x_i|y)$ là xác suất mà đặc trưng $x_i$ xảy ra với điều kiện lớp $y$.

Từ đó, ta có thể viết lại công thức Bayes như sau:

$$ P(y|x) \propto P(x_1|y)P(x_2|y)\ldots P(x_n|y)P(y) $$

Đến đây, về lý thuyết, ta đã có thể tính toán được xác suất điều kiện $P(y|x)$ cho tất cả các lớp $y$ mà mô hình cần phân lớp dữ liệu đầu vào $x$ vào.

Tuy nhiên, trong thực tế, việc tính toán với công thức trên gặp phải vấn đề về tràn số (overflow) vì các giá trị xác suất $P(x_i|y)$ có thể rất nhỏ, dẫn đến tích của chúng cũng rất nhỏ, và khi nhân với $P(y)$ cũng rất nhỏ, dẫn đến kết quả cuối cùng thường được làm tròn về 0.

Để khắc phục vấn đề này, ta thường sử dụng hàm logarit tự nhiên (natural logarithm) để tính toán.
Ta có công thức như sau:

$$ \log(P(y|x)) \propto \log(P(x_1|y)) + \log(P(x_2|y)) + \ldots + \log(P(x_n|y)) + \log(P(y)) $$

Việc tìm giá trị lớn nhất trong các giá trị xác suất điều kiện $P(y|x)$ vẫn không thay đổi, vì hàm logarit là một hàm đồng biến (monotonic function), tức là nếu $a > b$ thì $\log(a) > \log(b)$.

Do đó, ta có thể viết lại công thức Bayes như sau:
$$ y_{pred} = \arg\max_{y} \log(P(y|x)) $$
$$ y_{pred} = \arg\max_{y} \left( \log(P(x_1|y)) + \log(P(x_2|y)) + \ldots + \log(P(x_n|y)) + \log(P(y)) \right) $$
$$ y_{pred} = \arg\max_{y} \left( \sum_{i=1}^{n} \log(P(x_i|y)) + \log(P(y)) \right) $$

## 4. Ví dụ minh hoạ

Xét ví dụ bài toán Phân lớp văn bản: Phân lớp các lời nhận xét về món ăn thành 2 lớp: tích cực và tiêu cực.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/9-naive-bayes/food_positive_negative.jpeg" style="width: 600px;"/>

Ta có bộ dữ liệu train như sau:

| Văn bản                      | Nhãn     |
| ---------------------------- | -------- |
| tôi thích món ăn này         | Positive |
| thật tuyệt vời và ngon miệng | Positive |
| tôi không thích món ăn này   | Negative |
| món ăn tệ và quá mặn         | Negative |
| thật tuyệt và hấp dẫn        | Positive |
| không ngon và thất vọng      | Negative |

### 4.1: Xử lý dữ liệu văn bản và xây dựng từ điển

Ta tách từ trong mỗi văn bản (Tokenization)

| Văn bản                      | Tokenized Text                |
| ---------------------------- | ----------------------------- |
| tôi thích món ăn này         | "tôi", "thích", "món", "ăn", "này" |
| thật tuyệt vời và ngon miệng | "thật", "tuyệt", "vời", "và", "ngon", "miệng" |
| tôi không thích món ăn này   | "tôi", "không", "thích", "món", "ăn", "này" |
| món ăn tệ và quá mặn         | "món", "ăn", "tệ", "và", "quá", "mặn" |
| thật tuyệt và hấp dẫn        | "thật", "tuyệt", "và", "hấp", "dẫn" |
| không ngon và thất vọng      | "không", "ngon", "và", "thất", "vọng" |

Ta xây dựng từ điển từ các từ trong các văn bản trên, bằng việc lấy tất cả từ đã tách, bỏ trùng.
Từ điển bao gồm tổng cộng 19 từ 

```python
['tôi', 'thích', 'món', 'ăn', 'này', 'thật', 'tuyệt', 'vời', 'và', 'ngon', 'miệng', 
 'không', 'tệ', 'quá', 'mặn', 'hấp', 'dẫn', 'thất', 'vọng']
```

Ta ký hiệu từ điển là $V$. $|V| = 19$ là kích thước của từ điển.

### 4.2: Tính toán xác suất mà lớp $y$ xảy ra $P(y)$

$P(y=Positive) = \frac{3}{6} = 0.5$
$P(y=Negative) = \frac{3}{6} = 0.5$

### 4.3: Tính toán xác suất mà từ $w$ xuất hiện thuộc về lớp $y$ $P(w|y)$

Ta tính toán xác suất mà từ $w$ xuất hiện thuộc về lớp $y$ bằng cách đếm số lượng từ $w$ xuất hiện trong các văn bản thuộc về lớp $y$, chia cho tổng số lượng từ trong các văn bản thuộc về lớp $y$.

$$ P(w|y) = \frac{N(w, y)}{N(y)} $$

Để tránh trường hợp từ $w$ không xuất hiện trong các văn bản thuộc về lớp $y$, ta sử dụng kỹ thuật Laplace Smoothing (Laplace smoothing) để tính toán xác suất mà từ $w$ xuất hiện thuộc về lớp $y$ như sau:

$$ P(w|y) = \frac{N(w, y) + 1}{N(y) + |V|} $$

Từ công thức trên, ta tính toán được các xác suất tương ứng với mỗi từ trong từ điển $V$ như sau, đối với lớp Positive $P(w|y=Positive)$ và đối với lớp Negative $P(w|y=Negative)$:

| Từ trong từ điển $V$ | Đối với lớp Positive | Đối với lớp Negative |
| --------------------- | ------------------ | ------------------ |
| tôi                   | $\frac{2+1}{3+19} = 0.15$ | $\frac{0+1}{3+19} = 0.05$ |
| thích                 | $\frac{1+1}{3+19} = 0.10$ | $\frac{0+1}{3+19} = 0.05$ |
| món                   | $\frac{2+1}{3+19} = 0.15$ | $\frac{1+1}{3+19} = 0.10$ |
| ăn                    | $\frac{2+1}{3+19} = 0.15$ | $\frac{1+1}{3+19} = 0.10$ |
| này                   | $\frac{1+1}{3+19} = 0.10$ | $\frac{0+1}{3+19} = 0.05$ |
| thật                 | $\frac{2+1}{3+19} = 0.15$ | $\frac{0+1}{3+19} = 0.05$ |
| tuyệt                | $\frac{2+1}{3+19} = 0.15$ | $\frac{0+1}{3+19} = 0.05$ |
| vời                  | $\frac{1+1}{3+19} = 0.10$ | $\frac{0+1}{3+19} = 0.05$ |
| và                   | $\frac{2+1}{3+19} = 0.15$ | $\frac{2+1}{3+19} = 0.15$ |
| ngon                 | $\frac{1+1}{3+19} = 0.10$ | $\frac{1+1}{3+19} = 0.10$ |
| miệng                | $\frac{1+1}{3+19} = 0.10$ | $\frac{0+1}{3+19} = 0.05$ |
| không                | $\frac{0+1}{3+19} = 0.05$ | $\frac{2+1}{3+19} = 0.15$ |
| tệ                   | $\frac{0+1}{3+19} = 0.05$ | $\frac{1+1}{3+19} = 0.10$ |
| quá                  | $\frac{0+1}{3+19} = 0.05$ | $\frac{1+1}{3+19} = 0.10$ |
| mặn                 | $\frac{0+1}{3+19} = 0.05$ | $\frac{1+1}{3+19} = 0.10$ |
| hấp                  | $\frac{0+1}{3+19} = 0.05$ | $\frac{0+1}{3+19} = 0.05$ |
| dẫn                  | $\frac{0+1}{3+19} = 0.05$ | $\frac{0+1}{3+19} = 0.05$ |
| thất                 | $\frac{0+1}{3+19} = 0.05$ | $\frac{1+1}{3+19} = 0.10$ |
| vọng                | $\frac{0+1}{3+19} = 0.05$ | $\frac{1+1}{3+19} = 0.10$ |

### 4.4: Tính toán phân phối xác suất của dữ liệu đầu vào mới

Xét văn bản mới: "thật ngon và tuyệt".

Ta tách từ trong văn bản này: "thật", "ngon", "và", "tuyệt"

Ta tính toán xác suất điều kiện $P(y|x)$ cho từng lớp $y$ như sau:
- Đối với lớp $y=Positive$:
$$ log(P(y=Positive|x)) \propto \log(P(thật|y=Positive)) + \log(P(ngon|y=Positive)) + \log(P(và|y=Positive)) + \log(P(tuyệt|y=Positive)) + \log(P(y=Positive)) $$
$$ log(P(y=Positive|x)) \propto \log(0.15) + \log(0.10) + \log(0.15) + \log(0.15) + \log(0.5) $$
$$ log(P(y=Positive|x)) \propto -1.897 + -2.303 + -1.897 + -1.897 + -0.693 $$
$$ log(P(y=Positive|x)) \propto -8.687 $$

- Đối với lớp $y=Negative$:
$$ log(P(y=Negative|x)) \propto \log(P(thật|y=Negative)) + \log(P(ngon|y=Negative)) + \log(P(và|y=Negative)) + \log(P(tuyệt|y=Negative)) + \log(P(y=Negative)) $$
$$ log(P(y=Negative|x)) \propto \log(0.05) + \log(0.10) + \log(0.15) + \log(0.05) + \log(0.5) $$
$$ log(P(y=Negative|x)) \propto -2.996 + -2.303 + -1.897 + -2.996 + -0.693 $$
$$ log(P(y=Negative|x)) \propto -10.885 $$

Từ kết quả trên, ta đưa ra dự đoán cho văn bản mới thuộc lớp Positive vì $log(P(y=Positive|x)) > log(P(y=Negative|x))$.

## 5. Ưu và nhược điểm của Naive Bayes Classification

- Ưu điểm:
    - Đơn giản, dễ hiểu và dễ triển khai.
    - Tính toán nhanh, đặc biệt là với các tập dữ liệu lớn.
- Nhược điểm:
    - Giả định rằng các đặc trưng trong dữ liệu đầu vào là độc lập với nhau, điều này có thể không đúng trong thực tế.
