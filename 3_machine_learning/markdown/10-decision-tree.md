---
time: 08/12/2022
title: Mô hình Decision Tree
description: Decision Tree là một trong những mô hình học máy khá cổ điển nhưng vẫn được sử dụng rất nhiều trong thực tế. Mô hình này có thể được sử dụng cho cả bài toán phân loại và hồi quy. Hơn nữa, Decision Tree cũng là một trong những mô hình dễ hiểu và dễ giải thích nhất trong các mô hình học máy.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/banner.png
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Tổng quan

Decision Tree là một trong những mô hình học máy khá cổ điển nhưng vẫn được sử dụng rất nhiều trong thực tế.
Mô hình này có thể được sử dụng cho cả bài toán phân loại và hồi quy.

Hơn nữa, Decision Tree cũng là một trong những mô hình dễ hiểu và dễ giải thích nhất trong các mô hình học máy.
Lý do mà mô hình Decision Tree dễ hiểu và dễ giải thích là vì nó mô phỏng lại cách mà con người đưa ra quyết định.

Nói cách khác, Decision Tree có thể được nhìn như là một chuỗi các mệnh đề logic if ..... else .....

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/example.png" style="width: 400px;"/>

## 2. Các khái niệm trong Decision Tree

Nút (Node) là một điểm trên Decision Tree, một node có thể chứa một câu hỏi hoặc một dự đoán của mô hình.
- Nút gốc (Root Node) là nút đầu tiên của Decision Tree, nó chứa tất cả các dữ liệu đầu vào.
- Nút lá (Leaf Node hay Terminal Node) là nút cuối cùng của Decision Tree, nó chứa các dự đoán của mô hình.
- Nút phân nhánh (Branch Node) là nút nằm giữa các nút gốc và nút lá, nó chứa các câu hỏi để phân loại dữ liệu.
- Nhánh (Branch) là một đường nối giữa các nút, chứa câu trả lời của câu hỏi ở nút phân nhánh.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/parent_child_sibling.png" style="width: 400px;"/>

Các khái niệm mô tả mối quan hệ giữa các nút trong Decision Tree:
- Node cha (Parent Node) là nút nằm trên một nút khác, nó chứa dữ liệu đầu vào cho nút con.
- Node con (Child Node) là nút nằm dưới một nút khác, nó chứa dữ liệu đầu ra của nút cha.
- Node chị em (Sibling Node) là các nút nằm trên cùng một nhánh, chúng có cùng một node cha.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/non_binary_vs_binary.png" style="width: 800px;"/>

Nếu các câu hỏi ở các nút phân nhánh là các câu hỏi nhị phân (có 2 câu trả lời đúng sai), hay nói cách khác, các node cha chỉ có đúng 2 node con, thì Decision Tree được gọi là Decision Tree nhị phân (Binary Decision Tree).
Một Decision Tree bất kỳ nào cũng có thể chuyển đổi thành một Binary Decision Tree.

## 3. Hàm Entropy

Hàm Entropy là một hàm đo độ không chắc chắn của một biến ngẫu nhiên.
Giá trị của hàm Entropy càng lớn thì độ không chắc chắn của biến ngẫu nhiên đó càng lớn.
Ngược lại, giá trị của hàm Entropy càng nhỏ thì độ không chắc chắn của biến ngẫu nhiên đó càng nhỏ.

Hàm Entropy được sử dụng nhiều trong lý thuyết thông tin và học máy.
Hàm Entropy được định nghĩa như sau:

$$ H(X) = - \sum_{i=1}^{n} p(x_i) \cdot log_2(p(x_i)) $$

trong đó:
- $H(X)$ là độ không chắc chắn của biến ngẫu nhiên $X$.
- $x_i$ là các giá trị khác nhau của biến ngẫu nhiên $X$.
- $p(x_i)$ là xác suất của biến ngẫu nhiên $X$ nhận giá trị $x_i$.
- $n$ là số lượng giá trị khác nhau của biến ngẫu nhiên $X$.

Ví dụ: Xét biến ngẫu nhiên $X$ có 2 giá trị khác nhau là 0 và 1.
Lúc này, hàm Entropy của biến ngẫu nhiên $X$ được tính như sau:

$$ H(X) = - (p(0) \cdot log_2(p(0)) + p(1) \cdot log_2(p(1))) $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/binary_entropy_function.png" style="width: 600px;"/>

Thông qua biểu đồ trên, ta thấy:
- Nếu xác suất $p(0) = 0$ tương đương với $p(1) = 1$, thì hàm Entropy của biến ngẫu nhiên $X$ là 0 - nhỏ nhất.
Điều này có nghĩa là biến ngẫu nhiên $X$ **rất chắc chắn**.
- Nếu xác suất $p(0) = 1$ tương đương với $p(1) = 0$, thì hàm Entropy của biến ngẫu nhiên $X$ là 0 - nhỏ nhất.
Điều này có nghĩa là biến ngẫu nhiên $X$ **rất chắc chắn**.
- Nếu xác suất $p(0) = 0.5$ tương đương với $p(1) = 0.5$, thì hàm Entropy của biến ngẫu nhiên $X$ là 1 - lớn nhất.
Điều này có nghĩa là biến ngẫu nhiên $X$ **rất không chắc chắn**.

## 4. Decision Tree với hàm Entropy (ID3)

Việc quan trọng nhất khi xây dựng một Decision Tree là chọn các câu hỏi ở từng nút phân nhánh sao cho hợp lý nhất.
Một cách khái quát, ta có thể đánh giá mức độ hợp lý của câu hỏi ở một nút phân nhánh thông qua việc trả lời câu hỏi: "Việc sử dụng câu hỏi này ở nút phân nhánh này có giúp mô hình dễ dàng phân loại dữ liệu hơn hay không?".

ID3 là một thuật toán xây dựng Decision Tree dựa trên hàm Entropy, mô hình ID3 sử dụng hàm Entropy để đánh giá mức độ hiệu quả của từng câu hỏi ở mỗi nút phân nhánh. Nói cách khác, ID3 sử dụng hàm Entropy như là hàm loss để tối ưu.

Vì hàm Entropy là một hàm đo độ không chắc chắn của một biến ngẫu nhiên, nên ID3 sẽ chọn các câu hỏi ở các nút phân nhánh sao cho độ không chắc chắn của biến ngẫu nhiên ở các nút con là nhỏ nhất.
Từ đó, mô hình Decision Tree sẽ nhanh chóng đưa ra được các quyết định dự đoán.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/low_vs_high_entropy.png" style="width: 1000px;"/>

Xét một bài toán phân lớp với $n$ lớp khác nhau, ta gọi $C_1, C_2, \ldots, C_n$ là các lớp khác nhau.
- Với mỗi nút phân nhánh, ta có số lượng dữ liệu trước khi phân nhánh là $N$, số lượng dữ liệu thuộc lớp $C_i$ là $N_i$.
Ta có thể tính hàm Entropy của nút phân nhánh (Entropy trước khi phân nhánh) như sau:

$$ H_{before} = - \sum_{i=1}^{n} \frac{N_i}{N} \cdot log_2(\frac{N_i}{N}) $$

- Với mỗi câu hỏi $x$ có thể được đặt ra ở nút phân nhánh, dữ liệu sẽ được chia thành $k$ nhánh khác nhau.
Mỗi nhánh câu trả lời cho câu hỏi sẽ có số lượng dữ liệu khác nhau là $m_1, m_2, \ldots, m_k$.
Ta có thể tính hàm Entropy sau khi phân nhánh như sau:

$$ H^{x}_{after} = - \sum_{j=1}^{k} \frac{m_j}{N} \cdot H_j $$

- Độ giảm Entropy là giá trị quyết định xem câu hỏi $x$ có hiệu quả hay không.
Độ giảm Entropy còn được gọi là Information Gain.
Giá trị Information Gain càng lớn thì câu hỏi $x$ càng hiệu quả, và ngược lại.
Ta có thể tính độ giảm Entropy như sau:

$$ IG(x) = H_{before} - H^{X}_{after} $$

- Ta sẽ chọn câu hỏi $X$ có độ giảm Entropy lớn nhất để đặt ở nút phân nhánh.

$$ x^* = argmax_{x} IG(x) = argmin_{x} H^{X}_{after} $$

## 5. Ví dụ minh hoạ

Xét một ví dụ minh hoạ với bộ dữ liệu cho bài toán phân loại trái cây và rau củ.
Bộ dữ liệu này có 3 thuộc tính là Color (Màu sắc), Texture (Kết cấu) và Taste (Hương vị).

Bộ dữ liệu này có 10 mẫu dữ liệu với các thuộc tính như sau:

| Color  | Texture | Taste  | Label     |
|--------|---------|--------|-----------|
| Red    | Smooth  | Sweet  | Fruit     |
| Yellow | Smooth  | Sweet  | Fruit     |
| Green  | Rough   | Bland  | Vegetable |
| Red    | Rough   | Sweet  | Vegetable |
| Yellow | Smooth  | Sweet  | Fruit     |
| Green  | Smooth  | Bitter | Vegetable |
| Red    | Rough   | Bitter | Vegetable |
| Yellow | Rough   | Bland  | Fruit     |
| Green  | Smooth  | Bland  | Vegetable |
| Red    | Smooth  | Bitter | Fruit     |

### 5.1. Tính Entropy gốc

$$ H_{before} = - \sum_{i=1}^{2} \frac{N_i}{N} \cdot log_2(\frac{N_i}{N}) $$
$$ H_{before} = - \left( \frac{N_{Fruit}}{N} \cdot log_2(\frac{N_{Fruit}}{N}) + \frac{N_{Vegetable}}{N} \cdot log_2(\frac{N_{Vegetable}}{N}) \right) $$
$$ H_{before} = - \left( \frac{5}{10} \cdot log_2(\frac{5}{10}) + \frac{5}{10} \cdot log_2(\frac{5}{10}) \right) $$
$$ H_{before} = 1 $$

### 5.2. Tính Entropy sau khi phân nhánh với từng thuộc tính

#### 5.2.1. Color - Red, Yellow và Green.

- Red: 2 mẫu Fruit và 2 mẫu Vegetable, Entropy của nhánh Red:

$$ H_{Red} = - \left( \frac{N_{Fruit}}{N} \cdot log_2(\frac{N_{Fruit}}{N}) + \frac{N_{Vegetable}}{N} \cdot log_2(\frac{N_{Vegetable}}{N}) \right) $$

$$ H_{Red} = - \left( \frac{2}{4} \cdot log_2(\frac{2}{4}) + \frac{2}{4} \cdot log_2(\frac{2}{4}) \right) $$

$$ H_{Red} = 1 $$

- Yellow: 3 mẫu Fruit và 0 mẫu Vegetable, Entropy của nhánh Yellow:

$$ H_{Yellow} = - \left( \frac{N_{Fruit}}{N} \cdot log_2(\frac{N_{Fruit}}{N}) + \frac{N_{Vegetable}}{N} \cdot log_2(\frac{N_{Vegetable}}{N}) \right) $$

$$ H_{Yellow} = - \left( \frac{3}{3} \cdot log_2(\frac{3}{3}) + \frac{0}{3} \cdot log_2(\frac{0}{3}) \right) $$

$$ H_{Yellow} = 0 $$

- Green: 0 mẫu Fruit và 3 mẫu Vegetable, Entropy của nhánh Green:

$$ H_{Green} = - \left( \frac{N_{Fruit}}{N} \cdot log_2(\frac{N_{Fruit}}{N}) + \frac{N_{Vegetable}}{N} \cdot log_2(\frac{N_{Vegetable}}{N}) \right) $$

$$ H_{Green} = - \left( \frac{0}{3} \cdot log_2(\frac{0}{3}) + \frac{3}{3} \cdot log_2(\frac{3}{3}) \right) $$

$$ H_{Green} = 0 $$

Ta có thể tính hàm Entropy sau khi phân nhánh với thuộc tính Color như sau:

$$ H^{Color}_{after} = - \left( \frac{4}{10} \cdot H_{Red} + \frac{3}{10} \cdot H_{Yellow} + \frac{3}{10} \cdot H_{Green} \right) $$

$$ H^{Color}_{after} = - \left( \frac{4}{10} \cdot 1 + \frac{3}{10} \cdot 0 + \frac{3}{10} \cdot 0 \right) $$

$$ H^{Color}_{after} = 0.4 $$

$$ IG(Color) = H_{before} - H^{Color}_{after} $$

$$ IG(Color) = 1 - 0.4 $$

$$ IG(Color) = 0.6 $$

#### 5.2.2. Texture - Smooth và Rough.

- Smooth: 4 mẫu Fruit và 2 mẫu Vegetable, Entropy của nhánh Smooth:

$$ H_{Smooth} = - \left( \frac{N_{Fruit}}{N} \cdot log_2(\frac{N_{Fruit}}{N}) + \frac{N_{Vegetable}}{N} \cdot log_2(\frac{N_{Vegetable}}{N}) \right) $$

$$ H_{Smooth} = - \left( \frac{4}{6} \cdot log_2(\frac{4}{6}) + \frac{2}{6} \cdot log_2(\frac{2}{6}) \right) $$

$$ H_{Smooth} = 0.9183 $$

- Rough: 1 mẫu Fruit và 3 mẫu Vegetable, Entropy của nhánh Rough:

$$ H_{Rough} = - \left( \frac{N_{Fruit}}{N} \cdot log_2(\frac{N_{Fruit}}{N}) + \frac{N_{Vegetable}}{N} \cdot log_2(\frac{N_{Vegetable}}{N}) \right) $$

$$ H_{Rough} = - \left( \frac{1}{4} \cdot log_2(\frac{1}{4}) + \frac{3}{4} \cdot log_2(\frac{3}{4}) \right) $$

$$ H_{Rough} = 0.8113 $$

Ta có thể tính hàm Entropy sau khi phân nhánh với thuộc tính Texture như sau:

$$ H^{Texture}_{after} = - \left( \frac{6}{10} \cdot H_{Smooth} + \frac{4}{10} \cdot H_{Rough} \right) $$

$$ H^{Texture}_{after} = - \left( \frac{6}{10} \cdot 0.9183 + \frac{4}{10} \cdot 0.8113 \right) $$

$$ H^{Texture}_{after} = 0.8755 $$

$$ IG(Texture) = H_{before} - H^{Texture}_{after} $$

$$ IG(Texture) = 0.1245 $$

#### 5.2.3. Taste - Sweet, Bitter và Bland.

- Sweet: 3 mẫu Fruit và 1 mẫu Vegetable, Entropy của nhánh Sweet:

$$ H_{Sweet} = - \left( \frac{N_{Fruit}}{N} \cdot log_2(\frac{N_{Fruit}}{N}) + \frac{N_{Vegetable}}{N} \cdot log_2(\frac{N_{Vegetable}}{N}) \right) $$

$$ H_{Sweet} = - \left( \frac{3}{4} \cdot log_2(\frac{3}{4}) + \frac{1}{4} \cdot log_2(\frac{1}{4}) \right) $$

$$ H_{Sweet} = 0.8113 $$

- Bitter: 1 mẫu Fruit và 2 mẫu Vegetable, Entropy của nhánh Bitter:

$$ H_{Bitter} = - \left( \frac{N_{Fruit}}{N} \cdot log_2(\frac{N_{Fruit}}{N}) + \frac{N_{Vegetable}}{N} \cdot log_2(\frac{N_{Vegetable}}{N}) \right) $$

$$ H_{Bitter} = - \left( \frac{1}{3} \cdot log_2(\frac{1}{3}) + \frac{2}{3} \cdot log_2(\frac{2}{3}) \right) $$

$$ H_{Bitter} = 0.9183 $$

- Bland: 1 mẫu Fruit và 2 mẫu Vegetable, Entropy của nhánh Bland:

$$ H_{Bland} = - \left( \frac{N_{Fruit}}{N} \cdot log_2(\frac{N_{Fruit}}{N}) + \frac{N_{Vegetable}}{N} \cdot log_2(\frac{N_{Vegetable}}{N}) \right) $$

$$ H_{Bland} = - \left( \frac{1}{3} \cdot log_2(\frac{1}{3}) + \frac{2}{3} \cdot log_2(\frac{2}{3}) \right) $$

$$ H_{Bland} = 0.9183 $$

Ta có thể tính hàm Entropy sau khi phân nhánh với thuộc tính Taste như sau:

$$ H^{Taste}_{after} = - \left( \frac{4}{10} \cdot H_{Sweet} + \frac{3}{10} \cdot H_{Bitter} + \frac{3}{10} \cdot H_{Bland} \right) $$

$$ H^{Taste}_{after} = - \left( \frac{4}{10} \cdot 0.8113 + \frac{3}{10} \cdot 0.9183 + \frac{3}{10} \cdot 0.9183 \right) $$

$$ H^{Taste}_{after} = 0.8755 $$

$$ IG(Taste) = H_{before} - H^{Taste}_{after} $$

$$ IG(Taste) = 0.1245 $$

### 5.3. Chọn thuộc tính phân nhánh

Với các giá trị Information Gain đã tính toán ở trên, ta có thể chọn thuộc tính phân nhánh như sau:
- $IG(Color) = 0.6$
- $IG(Texture) = 0.1245$
- $IG(Taste) = 0.1245$

Vì $IG(Color)$ lớn nhất, nên ta sẽ chọn thuộc tính Color để phân nhánh đầu tiên.

Sau khi phân nhánh, ta sẽ có 3 nhánh là Red, Yellow và Green.
- Nhánh Red sẽ có 4 mẫu dữ liệu với 2 mẫu Fruit và 2 mẫu Vegetable, cần phân nhánh tiếp.
- Nhánh Yellow sẽ có 3 mẫu dữ liệu với 3 mẫu Fruit và 0 mẫu Vegetable, không cần phân nhánh tiếp, nhánh lá với nhãn là Fruit.
- Nhánh Green sẽ có 3 mẫu dữ liệu với 0 mẫu Fruit và 3 mẫu Vegetable, không cần phân nhánh tiếp, nhánh lá với nhãn là Vegetable.

### 5.4. Phân nhánh tiếp với nhánh Red

Với nhánh Red, ta sẽ tiếp tục phân nhánh với các thuộc tính Texture và Taste.

#### 5.4.1. Texture - Smooth và Rough.

- Smooth: 2 mẫu Fruit và 0 mẫu Vegetable, Entropy của nhánh Smooth:

$$ H_{Smooth} = - \left( \frac{2}{2} \cdot log_2(\frac{2}{2}) + \frac{0}{2} \cdot log_2(\frac{0}{2}) \right) $$

$$ H_{Smooth} = 0 $$

- Rough: 0 mẫu Fruit và 2 mẫu Vegetable, Entropy của nhánh Rough:

$$ H_{Rough} = - \left( \frac{0}{2} \cdot log_2(\frac{0}{2}) + \frac{2}{2} \cdot log_2(\frac{2}{2}) \right) $$

$$ H_{Rough} = 0 $$

Ta có thể tính hàm Entropy sau khi phân nhánh với thuộc tính Texture như sau:

$$ H^{Texture}_{after} = 0 $$

$$ IG(Texture) = 1 $$

#### 5.4.2. Taste - Sweet và Bitter.

- Sweet: 1 mẫu Fruit và 1 mẫu Vegetable, Entropy của nhánh Sweet:

$$ H_{Sweet} = - \left( \frac{1}{2} \cdot log_2(\frac{1}{2}) + \frac{1}{2} \cdot log_2(\frac{1}{2}) \right) $$

$$ H_{Sweet} = 1 $$

- Bitter: 1 mẫu Fruit và 1 mẫu Vegetable, Entropy của nhánh Bitter:

$$ H_{Bitter} = - \left( \frac{1}{2} \cdot log_2(\frac{1}{2}) + \frac{1}{2} \cdot log_2(\frac{1}{2}) \right) $$

$$ H_{Bitter} = 1 $$

Ta có thể tính hàm Entropy sau khi phân nhánh với thuộc tính Taste như sau:

$$ H^{Taste}_{after} = 1 $$

$$ IG(Taste) = 0 $$

### 5.5. Chọn thuộc tính phân nhánh

Với các giá trị Information Gain đã tính toán ở trên, ta có thể chọn thuộc tính phân nhánh như sau:
- $IG(Texture) = 1$
- $IG(Taste) = 0$

Vì $IG(Texture)$ lớn nhất, nên ta sẽ chọn thuộc tính Texture để phân nhánh tiếp theo.
Sau khi phân nhánh, ta sẽ có 2 nhánh là Smooth và Rough.
- Nhánh Smooth sẽ có 2 mẫu dữ liệu với 2 mẫu Fruit và 0 mẫu Vegetable, không cần phân nhánh tiếp, nhánh lá với nhãn là Fruit.
- Nhánh Rough sẽ có 2 mẫu dữ liệu với 0 mẫu Fruit và 2 mẫu Vegetable, không cần phân nhánh tiếp, nhánh lá với nhãn là Vegetable.

### 5.6. Kết quả

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/example_results.png" style="width: 800px;"/>

## 6. Decision Tree với bài toán hồi quy

Decision Tree cũng có thể được sử dụng cho bài toán hồi quy.
Tuy nhiên, hàm Entropy không thể được sử dụng cho bài toán hồi quy.
Thay vào đó, Decision Tree sẽ sử dụng hàm Mean Squared Error (MSE) để đánh giá mức độ hiệu quả của từng câu hỏi ở mỗi nút phân nhánh.

Ví dụ: Xét bài toán hồi quy Định giá giá trị căn nhà với bộ dữ liệu như sau:

| Diện tích | Số lượng phòng ngủ | Giá trị |
|-----------|--------------------|---------|
| 100       | 2                  | 1.000   |
| 150       | 3                  | 1.500   |
| 200       | 4                  | 2.000   |
| 250       | 5                  | 2.500   |
| 300       | 6                  | 3.000   |
| 350       | 7                  | 3.500   |
| 400       | 8                  | 4.000   |
| 450       | 9                  | 4.500   |
| 500       | 10                 | 5.000   |

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/regression.png" style="width: 1000px;"/>

## 7. Ưu và nhược điểm của mô hình Decision Tree

- Ưu điểm:
  - Dễ hiểu và dễ giải thích.
  - Không cần tiền xử lý dữ liệu.
  - Có thể xử lý dữ liệu phân loại và hồi quy.

- Nhược điểm:
  - Dễ bị overfitting.
  
## 8. Các kỹ thuật giảm thiểu hiện tượng overfitting

Đối với mô hình Decision Tree, hiện tượng overfitting có thể xảy ra một cách dễ dàng.
Điều này dẫn đến việc mô hình sẽ không thể tổng quát hóa tốt cho các phần tử trong bộ dữ liệu kiểm tra.

Với một Decision Tree không có bất kỳ ràng buộc nào, mô hình sẽ dễ dàng để phân lớp chính xác 100% tất cả các phần tử trong bộ dữ liệu huấn luyện bằng việc liên tục phân nhánh cho đến khi mỗi nhánh chỉ còn lại một phần tử.

Để giảm thiểu hiện tượng overfitting, có một số kỹ thuật có thể được sử dụng như sau:
- **Nếu độ sâu của Decision Tree đạt đến một ngưỡng nào đó thì dừng lại**: Giới hạn độ sâu của Decision Tree sẽ giúp mô hình không bị phân nhánh quá sâu.
- **Nếu số lượng phần tử trong một nhánh nhỏ hơn một ngưỡng nào đó thì dừng lại**: Giới hạn số lượng phần tử trong một nhánh sẽ giúp mô hình không bị "chính xác 100%".
- **Nếu số lượng lá lớn hơn một ngưỡng nào đó thì dừng lại**: Giới hạn số lượng lá sẽ giúp mô hình không phân nhánh quá nhiều.
- **Nếu information gain nhỏ hơn một ngưỡng nào đó thì dừng lại**: Giới hạn độ giảm Entropy sẽ giúp mô hình không phân nhánh với "độ chính xác 100%".

Tuy nhiên, trong thực tế, việc đặt các ngưỡng giới hạn này là rất khó khăn.
Các ngưỡng này sẽ phụ thuộc vào từng bài toán cụ thể và từng bộ dữ liệu cụ thể.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/prune.png" style="width: 1000px;"/>

Do đó, một cách khác là kỹ thuật **cắt tỉa (pruning)**.
- Bước 1: Xây dựng một Decision Tree hoàn chỉnh với tất cả các nhánh và lá.
Lúc này, mô hình sẽ có độ chính xác 100% trên bộ dữ liệu huấn luyện.
- Bước 2: Cắt tỉa các nhánh và lá không cần thiết, một số nhánh sẽ trở thành lá và độ chính xác trên bộ dữ liệu huấn luyện sẽ giảm xuống.
- Bước 3: Đánh giá độ chính xác của quá trình cắt tỉa
    - Cách 1: Sử dụng bộ dữ liệu kiểm tra để đánh giá độ chính xác của mô hình.
    - Cách 2: Thêm số hạng regularization vào hàm loss.
    Vai trò của số hạng regularization là để giảm thiểu độ phức tạp của mô hình.
    Tương tự như số lượng trọng số w trong các mô hình tuyến tính, số hạng regularization là một hàm số đo độ phức tạp của mô hình thông qua số lượng nhánh và lá.
