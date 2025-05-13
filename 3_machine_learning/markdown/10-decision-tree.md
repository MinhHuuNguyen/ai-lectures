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

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/example.png" style="width: 600px;"/>

## 2. Các khái niệm trong Decision Tree

Nút (Node) là một điểm trên Decision Tree, một node có thể chứa một câu hỏi hoặc một dự đoán của mô hình.
- Nút gốc (Root Node) là nút đầu tiên của Decision Tree, nó chứa tất cả các dữ liệu đầu vào.
- Nút lá (Leaf Node hay Terminal Node) là nút cuối cùng của Decision Tree, nó chứa các dự đoán của mô hình.
- Nút phân nhánh (Branch Node) là nút nằm giữa các nút gốc và nút lá, nó chứa các câu hỏi để phân loại dữ liệu.
- Nhánh (Branch) là một đường nối giữa các nút, chứa câu trả lời của câu hỏi ở nút phân nhánh.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/parent_child_sibling.png" style="width: 600px;"/>

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

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/10-decision-tree/non_binary_vs_binary.png" style="width: 600px;"/>

Thông qua biểu đồ trên, ta thấy:
- Nếu xác suất $p(0) = 0$ tương đương với $p(1) = 1$, thì hàm Entropy của biến ngẫu nhiên $X$ là 0 - nhỏ nhất.
Điều này có nghĩa là biến ngẫu nhiên $X$ **rất chắc chắn**.
- Nếu xác suất $p(0) = 1$ tương đương với $p(1) = 0$, thì hàm Entropy của biến ngẫu nhiên $X$ là 0 - nhỏ nhất.
Điều này có nghĩa là biến ngẫu nhiên $X$ **rất chắc chắn**.
- Nếu xác suất $p(0) = 0.5$ tương đương với $p(1) = 0.5$, thì hàm Entropy của biến ngẫu nhiên $X$ là 1 - lớn nhất.
Điều này có nghĩa là biến ngẫu nhiên $X$ **rất không chắc chắn**.

## 4. Decision Tree với hàm Entropy (ID3)

ID3 là một thuật toán xây dựng Decision Tree dựa trên hàm Entropy, cụ thể, ID3 sử dụng hàm Entropy như là hàm loss để tối ưu.

Xét một bài toán phân lớp với $n$ lớp khác nhau, ta gọi $C_1, C_2, \ldots, C_n$ là các lớp khác nhau.
Với mỗi 

## 5. Ví dụ minh hoạ

## 6. Ưu và nhược điểm của mô hình Decision Tree


