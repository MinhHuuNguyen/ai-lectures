---
time: 12/23/2021
title: Kiểu dữ liệu và toán tử trong Python
description: Để bắt đầu với bất kỳ ngôn ngữ lập trình nào, và Python cũng không ngoại lệ, chúng ta cần hiểu rõ về kiểu dữ liệu và toán tử cơ bản. Các kiểu dữ liệu giúp chúng ta lưu trữ và thao tác với dữ liệu một cách hiệu quả, trong khi các toán tử cho phép thực hiện các phép toán và so sánh giữa các giá trị. Bài viết này sẽ giới thiệu về các kiểu dữ liệu phổ biến trong Python như số học, xâu chuỗi, tập hợp, từ điển, cũng như các toán tử cơ bản như toán tử số học, so sánh và logic.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/1-introduction/python_logo.jpeg
tags: [python]
is_highlight: false
is_published: true
---

## 1. Kiểm tra kiểu dữ liệu của một biến

```python
a = 5
print(type(a)) # <class 'int'>
print(isinstance(a, int)) # True
print(isinstance(a, list)) # False
```

## 2. Kiểu dữ liệu số học và giá trị chân lý

Kiểu dữ liệu số nguyên `int`

```python
a = 5
print(type(a)) # <class 'int'>
print(isinstance(a, int)) # True
```

Kiểu dữ liệu số thực `float`

```python
b = 5.0
print(type(b)) # <class 'float'>
print(isinstance(b, float)) # True
```

Ngoài số nguyên và số thực, ta còn kiểu dữ liệu số phức `complex`. Tuy nhiên, trong thực tế, ta ít khi sử dụng kiểu dữ liệu này.

Kiểu dữ liệu giá trị chân lý `bool` Đúng/Sai - True/False

```python
c = True
print(type(c)) # <class 'bool'>
print(isinstance(c, bool)) # True
```

## 3. Kiểu dữ liệu xâu chuỗi

### 3.1. Kiểu dữ liệu xâu ký tự `str`

Xâu ký tự trên một dòng được đặt trong dấu `''` hoặc dấu `""`

```python
d = "Hello, World!"
print(type(d)) # <class 'str'>
print(isinstance(d, str)) # True
```

Xâu ký tự trên nhiều dòng được đặt trong dấu `''' '''` hoặc `""" """`

```python
e = """
Hello,
World!
"""
print(e) # Hello,
         # World!
```

### 3.2. Kiểu dữ liệu danh sách `list`

Danh sách được đặt trong dấu `[]`

```python
f = [1, 2.5, 3, 'stringgg', -20, 'Minh']
print(type(f)) # <class 'list'>
print(isinstance(f, list)) # True
```

### 3.3. Kiểu dữ liệu bộ `tuple`

Bộ được đặt trong dấu `()`

```python
g = (1, 2.5, 3, 'stringgg', -20, 'Minh')
print(type(g)) # <class 'tuple'>
print(isinstance(g, tuple)) # True
```

Điểm khác biệt giữa list và tuple là tuple không thể thay đổi giá trị của phần tử trong bộ.

```python
f[0] = 10
print(f) # [10, 2.5, 3, 'stringgg', -20, 'Minh']

g[0] = 10 # TypeError: 'tuple' object does not support item assignment
```

### 3.4. Một số thao tác với kiểu dữ liệu xâu chuỗi

Lấy độ dài của xâu chuỗi

```python
h = "Hello, World!"
print(len(h)) # 13

i = [1, 2, 3, 4, 5]
print(len(i)) # 5

j = (1, 2.5, 3, 'stringgg', -20, 'Minh')
print(len(j)) # 6
```

Truy cập phần tử trong xâu chuỗi

```python
k = "Hello, World!"
print(k[0]) # H
print(k[-3]) # l

l = [1, 2, 3, 4, 5]
print(l[0]) # 1
print(l[-1]) # 5

m = (1, 2.5, 3, 'stringgg', -20, 'Minh')
print(m[0]) # 1
print(m[-1]) # Minh
```

Lỗi khi truy cập phần tử vượt quá giới hạn

```python
print(k[20]) # IndexError: string index out of range
print(l[20]) # IndexError: list index out of range
print(m[20]) # IndexError: tuple index out of range
```

Lỗi khi sử dụng giá trị index không phải số nguyên

```python
print(k[1.5]) # TypeError: string indices must be integers
print(l[1.5]) # TypeError: list indices must be integers
print(m[1.5]) # TypeError: tuple indices must be integers
```

Cắt xâu chuỗi

```python
n = "Hello, World!"
print(n[2:5]) # llo
print(n[:5]) # Hello
print(n[-4:]) # orld!

o = [1, 2, 3, 4, 5]
print(o[2:4]) # [3, 4]
print(o[:3]) # [1, 2, 3]
print(o[-3:]) # [3, 4, 5]

p = (1, 2.5, 3, 'stringgg', -20, 'Minh')
print(p[2:4]) # (3, 'stringgg')
print(p[:3]) # (1, 2.5, 3)
print(p[-3:]) # ('stringgg', -20, 'Minh')
```

## 4. Kiểu dữ liệu tập hợp `set`

Tập hợp được đặt trong dấu `{}`

```python
q = {1, 2, 3, 4, 5}
print(type(q)) # <class 'set'>
print(isinstance(q, set)) # True
```

Điểm khác biệt giữa list và set là set không chứa phần tử trùng lặp.

```python
r = [1, 2, 3, 4, 5, 5, 5, 5]
print(r) # [1, 2, 3, 4, 5, 5, 5, 5]

s = {1, 2, 3, 4, 5, 5, 5, 5}
print(s) # {1, 2, 3, 4, 5}
```

Để lấy được số phần tử trong tập hợp, ta sử dụng hàm `len()`

```python
print(len(q)) # 5
```

Set không được xếp vào nhóm xâu chuỗi vì set không có thứ tự và không thể truy cập phần tử bằng index.

```python
print(q[0]) # TypeError: 'set' object is not subscriptable
```

## 5. Kiểu dữ liệu từ điển `dict`

Từ điển được đặt trong dấu `{}` với cặp key-value.
Trong đó, `key` là một kiểu dữ liệu không thay đổi và không trùng lặp, `value` có thể là bất kỳ giá trị nào.

```python
t = {
    'name' : 'Minh',
    'age': 20,
    'job': 'student',
    'gender': 'male'
}
print(type(t)) # <class 'dict'>
print(isinstance(t, dict)) # True
```

Để lấy được số phần tử trong dictionary, ta sử dụng hàm `len()`

```python
print(len(t)) # 4
```

Để truy cập giá trị của dictionary, ta sử dụng `key` và thu được `value` tương ứng.

```python
print(t['name']) # Minh
print(t['age']) # 20
```

Để lấy danh sách các `key` hoặc `value` trong dictionary, ta sử dụng hàm `keys()` hoặc `values()`

```python
print(t.keys()) # dict_keys(['name', 'age', 'job', 'gender'])
print(t.values()) # dict_values(['Minh', 20, 'student', 'male'])
```

Để lấy danh sách các cặp `key`-`value` trong dictionary, ta sử dụng hàm `items()`

```python
print(t.items()) # dict_items([('name', 'Minh'), ('age', 20), ('job', 'student), ('gender', 'male')])
```

Nếu `key` không tồn tại trong dictionary, ta sẽ nhận được lỗi `KeyError`

```python
print(t['address']) # KeyError: 'address'
```

Nếu `value` trong dictionary là một dictionary khác, ta có gọi đó là nested dictionary - dictionary lồng nhau.

```python
u = {
    'name' : 'Minh',
    'age': 20,
    'job': 'student
    'address': {
        'city': 'Hanoi',
        'district': 'Cau Giay'
    }
}
print(u['address']) # {'city': 'Hanoi', 'district': 'Cau Giay'}
print(u['address']['city']) # Hanoi
print(u['address']['district']) # Cau Giay
```

## 6. Ép kiểu giữa các kiểu dữ liệu

### 6.1. Giữa các kiểu dữ liệu số và str

```python
v = 5
print(type(v)) # <class 'int'>
print(v) # 5

v = str(v)
print(type(v)) # <class 'str'>
print(v) # 5

v = int(v)
print(type(v)) # <class 'int'>
print(v) # 5

v = float(v)
print(type(v)) # <class 'float'>
print(v) # 5.0

v = str(v)
print(type(v)) # <class 'str'>
print(v) # 5.0
```

### 6.2. Giữa list, tuple, set

```python
w = [1, 2, 3, 4, 5, 5, 5, 5]
print(type(w)) # <class 'list'>
print(w) # [1, 2, 3, 4, 5, 5, 5, 5]

w = tuple(w)
print(type(w)) # <class 'tuple'>
print(w) # (1, 2, 3, 4, 5, 5, 5, 5)

w[-1] = 10 # TypeError: 'tuple' object does not support item assignment

w = list(w)
print(type(w)) # <class 'list'>
w[-1] = 10
print(w) # [10, 2, 3, 4, 5, 5, 5, 5]

w = set(w)
print(type(w)) # <class 'set'>
print(w) # {2, 3, 4, 5, 10}
```

## 7. Toán tử trong Python

### 7.1. Toán tử số học

| Toán tử | Ý nghĩa |
|---------|---------|
| `+` | Cộng - Addition |
| `-` | Trừ - Subtraction |
| `*` | Nhân - Multiplication |
| `/` | Chia - Division |
| `//` | Chia lấy phần nguyên - Floor division |
| `%` | Chia lấy phần dư - Modulus |
| `**` | Lũy thừa - Exponentiation |

```python
a = 5
b = 2
c = -3

print(a + b) # 7
print(a - b) # 3
print(a * b) # 10
print(a / b) # 2.5
print(a // b) # 2
print(a % b) # 1
print(a ** b) # 25
```

### 7.2. Toán tử so sánh

| Toán tử | Ý nghĩa |
|---------|---------|
| `==` | Bằng - Equal |
| `!=` | Không bằng - Not equal |
| `>` | Lớn hơn - Greater than |
| `<` | Nhỏ hơn - Less than |
| `>=` | Lớn hơn hoặc bằng - Greater than or equal to |
| `<=` | Nhỏ hơn hoặc bằng - Less than or equal to |

```python
a = 5
b = 2

print(a == b) # False
print(a != b) # True
print(a > b) # True
print(a < b) # False
print(a >= b) # True
print(a <= b) # False
```

### 7.3. Toán tử logic

| Toán tử | Ý nghĩa | Giải thích |
|---------|---------|------------|
| `and` | Và - And | Trả về `True` nếu cả hai biểu thức đều đúng, `False` nếu một trong hai biểu thức sai |
| `or` | Hoặc - Or | Trả về `True` nếu một trong hai biểu thức đúng, `False` nếu cả hai biểu thức sai |
| `not` | Không - Not | Trả về `True` nếu biểu thức sai, `False` nếu biểu thức đúng |
| `in` | Thuộc - In | Trả về `True` nếu giá trị nằm trong tập hợp, `False` nếu giá trị không nằm trong tập hợp |

```python
a = 5
b = 2

print(a > 3 and b < 3) # True
print(a > 3 and b > 3) # False
print(a > 3 or b > 3) # True
print(a < 3 or b < 3) # False
print(not a > 3) # False
print(not b > 3) # True

c = [1, 2, 3, 4, 5]

print(1 in c) # True
print(6 in c) # False
```

---

## Luyện tập

<details>
<summary>Câu hỏi trắc nghiệm</summary>

**1. Kiểu dữ liệu nào dưới đây là kiểu số nguyên trong Python?**

**A.** `int`

**B.** `float`

**C.** `complex`

**D.** `bool`

**2. Giá trị của biểu thức `type(3.14)` là gì?**

**A.** `<class 'int'>`

**B.** `<class 'float'>`

**C.** `<class 'complex'>`

**D.** `<class 'decimal.Decimal'>`

**3. Kết quả của `bool(0)` là gì?**

**A.** `True`

**B.** `False`

**C.** `None`

**D.** `Error` 

**4. Kết quả của biểu thức `2 ** 3` là gì?**

**A.** `6`

**B.** `8`

**C.** `9`

**D.** `Error`

**5. Phép toán `//` trong Python dùng để làm gì?**

**A.** Phép chia lấy phần nguyên

**B.** Phép chia lấy dư

**C.** Phép nhân

**D.** Phép luỹ thừa

**6. Kiểu dữ liệu nào có thể chứa các giá trị không thay đổi sau khi được khởi tạo?**

**A.** `list`

**B.** `set`

**C.** `tuple`

**D.** `dict`

**7. Kết quả của `5 % 2` là gì?**

**A.** `2.5`

**B.** `1`

**C.** `2`

**D.** `0`

**8. Tập hợp trong Python (`set`) có thể chứa các phần tử trùng lặp không?**

**A.** Có

**B.** Không

**9. Kết quả của `bool("")` là gì?**

**A.** `True`

**B.** `False`

**C.** `None`

**D.** `Error`

**10. Biến nào dưới đây có kiểu `dict`?**

**A.** `{"name": "Alice", "age": 25}`

**B.** `[1, 2, 3]`

**C.** `(4, 5, 6)`

**D.** `{7, 8, 9}`

**11. Giá trị của biểu thức `4 == 4.0` là gì?**

**A.** `True`

**B.** `False`

**C.** `None`

**D.** `Error`

**12. Kết quả của `10 / 3` là gì?**

**A.** `3`

**B.** `3.3`

**C.** `3.3333333333333335`

**D.** `3.5`

**13. Phép toán nào sau đây được dùng để nối hai chuỗi trong Python?**

**A.** `+`

**B.** `*`

**C.** `-`

**D.** `/`

**14. Kết quả của `len("Python")` là gì?**

**A.** `5`

**B.** `6`

**C.** `7`

**D.** `Error`

**15. Phép toán `not` trong Python dùng để làm gì?**

**A.** Đảo ngược giá trị của một biểu thức logic

**B.** So sánh hai giá trị

**C.** Kiểm tra sự tồn tại của một phần tử

**D.** Kết hợp hai biểu thức logic

</details>

<details>
<summary>Đáp án</summary>

**1. A.** `int`

**2. B.** `<class 'float'>`

**3. B.** `False`

**4. B.** `8`

**5. A.** Phép chia lấy phần nguyên

**6. C.** `tuple`

**7. B.** `1`

**8. B.** Không

**9. B.** `False`

**10. A.** `{"name": "Alice", "age": 25}`

**11. A.** `True`

**12. C.** `3.3333333333333335`

**13. A.** `+`

**14. B.** `6`

**15. A.** Đảo ngược giá trị của một biểu thức logic

</details>
