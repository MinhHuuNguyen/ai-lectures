---
time: 12/30/2021
title: Vòng lặp trong Python
description: "Vòng lặp là một cấu trúc cơ bản trong lập trình giúp chương trình thực hiện một hành động nào đó lặp đi lặp lại. Vòng lặp giúp giảm thiểu việc lặp lại mã nguồn, giúp chương trình trở nên ngắn gọn và dễ đọc. Trong Python, có hai loại vòng lặp cơ bản là vòng lặp `for` và vòng lặp `while`."
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/0_series/images/python-logo.png
tags: [python]
is_highlight: false
is_published: true
---

## 1. Vòng lặp `for`

Vòng lặp `for` trong Python giúp chương trình thực hiện một hành động nào đó lặp đi lặp lại trên một tập hợp dữ liệu nào đó.

```python
for item in iterable:
    # do something with item
```

Ví dụ:

```python
for i in range(5):
    print(i)

# Output:
# 0
# 1
# 2
# 3
# 4
```

```python
for i in [1, 2, 3, 4, 5]:
    print(i)

# Output:
# 1
# 2
# 3
# 4
# 5
```

```python
for i in "Hello":
    print(i)

# Output:
# H
# e
# l
# l
# o
```

## 2. Một số hàm hỗ trợ vòng lặp `for`

### 2.1. Hàm `range()`

Hàm `range()` trong Python giúp tạo ra một dãy số nguyên từ một số bắt đầu đến một số kết thúc với bước nhảy cố định.

```python
for i in range(5):
    print(i)

# Output:
# 0
# 1
# 2
# 3
# 4
```

```python
for i in range(1, 6):
    print(i)

# Output:
# 1
# 2
# 3
# 4
# 5
```

```python
for i in range(0, 10, 2):
    print(i)

# Output:
# 0
# 2
# 4
# 6
# 8
```

### 2.2. Hàm `enumerate()`

Hàm `enumerate()` trong Python giúp chúng ta lặp qua một tập hợp dữ liệu và trả về cả index và giá trị của mỗi phần tử.

```python
for index, value in enumerate([1, 2, 3, 4, 5]):
    print('index', index, 'value', value)

# Output:
# index 0 value 1
# index 1 value 2
# index 2 value 3
# index 3 value 4
# index 4 value 5
```

### 2.3. Hàm `zip()`

Hàm `zip()` trong Python giúp chúng ta kết hợp các phần tử từ nhiều tập hợp dữ liệu thành một tập hợp dữ liệu duy nhất.

```python
for i, j in zip([1, 2, 3], [4, 5, 6]):
    print(i, j)

# Output:
# 1 4
# 2 5
# 3 6
```

### 2.4. Thư viện `tqdm`

Thư viện `tqdm` trong Python giúp chúng ta hiển thị thanh tiến trình khi chúng ta lặp qua một tập hợp dữ liệu.

```python
from tqdm import tqdm

for i in tqdm(range(100)):
    pass
```

https://github.com/tqdm/tqdm

### 2.5. List comprehension trong Python

List comprehension trong Python giúp chúng ta tạo ra một danh sách mới từ một danh sách đã có một cách ngắn gọn với tốc độ thực thi nhanh hơn.

Ví dụ về cách viết thông thường:

```python
squares = []
for i in range(10):
    squares.append(i**2)
print(squares)

# Output:
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

Ví dụ về cách viết bằng list comprehension:

```python
squares = [i**2 for i in range(10)]
print(squares)

# Output:
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

Ví dụ về cách viết với điều kiện:

```python
even_squares = [i**2 for i in range(10) if i % 2 == 0]
print(even_squares)

# Output:
# [0, 4, 16, 36, 64]
```

## 3. Vòng lặp `while`

Vòng lặp `while` trong Python giúp chương trình thực hiện một hành động nào đó lặp đi lặp lại cho đến khi điều kiện đầu vào không còn đúng.

```python
while condition:
    # do something
```

Ví dụ:

```python
i = 0
while i < 5:
    print(i)
    i += 1

# Output:
# 0
# 1
# 2
# 3
# 4
```

## 4. Vòng lặp lồng nhau

Vòng lặp lồng nhau trong Python giúp chúng ta thực hiện một vòng lặp bên trong một vòng lặp khác.

```python
for i in range(2):
    for j in range(3):
        print(i, j)

# Output:
# 0 0
# 0 1
# 0 2
# 1 0
# 1 1
# 1 2
```

```python
for i in range(2):
    j = 0
    while j < 3:
        print(i, j)
        j += 1

# Output:
# 0 0
# 0 1
# 0 2
# 1 0
# 1 1
# 1 2
```

```python
i = 0
while i < 2:
    j = 0
    while j < 3:
        print(i, j)
        j += 1
    i += 1

# Output:
# 0 0
# 0 1
# 0 2
# 1 0
# 1 1
# 1 2
```

## 5. Câu lệnh `break`

Câu lệnh `break` trong Python giúp chương trình thoát khỏi vòng lặp gần nhất ngay lập tức khi gặp câu lệnh `break`.

```python
i = 0
while i < 5:
    print(i)
    if i == 3:
        break
    i += 1

# Output:
# 0
# 1
# 2
# 3
```

```python
for i in range(5):
    print(i)
    if i == 3:
        break

# Output:
# 0
# 1
# 2
# 3
```

```python
for i in range(3):
    for j in range(5):
        print(i, j)
        if j == 1:
            break

# Output:
# 0 0
# 0 1
# 1 0
# 1 1
# 2 0
# 2 1
```

## 6. Câu lệnh `continue`

Câu lệnh `continue` trong Python giúp chương trình bỏ qua phần còn lại của vòng lặp và tiếp tục với lượt lặp tiếp theo.

```python
for i in range(5):
    if i == 2:
        continue
    print(i)

# Output:
# 0
# 1
# 3
# 4
```

```python
for i in range(3):
    for j in range(5):
        if j == 2:
            continue
        print(i, j)

# Output:
# 0 0
# 0 1
# 0 3
# 0 4
# 1 0
# 1 1
# 1 3
# 1 4
# 2 0
# 2 1
# 2 3
# 2 4
```

```python
i = 0
while i < 5:
    i += 1
    if i == 3:
        continue
    print(i)

# Output:
# 1
# 2
# 4
# 5
```

## 7. Câu lệnh `else` trong vòng lặp

Câu lệnh `else` trong vòng lặp giúp chương trình thực hiện một hành động nào đó khi vòng lặp kết thúc mà không gặp câu lệnh `break`.

```python
for i in range(5):
    print(i)
else:
    print("End of loop")

# Output:
# 0
# 1
# 2
# 3
# 4
# End of loop
```

```python
for i in range(5):
    if i == 3:
        break
    print(i)
else:
    print("End of loop")

# Output:
# 0
# 1
# 2
```

```python
for i in range(5):
    if i == 3:
        continue
    print(i)
else:
    print("End of loop")

# Output:
# 0
# 1
# 2
# 4
# End of loop
```
