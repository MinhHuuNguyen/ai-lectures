---
time: 01/06/2022
title: Hàm trong Python
description: "Hàm là một khối mã lệnh độc lập, thực hiện một công việc cụ thể và có thể được gọi ở bất kỳ đâu trong chương trình. Hàm giúp chương trình trở nên ngắn gọn, dễ đọc và dễ bảo trì. Trong Python, hàm được định nghĩa thông qua từ khóa `def` hoặc `lambda`."
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/0_series/images/python-logo.png
tags: [python]
is_highlight: false
is_published: true
---

## 1. Hàm đơn giản với câu lệnh `def`

### 1.1. Cấu trúc hàm trong Python

Hàm là một khối mã lệnh độc lập, thực hiện một công việc cụ thể và có thể được gọi ở bất kỳ đâu trong chương trình.
Hàm giúp chương trình trở nên ngắn gọn, dễ đọc và dễ bảo trì.

Hàm là một khái niệm quan trọng và được sử dụng trong hầu như tất cả các ngôn ngữ lập trình.
Trong Python, hàm được định nghĩa thông qua từ khóa `def`.

```python
def function_name(parameters):
    """docstring"""
    # do something
    return value
```
trong đó:
- `function_name`: tên của hàm, phải tuân theo quy tắc đặt tên biến, thường được đặt để mô tả ngắn gọn công việc của hàm.
- `parameters`: là các tham số mà hàm nhận vào, có thể không có tham số đầu vào hoặc nhiều tham số.
- `docstring`: là một chuỗi dùng để mô tả công việc của hàm, không bắt buộc. Vai trò của docstring giống như việc viết comment cho hàm.
- `return value`: là giá trị trả về của hàm, có thể không có hoặc nhiều giá trị trả về.

### 1.2. Hàm không có tham số đầu vào, không có giá trị trả về

```python
def hello():
    print("Hello, World!")
    
hello()

# Output:
# Hello, World!
```

### 1.3. Hàm có tham số đầu vào, không có giá trị trả về

```python
def add(a, b):
    c = a + b
    print('c =' c)

d = add(1, 2)
print('d = ', d)

# Output:
# c = 3
# d = None
```

### 1.4. Hàm không có tham số đầu vào, có giá trị trả về

```python
def pi():
    return 3.14

p = pi()
print('p = ', p)

# Output:
# p = 3.14
```

### 1.5. Hàm có tham số đầu vào, có giá trị trả về

```python
def add(a, b):
    return a + b

c = add(1, 2)
print('c = ', c)

# Output:
# c = 3
```

### 1.6. Hàm có tham số đầu vào mặc định

```python

def add(a, b=1):
    return a + b

c = add(1)
print('c = ', c)

d = add(1, 2)
print('d = ', d)

# Output:
# c = 2
# d = 3
```

### 1.7. Hàm có số lượng tham số đầu vào không xác định

```python
def add(*args):
    return sum(args)

c = add(1, 2, 3, 4, 5)
print('c = ', c)

d = add(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
print('d = ', d)

# Output:
# c = 15
# d = 55
```

```python
def add(a, b, *args):
    print('a = ', a)
    print('b = ', b)
    print('args = ', args)
    return a * b + sum(args)

c = add(1, 2, 3, 4, 5)
print('c = ', c)

# Output:
# a = 1
# b = 2
# args = (3, 4, 5)
# c = 20
```

```python
def add(a, b, *args):
    print('a = ', a)
    print('b = ', b)
    print('args = ', args)
    return a * b + sum(args)

c = add(1, 2)
print('c = ', c)

# Output:
# a = 1
# b = 2
# args = ()
# c = 2
```

```python
def add(a, b, *args):
    print('a = ', a)
    print('b = ', b)
    print('args = ', args)
    return a * b + sum(args)

c = add(1)
print('c = ', c)

# Output:
# TypeError: add() missing 1 required positional argument: 'b'
```

### 1.8. Hàm có tham số đầu vào dạng từ khóa

```python
def add(a, b, **kwargs):
    print('a = ', a)
    print('b = ', b)
    print('kwargs = ', kwargs)
    print(type(kwargs))
    return a + b + sum(kwargs.values())

f = add(1, 2, c=3, d=4, e=5)
print('f = ', f)

# Output:
# a = 1
# b = 2
# kwargs = {'c': 3, 'd': 4, 'e': 5}
# <class 'dict'>
# f = 15
```

```python
def add(a, b, **kwargs):
    print('a = ', a)
    print('b = ', b)
    print('kwargs = ', kwargs)
    print(type(kwargs))
    return a + b + sum(kwargs.values())

f = add(1, 2)
print('f = ', f)

# Output:
# a = 1
# b = 2
# kwargs = {}
# <class 'dict'>
# f = 3
```

```python
def add(a, b, **kwargs):
    print('a = ', a)
    print('b = ', b)
    print('kwargs = ', kwargs)
    print(type(kwargs))
    return a + b + sum(kwargs.values())

f = add(1)
print('f = ', f)

# Output:
# TypeError: add() missing 1 required positional argument: 'b'
```

## 2. Hàm lambda

Hàm lambda là một hàm không tên, được định nghĩa bằng từ khóa `lambda`.
Hàm lambda thường được sử dụng khi cần một hàm đơn giản, chỉ quan tâm nhiều đến đầu vào và đầu ra, mà không cần định nghĩa hàm thông thường dài dòng bằng từ khóa `def`.

```python
add = lambda a, b: a + b

c = add(1, 2)
print('c = ', c)

# Output:
# c = 3
```

```python
add = lambda a, b=1: a + b

c = add(1)
print('c = ', c)

d = add(1, 2)
print('d = ', d)

# Output:
# c = 2
# d = 3
```

```python
add = lambda *args: sum(args)

c = add(1, 2, 3, 4, 5)
print('c = ', c)

d = add(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
print('d = ', d)

# Output:
# c = 15
# d = 55
```

```python
add = lambda a, b, *args: a * b + sum(args)

c = add(1, 2, 3, 4, 5)
print('c = ', c)

# Output:
# c = 20
```

```python
add = lambda a, b, **kwargs: a + b + sum(kwargs.values())

f = add(1, 2, c=3, d=4, e=5)
print('f = ', f)

# Output:
# f = 15
```
