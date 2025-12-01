---
time: 01/06/2022
title: Hàm trong Python
description: Hàm là một khối mã lệnh độc lập, thực hiện một công việc cụ thể và có thể được gọi ở bất kỳ đâu trong chương trình. Hàm giúp chương trình trở nên ngắn gọn, dễ đọc và dễ bảo trì, từ đó, quá trình phát triển phần mềm trở nên hiệu quả hơn. Trong Python, hàm được định nghĩa thông qua từ khóa `def` hoặc `lambda`.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/1-introduction/python_logo.jpeg
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

---

## Luyện tập

<details>
<summary>Câu hỏi trắc nghiệm</summary>

**1. Hàm `def` trong Python dùng để làm gì?**

**A.** Định nghĩa một biến

**B.** Định nghĩa một lớp

**C.** Định nghĩa một hàm

**D.** Định nghĩa một module  

**2. Hàm `lambda` trong Python còn được gọi là gì?**

**A.** Hàm vô danh

**B.** Hàm tĩnh

**C.** Hàm toàn cục

**D.** Hàm có tham số động  

**3. Hàm `lambda` có thể có bao nhiêu biểu thức?**

**A.** Không giới hạn

**B.** Chỉ một biểu thức

**C.** Tối đa ba biểu thức

**D.** Không có biểu thức nào  

**4. Khi nào nên sử dụng `lambda` thay vì `def`?**

**A.** Khi cần định nghĩa một hàm nhanh, ngắn gọn

**B.** Khi cần tạo một hàm phức tạp với nhiều dòng code

**C.** Khi cần một hàm có nhiều câu lệnh

**D.** Khi cần sử dụng vòng lặp trong hàm  

**5. Kết quả của đoạn code sau là gì?**

```python
add = lambda x, y: x + y
print(add(3, 5))
```

**A.** `3`

**B.** `5`

**C.** `8`

**D.** Lỗi

**6. Điều nào là đúng về hàm `lambda`?**

**A.** Có thể chứa nhiều câu lệnh

**B.** Luôn trả về `None`

**C.** Có thể có nhiều tham số nhưng chỉ một biểu thức

**D.** Không thể gán cho biến

**7. Hàm `def` có thể trả về giá trị không?**

**A.** Có, khi sử dụng từ khóa `return`

**B.** Không, hàm def không thể trả về giá trị

**C.** Chỉ trả về giá trị nếu gọi lại chính nó

**D.** Luôn trả về `None`

**8. Kết quả của đoạn code sau là gì?**

```python
def func():
    return lambda x: x * 2

double = func()
print(double(4))
```

**A.** `8`

**B.** `4`

**C.** `Lỗi`

**D.** `None`

**9. Câu nào đúng về hàm `lambda`?**

**A.** Không thể truyền làm đối số cho các hàm khác

**B.** Không thể trả về một giá trị

**C.** Có thể sử dụng trong `map()`, `filter()`, `sorted()`

**D.** Không thể được gán vào một biến

**10. Kết quả của đoạn code sau là gì?**

```python
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))
print(squared)
```

**A.** `[1, 2, 3, 4]`

**B.** `[2, 4, 6, 8]`

**C.** `[1, 4, 9, 16]`

**D.** Lỗi

**11. Hàm `lambda` có thể có bao nhiêu tham số?**

**A.** Chỉ một tham số

**B.** Chỉ hai tham số

**C.** Không giới hạn số lượng tham số

**D.** Không có tham số nào

**12. Điều nào là đúng về sự khác biệt giữa `lambda` và `def`?**

**A.** Hàm `lambda` có thể chứa nhiều dòng lệnh hơn `def`

**B.** Hàm `def` có thể chứa nhiều dòng lệnh hơn `lambda`

**C.** Hàm `lambda` có thể có nhiều câu lệnh và vòng lặp

**D.** Không có sự khác biệt nào

**13. Cách nào đúng để viết hàm `lambda` tính tổng hai số?**

**A.** `lambda x y: x + y`

**B.** `lambda (x, y): x + y`

**C.** `lambda x, y: x + y`

**D.** `lambda: x + y`

**14. Kết quả của đoạn code sau là gì?**

```python
func = lambda x: x if x > 0 else -x
print(func(-10))
```

**A.** `-10`

**B.** `10`

**C.** `0`

**D.** Lỗi

**15. Điều nào sau đây là sai về hàm `lambda`?**

**A.** Có thể trả về giá trị

**B.** Có thể có nhiều dòng lệnh

**C.** Có thể có nhiều tham số

**D.** Có thể dùng với `map()` và `filter()`

</details>

<details>
<summary>Đáp án</summary>

**1. C.** Định nghĩa một hàm

**2. A.** Hàm vô danh

**3. B.** Chỉ một biểu thức

**4. A.** Khi cần định nghĩa một hàm nhanh, ngắn gọn

**5. C.** 8

**6. C.** Có thể có nhiều tham số nhưng chỉ một biểu thức

**7. A.** Có, khi sử dụng từ khóa `return`

**8. A.** 8

**9. C.** Có thể sử dụng trong `map()`, `filter()`, `sorted()`

**10. C.** [1, 4, 9, 16]

**11. C.** Không giới hạn số lượng tham số

**12. B.** Hàm def có thể chứa nhiều dòng lệnh hơn lambda

**13. C.** `lambda x, y: x + y`

**14. B.** 10

**15. B.** Có thể có nhiều dòng lệnh

</details>
