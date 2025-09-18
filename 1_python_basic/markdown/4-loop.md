---
time: 12/30/2021
title: Vòng lặp trong Python
description: "Vòng lặp là một cấu trúc cơ bản trong lập trình giúp chương trình thực hiện một hành động nào đó lặp đi lặp lại. Vòng lặp giúp giảm thiểu việc lặp lại mã nguồn, giúp chương trình trở nên ngắn gọn và dễ đọc. Trong Python, có hai loại vòng lặp cơ bản là vòng lặp `for` và vòng lặp `while`."
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/1-introduction/python_logo.png
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

---

## Luyện tập

<details>
<summary>Câu hỏi trắc nghiệm</summary>

**1. Lệnh nào dưới đây được sử dụng để tạo vòng lặp trong Python?**

**A.** `for`  

**B.** `while`  

**C.** Cả A và B  

**D.** `loop`

**2. Kết quả của đoạn code sau là gì?**

```python
for i in range(3):
    print(i, end=" ")
```

**A.** `0 1 2`

**B.** `1 2 3`

**C.** `0 1 2 3`

**D.** `1 2`

**3. Lệnh nào để dừng vòng lặp sớm trong Python?**

**A.** `continue`

**B.** `stop`

**C.** `break`

**D.** `exit`

**4. Đoạn mã sau in ra gì?**

```python
x = 0
while x < 3:
    x += 1
print(x)
```

**A.** `0`

**B.** `1`

**C.** `3`

**D.** `2`

**5. Kết quả của đoạn mã sau là gì?**

```python
for i in range(2, 5):
    print(i, end=", ")
```

**A.** `2, 3, 4, `

**B.** `2, 3, 4, 5,`

**C.** `2 3 4`

**D.** `2, 3, 4`

**6. Lệnh `continue` trong vòng lặp có tác dụng gì?**

**A.** Dừng vòng lặp

**B.** Bỏ qua phần còn lại của vòng lặp hiện tại và tiếp tục vòng tiếp theo

**C.** Kết thúc chương trình

**D.** Không có tác dụng

**7. Câu lệnh `for i in range(5, 0, -1)` có ý nghĩa gì?**

**A.** Lặp từ 0 đến 5

**B.** Lặp từ 5 đến 1 giảm dần

**C.** Lặp từ 5 đến 0 giảm dần

**D.** Lặp vô hạn

**8. Vòng lặp `while` nào dưới đây là vòng lặp vô hạn?**

**A.** `while i < 10:`

**B.** `while True:`

**C.** `while False:`

**D.** `while i == 10:`

**9. Lệnh `else` trong vòng lặp `while` được thực thi khi nào?**

**A.** Khi vòng lặp kết thúc bình thường

**B.** Khi vòng lặp bị `break`

**C.** Khi vòng lặp không có điều kiện

**D.** Khi gặp `continue`

**10. Kết quả của đoạn mã sau là gì?**

```python
for i in range(3):
    if i == 1:
        break
    print(i)
```

**A.** `0 1 2`

**B.** `0`

**C.** `1`

**D.** Không in gì cả

**11. Điều gì xảy ra khi `range(10, 0, -2)` được sử dụng trong vòng lặp `for`?**

**A.** Lặp từ 10 đến 1 với bước nhảy -2

**B.** Lặp từ 10 đến 0 với bước nhảy -2

**C.** Lặp từ 10 đến -2 với bước nhảy -2

**D.** Lặp từ 0 đến 10 với bước nhảy 2

**12. Trong Python, vòng lặp `while` thực thi khi nào?**

**A.** Khi điều kiện là False

**B.** Khi điều kiện là True

**C.** Khi điều kiện không xác định

**D.** Chỉ khi có biến đếm

**13. Kết quả của đoạn mã sau là gì?**

```python
for i in range(1, 6, 2):
    print(i, end=" ")
```

**A.** `1 2 3 4 5`

**B.** `1 3 5`

**C.** `1 2 3 5`

**D.** `1 4 5`

**14. Vòng lặp nào phù hợp nhất để lặp qua danh sách `my_list = [10, 20, 30]`?**

**A.** `for i in my_list:`

**B.** `while i in my_list:`

**C.** `for i range(my_list):`

**D.** `loop my_list:`

**15. Cách nào tốt nhất để lặp qua các chỉ mục của danh sách` my_list = [10, 20, 30]`?**

**A.** `for i in my_list:`

**B.** `for i in range(len(my_list)):`

**C.** `while i < len(my_list):`

**D.** `for i, val in my_list:`

</details>

<details>
<summary>Đáp án</summary>

**1. C.** Cả A và B

**2. A.** `0 1 2`

**3. C.** `break`

**4. C.** `3`

**5. A.** `2, 3, 4, `

**6. B.** Bỏ qua phần còn lại của vòng lặp hiện tại và tiếp tục vòng tiếp theo

**7. B.** Lặp từ 5 đến 1

**8. B.** `while True:`

**9. A.** Khi vòng lặp kết thúc bình thường

**10. B.** `0`

**11. A.** Lặp từ 10 đến 1 với bước nhảy -2

**12. B.** Khi điều kiện là `True`

**13. B.** `1 3 5`

**14. A.** `for i in my_list:`

**15. B.** `for i in range(len(my_list)):`

</details>

<details>
<summary>Bài tập thực hành</summary>

**1. In dãy số từ 1 đến 10:** Viết chương trình in ra các số từ 1 đến 10 bằng vòng lặp `for` và `while`.

**2. Tính tổng các số từ 1 đến n:** Nhập vào một số nguyên dương `n`, tính tổng các số từ `1` đến `n`.

**3. In các số chẵn từ 1 đến 20:** Viết chương trình in ra tất cả các số chẵn từ `1` đến `20`.

**4. In bảng cửu chương:** Viết chương trình in bảng cửu chương từ `2` đến `9`.

**5. Đảo ngược số nguyên:** Nhập vào một số nguyên dương, đảo ngược số đó.

**6. Tính giai thừa của một số:** Viết chương trình tính giai thừa của một số nguyên dương n. (Giai thừa n! = n * (n-1) * ... * 1.)

**7. Kiểm tra số Palindrome:** Nhập vào một số nguyên dương, kiểm tra xem số đó có phải là số Palindrome không (số Palindrome là số đối xứng, tức là số đọc từ trái sang phải và từ phải sang trái đều giống nhau).

**8. Tính tổng chữ số của một số nguyên:** Viết chương trình nhập một số nguyên dương và tính tổng các chữ số của số đó.

**9. In tam giác số:** Nhập vào số nguyên `n`, in ra tam giác số có `n` dòng.

**10. Tạo danh sách số Fibonacci:** Nhập vào số nguyên `n`, in ra dãy Fibonacci có `n` phần tử.

</details>

<details>
<summary>Lời giải</summary>

1.

```python
# Sử dụng for loop
print('Sử dụng for loop:')
for i in range(1, 11):
    print(i, end=' ')

# Sử dụng while loop
n = 1
print('Sử dụng while loop:')
while n <= 10:
    print(n, end=' ')
    n += 1

# Output
# Sử dụng for loop:
# 1 2 3 4 5 6 7 8 9 10
# Sử dụng while loop:
# 1 2 3 4 5 6 7 8 9 10
```

2.

```python
n = int(input("Nhập số nguyên dương n: "))
sum_n = sum(range(1, n + 1))
print("Tổng từ 1 đến", n, "là:", sum_n)

# Output
# Nhập số nguyên dương n: 5
# Tổng từ 1 đến 5 là: 15
```

3.

```python
for i in range(2, 21, 2):
    print(i, end=' ')

# Output
# 2 4 6 8 10 12 14 16 18 20
```

4.

```python
for i in range(2, 10):
    print(f"Bảng cửu chương {i}:")
    for j in range(1, 11):
        print(f"{i} x {j} = {i*j}")
    print()

# Output
# Bảng cửu chương 2:
# 2 x 1 = 2
# 2 x 2 = 4
# 2 x 3 = 6
# 2 x 4 = 8
# 2 x 5 = 10
# 2 x 6 = 12
# ...
```

5.

```python
n = int(input("Nhập số nguyên dương: "))
reversed_n = 0
while n > 0:
    reversed_n = reversed_n * 10 + n % 10
    n //= 10
print("Số sau khi đảo ngược là:", reversed_n)

# Output
# Nhập số nguyên dương: 12345
# Số sau khi đảo ngược là: 54321
```

6.

```python
num = int(input("Nhập một số nguyên dương: "))
factorial = 1

if num < 0:
    print("Không thể tính giai thừa của số âm.")
elif num == 0:
    print("Giai thừa của 0 là 1.")
else:
    for i in range(1, num + 1):
        factorial *= i
    print("Giai thừa của", num, "là", factorial)

# Output
# Nhập một số nguyên dương: 5
# Giai thừa của 5 là 120
```

7.

```python
n = input("Nhập số nguyên: ")
if n == n[::-1]:
    print("Là số Palindrome")
else:
    print("Không phải số Palindrome")

# Output
# Nhập số nguyên: 12321
# Là số Palindrome
```

8.

```python
num = int(input("Nhập một số nguyên dương: "))
total = 0
while num > 0:
    total += num % 10
    num //= 10

print("Tổng các chữ số của số nguyên là:", total)

# Output
# Nhập một số nguyên dương: 12345
# Tổng các chữ số của số nguyên là: 15
```

9.

```python
n = int(input("Nhập số dòng của tam giác: "))
for i in range(1, n + 1):
    print(" ".join(str(x) for x in range(1, i + 1)))

# Output
# Nhập số dòng của tam giác: 5
# 1
# 1 2
# 1 2 3
# 1 2 3 4
# 1 2 3 4 5
```

10.

```python
n = int(input("Nhập số phần tử Fibonacci: "))
fibo = [0, 1]
for i in range(2, n):
    fibo.append(fibo[-1] + fibo[-2])
print("Dãy Fibonacci:", fibo[:n])

# Output
# Nhập số phần tử Fibonacci: 10
# Dãy Fibonacci: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```
</details>
