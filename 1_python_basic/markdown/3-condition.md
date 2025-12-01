---
time: 12/26/2021
title: Câu lệnh điều kiện trong Python
description: Câu lệnh điều kiện là một trong những cấu trúc cơ bản nhất trong lập trình giúp chương trình thực hiện các hành động khác nhau dựa trên điều kiện đầu vào. Trong Python, câu lệnh điều kiện được thực hiện thông qua các từ khóa `if`, `elif` và `else`. Bên cạnh đó, câu lệnh `pass` và cấu trúc một dòng (inline structure) cũng là những phần quan trọng cần nắm vững khi làm việc với câu lệnh điều kiện trong Python.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/1-introduction/python_logo.jpeg
tags: [python]
is_highlight: false
is_published: true
---

## 1. Câu lệnh `if`

Câu lệnh `if` giúp chương trình thực hiện một hành động nếu điều kiện đầu vào là đúng, nếu điều kiện không đúng thì chương trình sẽ bỏ qua hành động đó.

```python
# do something

if condition:
    # do something if condition is True

# do something
```

Ví dụ:

```python
x = 10
print("x =", x)
if x > 5:
    print("x is greater than 5")
print("End of program")

# Output:
# x = 10
# x is greater than 5
# End of program
```

```python
x = 3
print("x =", x)
if x > 5:
    print("x is greater than 5")
print("End of program")

# Output:
# x = 3
# End of program
```

## 2. Câu lệnh `if-else`

Câu lệnh `if-else` giúp chương trình thực hiện một hành động nếu điều kiện đầu vào là đúng, nếu điều kiện không đúng thì chương trình sẽ thực hiện một hành động khác.

```python
# do something

if condition:
    # do something if condition is True
else:
    # do something if condition is False

# do something
```

Ví dụ:

```python
x = 10
print("x =", x)
if x > 5:
    print("x is greater than 5")
else:
    print("x is less than or equal to 5")
print("End of program")

# Output:
# x = 10
# x is greater than 5
# End of program
```

```python
x = 3
print("x =", x)
if x > 5:
    print("x is greater than 5")
else:
    print("x is less than or equal to 5")
print("End of program")

# Output:
# x = 3
# x is less than or equal to 5
# End of program
```

## 3. Câu lệnh `if-elif-else`

Câu lệnh `if-elif-else` giúp chương trình thực hiện một hành động nếu điều kiện đầu vào là đúng, nếu điều kiện không đúng thì chương trình sẽ kiểm tra điều kiện tiếp theo.

Nếu không có điều kiện nào đúng thì chương trình sẽ thực hiện hành động mặc định.

Số lượng điều kiện `elif` có thể là bất kỳ, không giới hạn.


```python
# do something

if condition1:
    # do something if condition1 is True
elif condition2:
    # do something if condition2 is True
elif condition3:
    # do something if condition3 is True
else:
    # do something if all conditions are False

# do something
```

Ví dụ:

```python
x = 7
print("x =", x)
if x > 5:
    print("x is greater than 5")
elif x > 8:
    print("x is greater than 8")
else:
    print("x is less than or equal to 5")
print("End of program")

# Output:
# x = 7
# x is greater than 5
# End of program
```

Trong trường hợp nếu có nhiều điều kiện đúng, chương trình sẽ thực hiện hành động của điều kiện đầu tiên mà đúng.

```python
x = 10
print("x =", x)
if x > 5:
    print("x is greater than 5")
elif x > 8:
    print("x is greater than 8")
else:
    print("x is less than or equal to 5")
print("End of program")

# Output:
# x = 10
# x is greater than 5
# End of program
```

## 4. Câu lệnh `pass`

Câu lệnh `pass` được sử dụng khi chúng ta muốn bỏ qua một điều kiện mà không cần thực hiện hành động nào.

```python

if condition:
    pass
else:
    # do something

```

Ví dụ:

```python
x = 3
print("x =", x)
if x > 5:
    pass
else:
    print("x is less than or equal to 5")
print("End of program")

# Output:
# x = 3
# x is less than or equal to 5
# End of program
```

```python
x = 10
print("x =", x)
if x > 5:
    pass
else:
    print("x is less than or equal to 5")
print("End of program")

# Output:
# x = 10
# End of program
```

## 5. Câu lệnh `if` lồng nhau

Chúng ta có thể lồng nhau nhiều câu lệnh `if` hoặc `elif` trong nhau.

```python
if condition1:
    if condition2:
        # do something
    else:
        # do something
else:
    # do something
```

Ví dụ:

```python
x = 10
print("x =", x)
if x > 5:
    if x > 8:
        print("x is greater than 8")
    else:
        print("x is greater than 5 but less than or equal to 8")
else:
    print("x is less than or equal to 5")
print("End of program")

# Output:
# x = 10
# x is greater than 8
# End of program
```

```python
x = 6
print("x =", x)
if x > 5:
    if x > 8:
        print("x is greater than 8")
    else:
        print("x is greater than 5 but less than or equal to 8")
else:
    print("x is less than or equal to 5")
print("End of program")

# Output:
# x = 6
# x is greater than 5 but less than or equal to 8
# End of program
```

## 6. Cấu trúc một dòng - Inline structure

Trong Python, chúng ta có thể viết câu lệnh điều kiện trên một dòng bằng cách sử dụng cấu trúc `if-else` trên một dòng.

```python
# do something if condition is True else do something if condition is False
```

Ví dụ:

```python
x = 10
print("x is greater than 5") if x > 5 else print("x is less than or equal to 5")

# Output:
# x is greater than 5
```

```python
x = 3
print("x is greater than 5") if x > 5 else print("x is less than or equal to 5")

# Output:
# x is less than or equal to 5
```

---

## Luyện tập

<details>
<summary>Câu hỏi trắc nghiệm</summary>

**1. Câu lệnh điều kiện nào sau đây là hợp lệ trong Python?**

**A.** `if x = 5:`

**B.** `if (x == 5)`

**C.** `if x == 5:`

**D.** `if x === 5:`

**2. Kết quả của đoạn mã sau là gì?**

```python
x = 10
if x > 5:
    print("Lớn hơn 5")
else:
    print("Không lớn hơn 5")
```

**A.** Lớn hơn 5

**B.** Không lớn hơn 5

**C.** Lỗi cú pháp

**D.** Không có gì hiển thị

**3. Câu lệnh nào dùng để kiểm tra nhiều điều kiện trong Python?**

**A.** `else if`

**B.** `elif`

**C.** `elseif`

**D.** `ifelse`

**4. Câu lệnh điều kiện nào là đúng để kiểm tra nếu biến x nằm trong khoảng từ 10 đến 20 (bao gồm cả 10 và 20)?**

**A.** `if x >= 10 or x <= 20:`

**B.** `if x >= 10 and x <= 20:`

**C.** `if 10 >= x <= 20:`

**D.** `if x => 10 and x =< 20:`

**5. Trong Python, cú pháp nào cho phép viết câu lệnh `if` trên một dòng?**

**A.** `if x > 5 print("Lớn hơn 5")`

**B.** `if x > 5: print("Lớn hơn 5")`

**C.** `if (x > 5) { print("Lớn hơn 5") }`

**D.** `if x > 5 then print("Lớn hơn 5")`

**6. Điều gì xảy ra nếu không có `else` trong câu lệnh `if`?**

**A.** Lỗi chương trình

**B.** Chương trình vẫn chạy bình thường

**C.** Python sẽ tự động thêm `else`

**D.** Không có đáp án nào đúng

**7. Kết quả của đoạn mã sau là gì?**

```python
x = 7
y = 10
if x > 5 and y < 15:
    print("Điều kiện đúng")
else:
    print("Điều kiện sai")
```

**A.** Điều kiện đúng

**B.** Điều kiện sai

**C.** Lỗi cú pháp

**D.** Không có gì hiển thị

**8. Lệnh `pass` trong câu lệnh `if` có tác dụng gì?**

**A.** Dùng để bỏ qua điều kiện

**B.** Gây lỗi chương trình

**C.** Giúp tránh lỗi cú pháp khi không có thân `if`

**D.** Không có tác dụng gì

**9. Đoạn mã sau có lỗi không?**

```python
x = 5
if x > 3:
print("Lớn hơn 3")
```

**A.** Có lỗi vì thiếu dấu `:`

**B.** Có lỗi vì thụt lề sai

**C.** Không có lỗi

**D.** Có lỗi vì không có `else`

**10. Câu lệnh nào sau đây là hợp lệ trong Python?**

**A.** `if x > 5: print("Lớn hơn 5")`

**B.** `if x > 5: { print("Lớn hơn 5") }`

**C.** `if x > 5: { print("Lớn hơn 5")`

**D.** `if x > 5: print("Lớn hơn 5") }`

</details>

<details>
<summary>Đáp án</summary>

**1. C.** `if x == 5:`

**2. A.** Lớn hơn 5

**3. B.** `elif`

**4. B.** `if x >= 10 and x <= 20:`

**5. B.** `if x > 5: print("Lớn hơn 5")`

**6. B.** Chương trình vẫn chạy bình thường

**7. A.** Điều kiện đúng

**8. A.** Giúp tránh lỗi cú pháp khi không có thân `if`

**9. B.** Có lỗi vì thụt lề sai

**10. A.** `if x > 5: print("Lớn hơn 5")`

</details>
