---
time: 12/19/2021
title: Giới thiệu chung về ngôn ngữ lập trình Python
description: Bài viết này sẽ giới thiệu một số ứng dụng và lý do khiến Python trở nên phổ biến. Ngoài ra, chúng ta cũng sẽ tìm hiểu một số thao tác lập trình cơ bản với Python như import thư viện, comment, đặt tên biến dữ liệu, ...
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/0_syllabus/images/python-logo.png
tags: [python]
is_highlight: false
is_published: true
---

## 1. Tại sao ngôn ngữ lập trình Python lại phổ biến?

```
Python is a programming language that lets you work more quickly and integrate your systems more effectively.
```

Đây là một câu mô tả về Python trên trang chủ của ngôn ngữ lập trình này. Python là một ngôn ngữ lập trình được sử dụng rộng rãi trong thời gian gần đây trong lĩnh vực trí tuệ nhân tạo, khoa học dữ liệu, phân tích dữ liệu, và nhiều lĩnh vực khác. 

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/1-introduction/tiobe_index_2025.png" style="width: 1200px;"/>
Danh sách các ngôn ngữ lập trình phổ biến nhất trên thế giới theo https://www.tiobe.com/tiobe-index/.

Một lý do khiến Python trở nên phổ biến là vì nó đơn giản và gần gũi với ngôn ngữ tự nhiên, tiếng Anh. Điều này khiến cho ai ai cũng có thể học Python mà không cần phải có kiến thức nền tảng chuyên sâu về lập trình.

Ví dụ về sự đơn giản của Python so sánh với một số ngôn ngữ lập trình khác:

- Sử dụng Python:

```python
print("Hello, World!")
```

- Sử dụng Java:

```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

## 2. Các ứng dụng của ngôn ngữ lập trình Python

Với sự bùng nổ của Cách mạng công nghiệp 4.0 với Dữ liệu lớn (Big Data), Trí tuệ nhân tạo (AI), Điện toán đám mây (Cloud Computing), Internet vạn vật (Internet of Things) ..., Python được sử dụng nhiều trong các công việc như:
- **Phát triển web**: Python có các framework mạnh mẽ như Django và Flask giúp phát triển các ứng dụng web một cách nhanh chóng và hiệu quả.
- **Khoa học dữ liệu**: Python có các thư viện như Pandas, NumPy, và Matplotlib giúp phân tích và trực quan hóa dữ liệu.
- **Trí tuệ nhân tạo và học máy**: Các thư viện như TensorFlow, PyTorch, và Scikit-learn hỗ trợ phát triển các mô hình AI và machine learning.
- **Tự động hóa**: Python có thể được sử dụng để viết các script tự động hóa các tác vụ hàng ngày.
- **Phát triển phần mềm**: Python có thể được sử dụng để phát triển các ứng dụng phần mềm từ nhỏ đến lớn.
- **Điện toán đám mây**: Python hỗ trợ phát triển và quản lý các dịch vụ đám mây với các thư viện như Boto3 cho AWS.
- **Internet of Things (IoT)**: Python có thể được sử dụng để lập trình các thiết bị IoT với các thư viện như MicroPython.

## 3. Một số thao tác lập trình cơ bản với Python

### 3.1. Import thư viện trong Python

Các thư viện trong bất kỳ ngôn ngữ lập trình nào đều giúp chúng ta giảm thiểu thời gian và công sức khi viết code. Các thư viện là một tập hợp các hàm và lớp được viết sẵn để giúp chúng ta thực hiện các công việc cụ thể.

Trong Python, chúng ta có thể import các thư viện có sẵn bằng cách sử dụng câu lệnh `import`.

#### 3.1.1. Import các thư viện có sẵn

Các thư viện có sẵn là các thư viện mà nhà phát triển Python cung cấp sẵn cho chúng ta để sử dụng mà không cần phải cài đặt thêm.

```python
import os
import math
import random

print(os.getcwd()) # Output: ./data/
print(math.pi) # Output: 3.141592653589793
print(random.randint(1, 100)) # Output: 42
```

#### 3.1.2 Cài đặt và import các thư viện bên ngoài

Các thư viện bên ngoài là các thư viện mà chúng ta cần phải cài đặt thêm vào môi trường làm việc của mình trước khi sử dụng.

```bash
# Cài đặt thư viện pandas với pip
pip install pandas

# Cài đặt thư viện pandas với conda
conda install anaconda::pandas
```

```python
import pandas
```

Nếu ra import thư viện `pandas` mà không cài đặt thư viện này trước, Python sẽ báo lỗi như sau:

```python
import pandas

# ModuleNotFoundError: No module named 'pandas'
```

#### 3.1.3. Câu lệnh `as`

Câu lệnh `as` được sử dụng để đặt một tên định danh khác cho thư viện mà chúng ta import.

```python
import pandas as pd
import numpy as np
```

Câu lệnh `as` thường được sử dụng để viết code ngắn gọn và dễ đọc hơn. Ngoài ra, nó còn giúp tránh việc trùng tên giữa các thư viện.

#### 3.1.4. Câu lệnh `from`

Câu lệnh `from` được sử dụng để import một hoặc nhiều hàm hoặc lớp cụ thể nào đó từ một thư viện.

```python
from random import randint

print(randint(1, 100)) # Output: 42
```

```python
from math import pi, e

print(pi) # Output: 3.141592653589793
print(e) # Output: 2.718281828459045
```

#### 3.1.5. Import sử dụng `*`

Câu lệnh `*` được sử dụng để import tất cả các hàm và lớp từ một thư viện.

```python
from math import *

print(pi) # Output: 3.141592653589793
print(e) # Output: 2.718281828459045
print(cos(0)) # Output: 1.0
```

#### 3.1.6. Import thư viện từ một file khác

Phần này sẽ được nói chi tiết hơn ở bài viết sau.

### 3.2. Dấu ; trong Python

Trong Python, dấu `;` không bắt buộc phải được sử dụng ở cuối như trong một số ngôn ngữ lập trình khác như Java, C++, hay JavaScript. Dấu `;` được sử dụng để phân tách các câu lệnh trong cùng một dòng code.

```python
print("Hello, World!"); print("Hello, Python!")
```

Tuy nhiên, việc sử dụng dấu `;` không được khuyến khích trong Python vì nó làm cho code trở nên khó đọc và khó bảo trì.

### 3.3. Comment trong Python

Trong bất kỳ ngôn ngữ lập trình nào, comment là một cách để chú thích cho code của mình. Điều này giúp cho người đọc code hiểu rõ hơn về mục đích của code, dễ dàng bảo trì và sửa lỗi cũng như sử dụng phát triển thêm tính năng mới.

Comment không được thực thi khi chúng ta chạy chương trình.

Trong Python, comment được bắt đầu bằng dấu `#`.

```python
# Đây là một comment
print("Hello, World!") # Đây cũng là một comment
```

Để chuyển một đoạn code thành comment, chúng ta có thể sử dụng phím tắt `Ctrl + /` (đối với hệ điều hành Windows hoặc Linux) hoặc `Cmd + /` (đối với hệ điều hành MacOS) trên các trình soạn thảo code như Visual Studio Code, PyCharm, Jupyter Notebook, ...

Ngoài comment, chúng ta còn có docstring. Docstring là một chuỗi dài được viết giữa ba dấu `"""` hoặc `'''`. Docstring thường được sử dụng để giải thích về mục đích của một hàm hoặc một class.

```python
def hello_world():
    """
    Hàm này được sử dụng để in ra
    chuỗi "Hello, World!"
    """
    print("Hello, World!")
```

### 3.4. Biến dữ liệu trong Python

#### 3.4.1. Khai báo và gán giá trị cho biến

Một biến là một vùng nhớ trong bộ nhớ máy tính được sử dụng để lưu trữ dữ liệu.

Trong Python, chúng ta không cần phải khai báo kiểu dữ liệu cho biến trước khi sử dụng mà có thể gán giá trị cho biến ngay lập tức. Kiểu dữ liệu của biến sẽ được xác định dựa trên giá trị mà chúng ta gán cho biến. Để gán giá trị cho biến, chúng ta sử dụng dấu `=`.

```python
# Khai báo và gán giá trị cho biến
name = "Alice"
age = 20
height = 1.75
is_student = True
```

Ta có thể gán giá trị cho nhiều biến cùng một lúc:

```python
name, age, height, is_student = "Alice", 20, 1.75, True
```

Ngoài ra, chúng ta cũng có thể gán giá trị cho nhiều biến cùng một giá trị:

```python
x = y = z = 0

print(x) # Output: 0
print(y) # Output: 0
print(z) # Output: 0
```

#### 3.4.2. Các quy tắc đặt tên biến

Trong Pyton, ta có các quy tắc **BẮT BUỘC** và các quy tắc **KHUYẾN KHÍCH** khi đặt tên biến:

Đối với các quy tắc bắt buộc:

- Tên biến chỉ được phép chứa các ký tự chữ cái, số, và dấu gạch dưới `_`, không được chứa các ký tự đặc biệt như `! @ # $ % ^ & * ( ) { } [ ] ; : ' " < > , . ? / \ | ~ + - = * / ^ ~ ? > < : ; , .`.
- Tên biến chỉ được phép bắt đầu bằng chữ cái hoặc dấu gạch dưới `_`, không được bắt đầu bằng số.
- Tên biến không nên trùng với các từ khóa của Python như `if else for while def class import from as ...`.

Nếu vi phạm các quy tắc trên, Python sẽ báo lỗi khi chúng ta chạy chương trình.

```python
name@ = "Alice" # Tên biến không được chứa ký tự đặc biệt
1name = "Alice" # Tên biến không được bắt đầu bằng số
if = "Alice"

# SyntaxError: invalid syntax
```

```python
name = "Alice"
Name = "Ben"

print(name) # Output: Alice
print(Name) # Output: Ben
```

Đối với các quy tắc khuyến khích:

- Tên biến là case-sensitive, tức là biến `name` và `Name` là hai biến khác nhau.
- Tên biến nên được đặt sao cho dễ hiểu và dễ nhớ.
- Tên biến nên được đặt theo một trong số các quy tắc Camel Case, Snake Case, hoặc Pascal Case.

```python
# Camel Case
firstName = "Alice"
lastName = "Ben"

# Snake Case
first_name = "Alice"
last_name = "Ben"

# Pascal Case
FirstName = "Alice"
LastName = "Ben"
```
