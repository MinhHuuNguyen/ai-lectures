---
time: 01/13/2022
title: Thư viện tính toán khoa học NumPy
description: NumPy là một thư viện Python mạnh mẽ hỗ trợ tính toán khoa học và toán học trên mảng nhiều chiều. NumPy cung cấp một loạt các hàm và phương thức giúp thực hiện các phép toán cơ bản và nâng cao trên dữ liệu mảng nhanh chóng và hiệu quả.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/1-numpy/banner.png
tags: [python]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về thư viện NumPy

NumPy (Numerical Python) là một thư viện mã nguồn mở hỗ trợ tính toán khoa học mạnh mẽ và là nền tảng cho khoa học dữ liệu trong Python.
Thư viện này cung cấp cấu trúc dữ liệu mảng N chiều (ndarray) giúp thực hiện các phép toán trên tập dữ liệu lớn một cách hiệu quả hơn rất nhiều so với dùng danh sách (list) thông thường của Python.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/1-numpy/base_lib.png" style="width: 700px;"/>

Đối với lĩnh vực khoa học dữ liệu và AI, NumPy có vai trò rất quan trọng vì nó giúp thao tác hiệu quả trên mảng dữ liệu lớn và tính toán ma trận nhanh chóng.
Nhiều thư viện phổ biến như Pandas, SciPy, Scikit-Learn, TensorFlow, PyTorch... đều dựa trên hoặc tương thích với NumPy.
Nhờ có NumPy, các nhà khoa học dữ liệu và kỹ sư AI có thể xử lý dữ liệu, tính toán mô hình một cách thuận tiện và nhanh chóng.

#### Tại sao NumPy lại nhanh và hiệu quả?

Thứ nhất, **NumPy sử dụng cấu trúc mảng đồng nhất (homogeneous array)**, tức mọi phần tử trong mảng đều có cùng một kiểu dữ liệu như số nguyên, số thực ...

Điều này giúp mảng NumPy được lưu trữ liên tục trong bộ nhớ, không tản mát như các phần tử trong list của Python.
Nhờ dữ liệu nằm cạnh nhau, việc truy cập và tính toán trên mảng được CPU tối ưu cache tốt hơn, tăng tốc độ xử lý.

Thứ hai, nhiều thao tác của **NumPy được cài đặt ở mức độ thấp bằng ngôn ngữ C (và Fortran)** thay vì từng bước thực hiện bằng Python thuần.

Nói một cách đơn giản, NumPy cho phép vector hóa các phép toán: thay vì phải lặp for qua từng phần tử, ta có thể áp dụng trực tiếp phép tính trên cả mảng.
Các phép toán được "vector hóa" này thường nhanh hơn mã Python thuần hàng chục lần nhờ tận dụng mã máy hiệu suất cao

Thứ ba, do mảng NumPy cố định kiểu dữ liệu và kích thước, **chi phí lưu trữ và quản lý đối tượng thấp hơn list của Python**.

Python list phải lưu thêm thông tin cho từng phần tử (loại, kích thước,...), trong khi NumPy chỉ lưu thông tin
này một lần cho cả mảng.
Kết quả là tiết kiệm bộ nhớ và giảm tải cho bộ thông dịch Python.

Nhờ những ưu điểm trên, NumPy đặc biệt hiệu quả trong tính toán khoa học.
Trong các phép tính số học trên mảng, NumPy có thể nhanh hơn Python thuần từ hàng chục đến hàng trăm lần.
Tuy nhiên, lưu ý rằng lợi ích tốc độ này chỉ đạt được khi ta tận dụng phép toán vector hóa; nếu dùng NumPy nhưng vẫn lặp từng phần tử bằng Python (ví dụ, dùng vòng for duyệt mảng NumPy), hiệu năng có thể chậm hơn do phải chuyển đổi qua lại giữa Python và NumPy.

## 2. Kiểu dữ liệu numpy.ndarray

**ndarray (N-dimensional array)** là lớp đối tượng chính của NumPy, dùng để biểu diễn mảng N chiều các phần tử cùng kiểu.
Một ndarray về bản chất là một vùng nhớ liên tục chứa các phần tử có kích thước cố định, kèm theo một vài thuộc tính quan trọng:

#### shape (hình dạng)

shape là bộ kích thước của mảng theo mỗi chiều.
Ta kiểm tra shape bằng thuộc tính `.shape` của ndarray.

Ví dụ:
    - mảng 1 chiều (vector) có `shape = (n,)` trong đó n là số phần tử trong mảng
    - mảng 2 chiều (ma trận) có `shape = (n, m)` trong đó n là số hàng, m là số cột
    - mảng 3 chiều có `shape = (n, m, k)` trong đó n, m, k là kích thước theo từng chiều
    - mảng N chiều có `shape = (d1, d2, ..., dN)` trong đó di là kích thước theo chiều i


```python
import numpy as np

# Tạo mảng 1D
arr1d = np.array([1, 2, 3, 4])
print(arr1d.shape)  # Output: (4,)

# Tạo mảng 2D
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d.shape)  # Output: (2, 3)

# Tạo mảng 3D
arr3d = np.array([[[1], [2]], [[3], [4]], [[5], [6]]])
print(arr3d.shape)  # Output: (3, 2, 1)
```

#### dtype (kiểu dữ liệu)

dtype cho biết kiểu của các phần tử trong mảng.
Ta kiểm tra dtype bằng thuộc tính `.dtype` của ndarray.

Tất cả phần tử phải cùng kiểu dữ liệu, nếu không NumPy sẽ ép kiểu về loại chung.
Để chủ động ép kiểu dữ liệu, ta có thể chỉ định `dtype` khi tạo mảng bằng hàm `np.array()` hoặc dùng phương thức `.astype()` để chuyển đổi kiểu sau khi tạo mảng.

Một số kiểu dữ liệu phổ biến trong NumPy bao gồm:
- Số nguyên: `int8`, `int16`, `int32`, `int64`
- Số thực: `float16`, `float32`, `float64`
- Số phức: `complex64`, `complex128`
- Boolean: `bool`
- Chuỗi ký tự: `str_`, `unicode_`

```python
import numpy as np

# Tạo mảng với kiểu dữ liệu mặc định (int)
arr_default = np.array([1, 2, 3])
print(arr_default.dtype)  # Output: int64 (hoặc int32 tùy hệ điều hành)

# Tạo mảng với kiểu dữ liệu float
arr_float = np.array([1, 2, 3], dtype=np.float32)
print(arr_float.dtype)  # Output: float32

# Chuyển đổi kiểu dữ liệu sang float64
arr_converted = arr_default.astype(np.float64)
print(arr_converted.dtype)  # Output: float64
```

Một đặc điểm quan trọng của ndarray là tính đồng nhất (homogeneous).
Nếu ta đưa vào các giá trị khác kiểu, NumPy sẽ chuyển tất cả về cùng một kiểu.
Ví dụ: `np.array([1, 2.5, 3])` sẽ tạo mảng kiểu `float` vì phần tử `2.5` là `float`.

#### Các thuộc tính khác của ndarray

- **ndim (số chiều)**: Số chiều (hay bậc) của mảng. Mảng 1D có ndim=1, 2D ndim=2 ...
- **size (kích thước)**: Tổng số phần tử trong mảng (tích của các phần tử trong shape).
- **itemsize (độ lớn mỗi phần tử)**: Số byte mỗi phần tử chiếm (phụ thuộc dtype).

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
print(arr.ndim)      # Output: 2
print(arr.size)      # Output: 6
print(arr.itemsize)  # Output: 4 (vì int32 chiếm 4 bytes)
```

## 3. Thao tác và toán tử cơ bản với numpy.ndarray

### 3.1. Tạo mảng NumPy

Ngoài `np.array()`, NumPy cung cấp nhiều hàm để tạo mảng nhanh cho các trường hợp phổ biến:
- `np.zeros(shape, dtype)`: Tạo mảng với mọi phần tử `= 0` theo shape yêu cầu.
- `np.ones(shape, dtype)`: Tạo mảng với mọi phần tử `= 1`.
- `np.arange(start, stop, step)`: Tương tự range của Python, tạo mảng các số nguyên (hoặc số thực) cách đều.
- `np.linspace(start, stop, num)`: Tạo mảng gồm num giá trị nằm trong khoảng `[start, stop]` chia đều.
- `np.eye(N)` hoặc `np.identity(N)`: Tạo ma trận đơn vị kích thước NxN (các đường chéo `= 1`, còn lại `= 0`).

```python
import numpy as np

# Tạo ma trận đơn vị 3x3
identity_matrix = np.eye(3)
print(identity_matrix)
# Output:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Tạo mảng 1D từ 0 đến 9
arr_range = np.arange(10)
print(arr_range)  # Output: [0 1 2 3 4 5 6 7 8 9]

# Tạo mảng 1D gồm 5 giá trị từ 0 đến 1 chia đều
arr_linspace = np.linspace(0, 1, 5)
print(arr_linspace)  # Output: [0.   0.25 0.5  0.75 1.  ]

# Tạo mảng 2D 3x4 với tất cả phần tử = 0
arr_zeros = np.zeros((3, 4))
print(arr_zeros)
# Output:
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# Tạo mảng 2D 2x3 với tất cả phần tử = 1
arr_ones = np.ones((2, 3))
print(arr_ones)
# Output:
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

### 3.2. Truy cập và cắt lát (slicing) mảng

Cách truy cập phần tử của mảng NumPy tương tự như với list nhưng mạnh mẽ hơn.
Ta sử dụng dấu với chỉ số (index) để lấy giá trị.
Lưu ý: Chỉ số của mảng bắt đầu từ 0 (giống Python list).

- Với mảng 1 chiều: `a[i]` trả về phần tử tại vị trí i. Ví dụ: `a[0]` là phần tử đầu tiên.
- Với mảng nhiều chiều: ta dùng nhiều chỉ số phân cách bởi dấu phẩy. Ví dụ: `b[0, 0]` là phần tử đầu tiên, `b[1, 2]` lấy phần tử hàng 1 cột 2 của ma trận b (tương đương `b[1][2]`).

Ngoài ra, NumPy hỗ trợ cú pháp slicing dùng dấu `:` để lấy ra một đoạn phần tử: `a[start:stop:step]` lấy các phần tử từ vị trí start đến stop-1 với bước nhảy step .
Các tham số start, stop, step có thể bỏ trống để mặc định (bắt đầu từ đầu mảng, hoặc đến cuối mảng, bước = 1).
Với mảng đa chiều, ta có thể slice mỗi chiều độc lập.

```python
import numpy as np
# Tạo mảng 1D
arr1d = np.array([10, 20, 30, 40, 50])
print(arr1d[0])      # Output: 10 (phần tử đầu tiên)
print(arr1d[1:4])    # Output: [20 30 40] (phần tử từ index 1 đến 3)
print(arr1d[::2])    # Output: [10 30 50] (lấy mỗi phần tử thứ 2)
print(arr1d[-1])     # Output: 50 (phần tử cuối cùng)
print(arr1d[-3:])    # Output: [30 40 50] (3 phần tử cuối)

# Tạo mảng 2D
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[0, 0])      # Output: 1 (phần tử hàng 0 cột 0)
print(arr2d[1, :])      # Output: [4 5 6] (toàn bộ hàng 1)
print(arr2d[:, 2])      # Output: [3 6 9] (toàn bộ cột 2)
print(arr2d[0:2, 1:3])  # Output: [[2 3]
                        #          [5 6] (hàng 0-1, cột 1-2)
print(arr2d[::2, ::2])  # Output: [[1 3]
                        #          [7 9] (lấy mỗi hàng và cột thứ 2)
```

Một điểm cần chú ý: mảng con (sub-array) được tạo bằng slicing sẽ dùng chung bộ nhớ với mảng gốc, không tạo bản sao.
Điều này có nghĩa là nếu ta gán giá trị vào mảng con, mảng gốc cũng thay đổi.
Đây là chủ ý thiết kế của NumPy để xử lý dữ liệu lớn hiệu quả (tránh copy không cần thiết).
Nếu muốn copy hẳn mảng con, ta có thể dùng phương thức `.copy()`.

### 3.3. Thay đổi giá trị phần tử

Sau khi truy cập phần tử, ta có thể gán giá trị mới cho nó bằng toán tử gán `=`.

Ngoài ra, ta cũng có thể gán giá trị mới cho một mảng con (sub-array) bằng slicing, tuy nhiên, giá trị gán phải phù hợp với kích thước mảng con.

```python
import numpy as np

# Tạo mảng 1D
arr1d = np.array([10, 20, 30, 40, 50])
arr1d[0] = 100  # Thay đổi phần tử đầu tiên
print(arr1d)    # Output: [100  20  30  40  50]
arr1d[1:4] = [200, 300, 400]  # Thay đổi phần tử từ index 1 đến 3
print(arr1d)    # Output: [100 200 300 400  50]

# Tạo mảng 2D
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[0, 0] = 10  # Thay đổi phần tử hàng 0 cột 0
print(arr2d)
# Output:
# [[10  2  3]
#  [ 4  5  6]
#  [ 7  8  9]]
arr2d[1, :] = [40, 50, 60]  # Thay đổi toàn bộ hàng 1
print(arr2d)
# Output:
# [[10  2  3]
#  [40 50 60]
#  [ 7  8  9]]
arr2d[:, 2] = [30, 60, 90]  # Thay đổi toàn bộ cột 2
print(arr2d)
# Output:
# [[10  2 30]
#  [40 50 60]
#  [ 7  8 90]]
```

### 3.4. Phép toán đại số cơ bản (element-wise)

Các toán tử cộng `+`, trừ `-`, nhân `*`, chia `/`, luỹ thừa `**` khi áp dụng giữa hai mảng cùng kích thước sẽ thực hiện phép toán trên từng cặp phần tử tương ứng.

Ngoài các toán tử, NumPy còn cung cấp các hàm tương ứng như `np.add()`, `np.subtract()`, `np.multiply()`, `np.divide()`, `np.power()` để thực hiện các phép toán này.
Hàm `np.sqrt()` tính căn bậc hai từng phần tử trong mảng, `np.exp()` tính hàm mũ e^x, `np.log()` tính logarit tự nhiên, `np.abs()` lấy giá trị tuyệt đối, 

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)  # Output: [5 7 9]
print(a - b)  # Output: [-3 -3 -3]
print(a * b)  # Output: [ 4 10 18]
print(a / b)  # Output: [0.25 0.4 0.5]
print(a ** b)  # Output: [  1  32 729]

print(np.add(a, b))  # Output: [5 7 9]
print(np.subtract(a, b))  # Output: [-3 -3 -3]
print(np.multiply(a, b))  # Output: [ 4 10 18]
print(np.divide(a, b))  # Output: [0.25 0.4 0.5]
print(np.power(a, b))  # Output: [  1  32 729]
print(np.sqrt(a))  # Output: [1.         1.41421356 1.73205081]
print(np.exp(a))  # Output: [ 2.71828183  7.3890561  20.08553692]
print(np.log(a))  # Output: [0.         0.69314718 1.09861229]
print(np.abs(np.array([-1, -2, 3])))  # Output: [1 2 3]
```

### 3.5. Broadcasting

Broadcasting là tính năng đặc trưng và mạnh mẽ của NumPy, cho phép thực hiện các phép tính trên hai mảng có kích thước khác nhau bằng cách "mở rộng" mảng nhỏ hơn để phù hợp với mảng lớn hơn.

Thay vì phải nhân bản dữ liệu thủ công, NumPy làm điều đó một cách hiệu quả trong bộ nhớ và tính toán.
Nói cách khác, broadcasting giúp tự động điều chỉnh các mảng có shape khác nhau để có shape tương thích khi
tính toán, bằng cách nhân bản (replicate) giá trị dọc theo các chiều cần thiết.

Nhờ đó, ta có thể viết các phép toán phần tử-wise (element-wise) giữa mảng với mảng, mảng với số vô hướng một cách tự nhiên, ngắn gọn mà không cần vòng lặp.


Quy tắc broadcasting:
- Nếu hai mảng có số chiều khác nhau, mảng có ít chiều hơn sẽ được tự động thêm các chiều kích thước 1 ở bên trái.
Ví dụ shape (5,) sẽ coi như (1,5) nếu cần so sánh với (3,5).
- Sau đó, ở mỗi chiều, hai mảng tương thích nếu kích thước bằng nhau hoặc một trong hai bằng 1.
    - Nếu kích thước hai bên bằng nhau, ta giữ nguyên kích thước và thực hiện phép toán.
    - Nếu kích thước một bên là 1, NumPy sẽ coi như nhân bản giá trị ở chiều đó cho bằng kích thước bên kia.
    - Nếu một chiều nào đó không thỏa mãn hai điều kiện trên, NumPy sẽ báo lỗi.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
b = np.array([10, 20, 30])             # shape (3,)
# Bước 1: b được broadcast thành shape (1, 3).
# Từ đó, b được coi như [[10, 20, 30]], b.shape (1, 3)
# Bước 2: a.shape (2, 3) và b.shape (1, 3) tương thích vì chiều 0: 2 và 1 (1 được nhân bản thành 2), chiều 1: 3 và 3 (bằng nhau).
# Bước 3: b được broadcast thành shape (2, 3) bằng cách nhân bản giá trị dọc theo chiều 0.
# Kết quả b được coi như [[10, 20, 30],
#                          [10, 20, 30]]
# Bây giờ a và b cùng shape (2, 3), ta thực hiện phép toán.
print(a + b)  # Output: [[11 22 33]
              #          [14 25 36]]
```

Nhờ broadcasting, NumPy tiết kiệm bộ nhớ và thời gian vì không thực sự nhân bản mảng nhỏ trong RAM mà vẫn đạt được kết quả đúng

## 4. Các hàm thống kê và logic trong NumPy

### 4.1. Hàm thống kê

NumPy cung cấp nhiều hàm thống kê hữu ích để tính toán các đặc trưng của dữ liệu trong mảng:
- `np.sum(arr, axis=None)`: Tính tổng các phần tử trong mảng. Tham số `axis` xác định chiều để tính tổng (mặc định là toàn bộ mảng).
- `np.mean(arr, axis=None)`: Tính giá trị trung bình (mean)
- `np.median(arr, axis=None)`: Tính giá trị trung vị (median)
- `np.std(arr, axis=None)`: Tính độ lệch chuẩn (standard deviation)
- `np.var(arr, axis=None)`: Tính phương sai (variance)
- `np.min(arr, axis=None)`: Tìm giá trị nhỏ nhất
- `np.max(arr, axis=None)`: Tìm giá trị lớn nhất
- `np.argmin(arr, axis=None)`: Trả về chỉ số của phần tử nhỏ nhất
- `np.argmax(arr, axis=None)`: Trả về chỉ số của phần tử lớn nhất

```python
import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(data))          # Output: 21 (tổng tất cả phần tử)
print(np.sum(data, axis=0))  # Output: [5 7 9] (tổng theo cột)
print(np.sum(data, axis=1))  # Output: [ 6 15] (tổng theo hàng)
print(np.mean(data))         # Output: 3.5 (giá trị trung bình)
print(np.median(data))       # Output: 3.5 (giá trị trung vị)
print(np.std(data))          # Output: 1.707825127659933 (độ lệch chuẩn)
print(np.var(data))          # Output: 2.9166666666666665 (phương sai)
print(np.min(data))          # Output: 1 (giá trị nhỏ nhất)
print(np.max(data))          # Output: 6 (giá trị lớn nhất)
print(np.argmin(data))       # Output: 0 (chỉ số phần tử nhỏ nhất)
print(np.argmax(data))       # Output: 5 (chỉ số phần tử lớn nhất)
```

### 4.2. Hàm logic và truy vấn điều kiện

NumPy cũng cung cấp các hàm logic để kiểm tra điều kiện trên mảng:
- `np.all(arr)`: Trả về True nếu tất cả phần tử trong mảng là True (hoặc khác 0).
- `np.any(arr)`: Trả về True nếu có ít nhất một phần tử trong mảng là True (hoặc khác 0).
- `np.where(condition, x, y)`: Trả về mảng mới với phần tử từ `x` nếu điều kiện đúng, từ `y` nếu điều kiện sai.
- `np.nonzero(arr)`: Trả về chỉ số của các phần tử khác 0 trong mảng

```python
import numpy as np

data = np.array([[1, 0, 3], [0, 5, 0]])
print(np.all(data))          # Output: False (không phải tất cả phần tử đều khác 0)
print(np.any(data))          # Output: True (có phần tử khác 0)
print(np.where(data > 2, data, 0))  # Output: [[0 0 3]
                                    #          [0 5 0]] (giữ nguyên phần tử > 2, còn lại = 0)
print(np.nonzero(data))      # Output: (array([0, 0, 1]), array([0, 2, 1])) (chỉ số các phần tử khác 0)
```

### 4.3. Một số thao tác xử lý dữ liệu khác

- **Thay đổi kích thước và hình dạng:** Dùng `reshape()` để tạo một mảng mới với shape khác (mà vẫn dùng chung dữ liệu). Hoặc `flatten()` / `ravel()` để dàn mảng nhiều chiều thành 1 chiều.
- **Ghép và tách mảng:** `np.concatenate((A,B), axis=...)` để ghép hai mảng theo chiều chỉ định. Ngoài ra có `np.vstack`, `np.hstack` lần lượt ghép theo chiều dọc (thêm hàng) và ngang (thêm cột) cho tiện. Để tách mảng, có thể dùng `np.split`.
- **Sao chép mảng:** `arr.copy()` tạo bản sao mới độc lập với `arr`.
- **Sắp xếp:** `np.sort(A)` trả về bản copy đã sắp xếp, hoặc dùng `A.sort()` sắp xếp tại chỗ. Có thể sắp xếp theo axis.
- **Tìm các giá trị unique:** `np.unique(A)` trả về các giá trị duy nhất, thường dùng cho dữ liệu phân loại.
- **Ma trận chéo và tam giác:** `np.diag(v)` tạo ma trận chéo từ vector `v` (hoặc trích xuất đường chéo), `np.triu`, `np.tril` lấy phần tam giác trên/dưới.

```python
import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6]])
print(data.shape)  # Output: (2, 3)

reshaped = data.reshape((3, 2))
print(reshaped.shape)  # Output: (3, 2)

flattened = data.flatten()
print(flattened)  # Output: [1 2 3 4 5 6]

data2 = np.array([[7, 8, 9], [10, 11, 12]])
concatenated = np.concatenate((data, data2), axis=0)
print(concatenated)
# Output:
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

vstacked = np.vstack((data, data2))
print(vstacked)
# Output:
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

hstacked = np.hstack((data, data2))
print(hstacked)
# Output:
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]

split_data = np.split(hstacked, 2, axis=1)
print(split_data[0])
# Output:
# [[1 2 3]
#  [4 5 6]]
print(split_data[1])
# Output:
# [[ 7  8  9]
#  [10 11 12]]

copy_data = data.copy()
copy_data[0, 0] = 100
print(data)      # Output: [[1 2 3]
                 #          [4 5 6]] (data không thay đổi)
print(copy_data) # Output: [[100   2   3]
                 #          [  4   5   6]] (copy_data thay đổi)

sorted_data = np.sort(data, axis=1)
print(sorted_data)
# Output:
# [[1 2 3]
#  [4 5 6]] (đã sắp xếp theo hàng)

unique_values = np.unique(np.array([1, 2, 2, 3, 1, 4]))
print(unique_values)  # Output: [1 2 3 4]

diag_matrix = np.diag([1, 2, 3])
print(diag_matrix)
# Output:
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```
