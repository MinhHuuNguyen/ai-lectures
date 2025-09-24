---
time: 01/17/2022
title: Thư viện xử lý dữ liệu dạng bảng Pandas
description: Pandas là một thư viện mã nguồn mở cung cấp cấu trúc dữ liệu và công cụ xử lý dữ liệu mạnh mẽ, dễ sử dụng. Pandas hỗ trợ đọc, ghi, xử lý và phân tích dữ liệu dạng bảng nhanh chóng và hiệu quả.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/2-pandas/banner.png
tags: [python, data-analysis]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về thư viện Pandas

Pandas là một thư viện mã nguồn mở của Python, được thiết kế đặc biệt để xử lý và phân tích dữ liệu.
Với Pandas, bạn có thể dễ dàng làm việc với dữ liệu dạng bảng (như bảng tính Excel hoặc bảng trong cơ sở dữ liệu), bao gồm các thao tác như lọc, nhóm (grouping), tính toán thống kê, và thậm chí trực quan hóa dữ liệu.
Pandas được xây dựng trên nền tảng NumPy nên có tốc độ nhanh và hiệu quả cao trong việc thao tác dữ liệu.

Trong lĩnh vực khoa học dữ liệu, Pandas là công cụ then chốt để thực hiện xử lý, phân tích dữ liệu thô thành thông tin hữu ích.
Chẳng hạn, trước khi xây dựng mô hình học máy (machine learning) hoặc trí tuệ nhân tạo, bạn thường dùng Pandas để khám phá dữ liệu (tính các thống kê mô tả, vẽ biểu đồ), tiền xử lý (lọc những dữ liệu cần thiết, xử lý dữ liệu thiếu, chuyển đổi định dạng), và biến đổi đặc trưng (tạo thêm cột đặc trưng, mã hóa giá trị phân loại, v.v.) sao cho phù hợp với thuật toán học máy.
Nhờ có Pandas, quá trình chuẩn bị dữ liệu (data preprocessing) cho các mô hình AI/ML trở nên thuận tiện và hiệu quả hơn.

Để sử dụng Pandas, trước hết ta cần cài đặt thư viện này.
Nếu chưa cài, ta có thể dùng pip:

```bash
pip install pandas
```

hoặc nếu dùng Anaconda:

```bash
conda install pandas
```

Sau khi cài đặt, ta có thể import Pandas trong mã Python như sau:

```python
import pandas as pd
```

## 2. Kiểu dữ liệu pandas.Series và pandas.DataFrame

Hai cấu trúc dữ liệu chính trong Pandas là:

- **Series:**
    - Cấu trúc dữ liệu 1 chiều, tương tự như một mảng một chiều của NumPy hoặc một cột trong bảng dữ liệu.
    - Series là từng cột đơn lẻ của DataFrame (hoặc cũng có thể tồn tại độc lập như một mảng có nhãn).
- **DataFrame:**
    - Cấu trúc dữ liệu 2 chiều dạng bảng với các hàng (dòng) và cột, tương tự như bảng trong cơ sở dữ liệu hoặc trang tính Excel.
    - DataFrame cho phép các cột có kiểu dữ liệu khác nhau (số, chuỗi, logic, v.v.), kích thước bảng có thể thay đổi linh hoạt (có thể thêm hoặc xoá cột), và các trục (hàng, cột) đều được gán nhãn để truy cập dễ dàng.
    - Bạn có thể thực hiện nhiều phép toán nhanh trên toàn bộ hàng hoặc cột của DataFrame.

#### Khởi tạo pandas.Series

Ta có thể tạo Series từ nhiều loại dữ liệu khác nhau như danh sách (list), mảng NumPy (ndarray), từ điển (dict) hoặc thậm chí từ một giá trị đơn (scalar).

Ví dụ tạo Series từ danh sách:

```python
import pandas as pd

data = [10, 20, 30, 40]
s = pd.Series(data)
print(s)

# Output:
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64
```

Ví dụ tạo Series từ từ điển:

```python
import pandas as pd

data = {'a': 10, 'b': 20, 'c': 30}
s = pd.Series(data)
print(s)

# Output:
# a    10
# b    20
# c    30
# dtype: int64
```

Ví dụ tạo Series từ một giá trị đơn:

```python
import pandas as pd

data = 5
s = pd.Series(data, index=['a', 'b', 'c'])
print(s)

# Output:
# a    5
# b    5
# c    5
# dtype: int64
```

#### Khởi tạo pandas.DataFrame

Ta cũng có thể khởi tạo pandas.DataFrame từ dictionary (dict) các danh sách (hoặc mảng NumPy), từ danh sách các danh sách (hoặc mảng NumPy 2D), từ danh sách các dict ...

Ví dụ tạo DataFrame từ dictionary các danh sách hoặc mảng NumPy:

```python
import pandas as pd
import numpy as np

data = {
    'A': [1, 2, 3],
    'B': np.array([4, 5, 6]),
    'C': ['a', 'b', 'c']
}

df = pd.DataFrame(data)
print(df)

# Output:
#    A  B  C
# 0  1  4  a
# 1  2  5  b
# 2  3  6  c
```

Ví dụ tạo DataFrame từ danh sách các danh sách (hoặc mảng NumPy 2D):

```python
import pandas as pd
import numpy as np

data = [
    [1, 4, 'a'],
    [2, 5, 'b'],
    np.array([3, 6, 'c'])
]
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print(df)

# Output:
#    A  B  C
# 0  1  4  a
# 1  2  5  b
# 2  3  6  c
```

Ví dụ tạo DataFrame từ danh sách các dict:

```python
import pandas as pd

data = [
    {'A': 1, 'B': 4, 'C': 'a'},
    {'A': 2, 'B': 5, 'C': 'b'},
    {'A': 3, 'B': 6, 'C': 'c'}
]
df = pd.DataFrame(data)
print(df)

# Output:
#    A  B  C
# 0  1  4  a
# 1  2  5  b
# 2  3  6  c
```

## 3. Thao tác với pandas.Series và pandas.DataFrame

### Khai phá dữ liệu cơ bản

Ta có thể sử dụng các phương thức và thuộc tính của Series và DataFrame để khai phá dữ liệu cơ bản như: kiểm tra kích thước, kiểu dữ liệu, thống kê mô tả, v.v.

```python
import pandas as pd

data = {
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': ['a', 'b', 'c', 'd']
}
df = pd.DataFrame(data)
print(df.shape)          # Kích thước DataFrame (số hàng, số cột)
# Output: (4, 3)
print(df.dtypes)        # Kiểu dữ liệu của từng cột
# Output:
# A     int64
# B     int64
# C    object
# dtype: object
print(df.info())        # Thông tin tổng quan về DataFrame
# Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4 entries, 0 to 3
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   A       4 non-null      int64
#  1   B       4 non-null      int64
#  2   C       4 non-null      object
# dtypes: int64(2), object(1)
# memory usage: 224.0+ bytes
print(df.describe())    # Thống kê mô tả cho các cột số
# Output:
#               A         B
# count  4.000000  4.000000
# mean   2.500000  6.500000
# std    1.290994  1.290994
# min    1.000000  5.000000
# 25%    1.750000  5.750000
# 50%    2.500000  6.500000
# 75%    3.250000  7.250000
# max    4.000000  8.000000
```

Đối với các DataFrame có nhiều bản ghi, ta có thể xem trước một vài dòng đầu hoặc cuối của bảng dữ liệu để có cái nhìn tổng quan về dữ liệu bằng cách sử dụng các phương thức `df.head()` và `df.tail()`:


### Truy cập dữ liệu trong Series và DataFrame

Ta truy cập các phần tử của Series tương tự như với mảng NumPy hoặc danh sách Python, thông qua index.
- Theo vị trí (giống list): Sử dụng cú pháp s[i] (với i là vị trí nguyên, bắt đầu từ 0).
- Theo nhãn: Sử dụng cú pháp s[label] (với label là nhãn của phần tử).

```python
import pandas as pd

data = [10, 20, 30, 40]
s = pd.Series(data, index=['a', 'b', 'c', 'd'])

print(s[0])      # Truy cập theo vị trí
# Output: 10

print(s['b'])    # Truy cập theo nhãn
# Output: 20
```

Tương tự, ta có thể truy cập các phần tử trong DataFrame bằng cách sử dụng nhãn cột và nhãn hàng.
- Truy cập cột: Sử dụng cú pháp df['column_name'] hoặc df.column_name.
- Truy cập hàng: Sử dụng phương thức df.loc[label] (theo nhãn) hoặc df.iloc[i] (theo vị trí).

```python
import pandas as pd

data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': ['a', 'b', 'c']
}
df = pd.DataFrame(data)

print(df['A'])        # Truy cập cột A
# Output:
# 0    1
# 1    2
# 2    3
# Name: A, dtype: int64

print(df.loc[1])      # Truy cập hàng có nhãn 1
# Output:
# A    2
# B    5
# C    b
# Name: 1, dtype: object

print(df.iloc[0])     # Truy cập hàng ở vị trí 0
# Output:
# A    1
# B    4
# C    a
# Name: 0, dtype: object
```

Ta cũng có thể truy cập một phần tử cụ thể trong DataFrame bằng cách kết hợp cả nhãn cột và nhãn hàng:

```python
import pandas as pd

data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': ['a', 'b', 'c']
}
df = pd.DataFrame(data)

print(df.at[1, 'B'])   # Truy cập phần tử ở hàng nhãn 1 và cột 'B'
# Output: 5

print(df.iat[0, 2])    # Truy cập phần tử ở vị trí hàng 0 và cột 2
# Output: 'a'
```

Ta cũng có thể truy cập nhiều hàng hoặc cột cùng lúc bằng cách sử dụng slicing hoặc danh sách nhãn:

```python
import pandas as pd

data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': ['a', 'b', 'c']
}
df = pd.DataFrame(data)

print(df[['A', 'C']])   # Truy cập nhiều cột
# Output:
#    A  C
# 0  1  a
# 1  2  b
# 2  3  c

print(df.loc[0:1])      # Truy cập nhiều hàng (hàng 0 và 1)
# Output:
#    A  B  C
# 0  1  4  a
# 1  2  5  b
```

### Truy cập dữ liệu với điều kiện

Ta có thể lọc dữ liệu trong Series hoặc DataFrame dựa trên điều kiện cụ thể.
Ví dụ lọc các giá trị trong Series lớn hơn 20:

```python
import pandas as pd

data = [10, 20, 30, 40]
s = pd.Series(data)

filtered_s = s[s > 20]
print(filtered_s)
# Output:
# 2    30
# 3    40
# dtype: int64
```

Ví dụ lọc các hàng trong DataFrame dựa trên điều kiện cột:

```python
import pandas as pd

data = {
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': ['a', 'b', 'c', 'd']
}
df = pd.DataFrame(data)

filtered_df = df[df['A'] > 2]
print(filtered_df)
# Output:
#    A  B  C
# 2  3  7  c
# 3  4  8  d
```

### Các thao tác với cột trong DataFrame

Ta có thể thêm, xoá hoặc đổi tên cột trong DataFrame một cách dễ dàng.

```python
import pandas as pd
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}
df = pd.DataFrame(data)

# Thêm cột mới
df['C'] = ['a', 'b', 'c']
print(df)
# Output:
#    A  B  C
# 0  1  4  a
# 1  2  5  b
# 2  3  6  c

# Xoá cột
df = df.drop(columns=['B'])
print(df)
# Output:
#    A  C
# 0  1  a
# 1  2  b
# 2  3  c

# Đổi tên cột
df = df.rename(columns={'A': 'Alpha', 'C': 'Charlie'})
print(df)
# Output:
#    Alpha Charlie
# 0      1       a
# 1      2       b
# 2      3       c
```

### Các thao tác với hàng trong DataFrame

Một số thao tác xử lý dữ liệu phổ biến như kiểm tra dữ liệu khuyết thiếu, điền dữ liệu khuyết thiếu, loại bỏ dữ liệu khuyết thiếu, kiểm tra dữ liệu trùng lặp, loại bỏ dữ liệu trùng lặp, sắp xếp dữ liệu, v.v.

Ví dụ xử lý dữ liệu khuyết thiếu:

```python
import pandas as pd

data = {
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': ['a', 'b', 'c', 'd']
}
df = pd.DataFrame(data)
print(df)
# Output:
#      A    B  C
# 0  1.0  5.0  a
# 1  2.0  NaN  b
# 2  NaN  7.0  c
# 3  4.0  8.0  d

# Kiểm tra dữ liệu khuyết thiếu
print(df.isnull())
# Output:
#        A      B      C
# 0  False  False  False
# 1  False   True  False
# 2   True  False  False
# 3  False  False  False

# Điền dữ liệu khuyết thiếu với giá trị cụ thể
df_filled = df.fillna(-99)
print(df_filled)
# Output:
#       A     B  C
# 0   1.0   5.0  a
# 1   2.0 -99.0  b
# 2 -99.0   7.0  c
# 3   4.0   8.0  d

# Loại bỏ hàng có dữ liệu khuyết thiếu
df_dropped = df.dropna()
print(df_dropped)
# Output:
#      A    B  C
# 0  1.0  5.0  a
# 3  4.0  8.0  d
```

Ví dụ xử lý dữ liệu trùng lặp:

```python
import pandas as pd

data = {
    'A': [1, 2, 2, 4],
    'B': [5, 6, 6, 8],
    'C': ['a', 'b', 'b', 'd']
}
df = pd.DataFrame(data)
print(df)
# Output:
#    A  B  C
# 0  1  5  a
# 1  2  6  b
# 2  2  6  b
# 3  4  8  d

# Kiểm tra dữ liệu trùng lặp
print(df.duplicated())
# Output:
# 0    False
# 1    False
# 2     True
# 3    False
# dtype: bool

# Loại bỏ dữ liệu trùng lặp
df_no_duplicates = df.drop_duplicates()
print(df_no_duplicates)
# Output:
#    A  B  C
# 0  1  5  a
# 1  2  6  b
# 3  4  8  d
```

Ví dụ sắp xếp dữ liệu:

```python
import pandas as pd

data = {
    'A': [3, 1, 4, 2],
    'B': [8, 5, 7, 6],
    'C': ['d', 'a', 'c', 'b']
}
df = pd.DataFrame(data)
print(df)
# Output:
#    A  B  C
# 0  3  8  d
# 1  1  5  a
# 2  4  7  c
# 3  2  6  b

# Sắp xếp theo cột A
df_sorted = df.sort_values(by='A')
print(df_sorted)
# Output:
#    A  B  C
# 1  1  5  a
# 3  2  6  b
# 0  3  8  d
# 2  4  7  c
```

## 4. Đọc và ghi dữ liệu với Pandas

Pandas hỗ trợ đọc và ghi dữ liệu từ nhiều định dạng khác nhau như CSV, Excel, JSON, SQL, v.v.

### Đọc dữ liệu từ file CSV

CSV (Comma-Separated Values) là định dạng phổ biến để lưu trữ dữ liệu dạng bảng và hàm `pd.read_csv()` cũng được sử dụng rộng rãi nhất để đọc dữ liệu vào DataFrame.

Trong hàm `pd.read_csv()`, ngoài tham số đường dẫn file, ta còn có thể sử dụng nhiều tham số khác để tùy chỉnh quá trình đọc dữ liệu như:
- `sep`: Xác định ký tự phân tách (mặc định là dấu phẩy `,`).
Ta có thể thay đổi nếu file sử dụng ký tự khác như tab (`\t`), dấu chấm phẩy (`;`) ... để đọc các file có định dạng khác như TSV (Tab-Separated Values) ...
- `usecols`: Chỉ định các cột cụ thể cần đọc từ file.
Đối với các file lớn, việc chỉ đọc những cột cần thiết giúp tiết kiệm bộ nhớ và tăng tốc độ xử lý.
- `chunksize`: Đọc dữ liệu theo từng phần (chunk) để tiết kiệm bộ nhớ khi làm việc với file lớn.
Đối với các file rất lớn, ta có thể đọc dữ liệu theo từng phần nhỏ (chunk) thay vì đọc toàn bộ file vào bộ nhớ cùng lúc.

Ví dụ đọc dữ liệu từ file CSV:

```python
import pandas as pd

df = pd.read_csv('data.csv')
print(df.head())

# Đọc dữ liệu từ file CSV với ký tự phân tách là tab
df_tsv = pd.read_csv('data.tsv', sep='\t')
print(df_tsv.head())

# Đọc chỉ các cột 'A' và 'B' từ file CSV
df_subset = pd.read_csv('data.csv', usecols=['A', 'B'])
print(df_subset.head())

# Đọc dữ liệu theo từng phần (chunk) với kích thước 1000 dòng
chunk_size = 1000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    print(chunk.head())
```

### Đọc dữ liệu từ file Excel

Pandas cung cấp hàm `pd.read_excel()` để đọc dữ liệu từ file Excel (.xls, .xlsx).
Khi sử dụng hàm này, ta có thể chỉ định tên sheet cần đọc thông qua tham số `sheet_name`.

Ví dụ đọc dữ liệu từ file Excel:

```python
import pandas as pd

df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
print(df.head())
```

### Đọc dữ liệu từ database SQL

Pandas cung cấp hàm `pd.read_sql()` để đọc dữ liệu từ cơ sở dữ liệu SQL.
Ví dụ đọc dữ liệu từ bảng trong cơ sở dữ liệu PostgreSQL:

```python
import pandas as pd
import sqlalchemy

# Tạo kết nối đến cơ sở dữ liệu PostgreSQL
engine = sqlalchemy.create_engine('postgresql://username:password@localhost:5432/mydatabase')

# Đọc dữ liệu từ bảng 'mytable'
df = pd.read_sql('mytable', con=engine)
print(df.head())
```

### Ghi dữ liệu ra file CSV

Pandas cung cấp hàm `df.to_csv()` để ghi dữ liệu từ DataFrame ra file CSV.

Ví dụ ghi dữ liệu ra file CSV:

```python
import pandas as pd

data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': ['a', 'b', 'c']
}
df = pd.DataFrame(data)
df.to_csv('output.csv', index=False)  # Ghi dữ liệu ra file CSV
```

## 5. Xử lý dữ liệu với Pandas

### Thao tác nhóm (grouping) và tổng hợp (aggregation)

Một thao tác quan trọng khác trong phân tích dữ liệu là nhóm và tính toán tổng hợp – ví dụ, tính điểm trung bình theo lớp, tổng doanh thu theo từng ngành hàng, số lượng khách hàng theo thành phố... Pandas hỗ trợ điều này thông qua cơ chế GroupBy.

Sau khi nhóm dữ liệu theo một hoặc nhiều cột, ta có thể áp dụng các hàm tổng hợp dữ liệu như `sum()`, `mean()`, `count()`, `min()`, `max()` để tính toán các giá trị tổng hợp cho từng nhóm.
Những thao tác này tương tự như các câu lệnh SQL với `GROUP BY` và các hàm tổng hợp.

Ví dụ nhóm dữ liệu:

```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Class': ['A', 'A', 'B', 'B', 'C'],
    'Score': [85, 90, 78, 88, 92]
}
df = pd.DataFrame(data)

grouped = df.groupby('Class')
print(grouped)
# Output:
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x...>

for class_name, group in grouped:
    print(f"Group: {class_name}")
    print(group)

# Output:
# Group: A
#     Name Class  Score
# 0  Alice     A     85
# 1    Bob     A     90
# Group: B
#       Name Class  Score
# 2  Charlie     B     78
# 3    David     B     88
# Group: C
#    Name Class  Score
# 4   Eva     C     92

# Tính điểm trung bình theo lớp
mean_scores = grouped['Score'].mean()
print(mean_scores)
# Output:
# Class
# A    87.5
# B    83.0
# C    92.0
# Name: Score, dtype: float64

# Tổng hợp dữ liệu trên nhiều cột với các hàm khác nhau
agg_scores = grouped.agg({'Score': ['mean', 'max', 'min', 'count'], 'Name': 'list'})
print(agg_scores)

# Output:
#        Score                 Name
#         mean max min count
# Class
# A      87.5  90  85     2  [Alice, Bob]
# B      83.0  88  78     2  [Charlie, David]
# C      92.0  92  92     1  [Eva]
```

### Kết hợp dữ liệu (merging, joining, concatenating)

Pandas cung cấp các hàm để kết hợp dữ liệu từ nhiều DataFrame khác nhau, bao gồm:
- `pd.merge()`: Kết hợp hai DataFrame dựa trên một hoặc nhiều cột chung (tương tự như phép JOIN trong SQL).
Các tham số quan trọng trong hàm `pd.merge()` bao gồm:
    - `on`: Xác định cột chung để kết hợp.
    - `how`: Xác định kiểu kết hợp (inner, outer, left, right).
- `pd.concat()`: Nối hai hoặc nhiều DataFrame theo chiều dọc (thêm hàng) hoặc chiều ngang (thêm cột).
Tham số quan trọng trong hàm `pd.concat()` bao gồm:
    - `axis`: Xác định chiều nối (0 cho hàng, 1 cho cột).
    - Lưu ý khi sử dụng `concat` là các DataFrame cần có cùng số cột (khi nối theo hàng) hoặc cùng số hàng (khi nối theo cột).

Ví dụ kết hợp dữ liệu với `merge`:

```python
import pandas as pd

data1 = {
    'ID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie']
}
data2 = {
    'ID': [2, 3, 4],
    'Score': [90, 85, 88]
}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

merged_df = pd.merge(df1, df2, on='ID', how='inner')  # Inner join
print(merged_df)
# Output:
#    ID     Name  Score
# 0   2      Bob     90
# 1   3  Charlie     85
```

Ví dụ kết hợp dữ liệu với `concat`:

```python
import pandas as pd

data1 = {
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}
data2 = {
    'A': [7, 8, 9],
    'B': [10, 11, 12]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Nối hai DataFrame theo chiều dọc
concat_df = pd.concat([df1, df2], axis=0, ignore_index=True)
print(concat_df)

# Output:
#    A   B
# 0  1   4
# 1  2   5
# 2  3   6
# 3  7  10
# 4  8  11
# 5  9  12

# Nối hai DataFrame theo chiều ngang
concat_df_horiz = pd.concat([df1, df2], axis=1)
print(concat_df_horiz)

# Output:
#    A  B  A   B
# 0  1  4  7  10
# 1  2  5  8  11
# 2  3  6  9  12
```

### Chỉnh sửa dữ liệu theo hàm (apply)

Ta có thể áp dụng một hàm tùy chỉnh cho từng phần tử trong Series hoặc từng hàng/cột trong DataFrame bằng phương thức `apply()`.
Điều này rất hữu ích khi ta cần thực hiện các biến đổi phức tạp mà các hàm tích hợp sẵn không hỗ trợ, giúp linh hoạt tuỳ biến quá trình xử lý dữ liệu.

Ví dụ sử dụng `apply` trên Series:

```python
import pandas as pd

data = [1, 2, 3, 4, 5]
s = pd.Series(data)
s_squared = s.apply(lambda x: x ** 2)
print(s_squared)
# Output:
# 0     1
# 1     4
# 2     9
# 3    16
# 4    25
# dtype: int64
```

Ví dụ sử dụng `apply` trên một cột và nhiều cột trong DataFrame:

```python
import pandas as pd

data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}
df = pd.DataFrame(data)

# Áp dụng hàm cho một cột, tạo ra một cột mới
df['A_squared'] = df['A'].apply(lambda x: x ** 2)
print(df)
# Output:
#    A  B  A_squared
# 0  1  4          1
# 1  2  5          4
# 2  3  6          9

# Áp dụng hàm cho nhiều cột, tạo ra một cột mới
df['A_plus_B'] = df.apply(lambda row: row['A'] + row['B'], axis=1)
print(df)
# Output:
#    A  B  A_squared  A_plus_B
# 0  1  4          1         5
# 1  2  5          4         7
# 2  3  6          9         9

# Áp dụng hàm cho nhiều cột, tạo ra nhiều cột mới
def compute_stats(row):
    return pd.Series({
        'A_cubed': row['A'] ** 3,
        'B_cubed': row['B'] ** 3,
        'A_minus_B': row['A'] - row['B']
    })

df_stats = df.apply(compute_stats, axis=1)
df = pd.concat([df, df_stats], axis=1)
print(df)
# Output:
#    A  B  A_squared  A_plus_B  A_cubed  B_cubed  A_minus_B
# 0  1  4          1         5        1       64         -3
# 1  2  5          4         7        8      125         -3
# 2  3  6          9         9       27      216         -3
```
