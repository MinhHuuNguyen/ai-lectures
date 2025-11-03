---
time: 02/18/2022
title: Trực quan hoá dữ liệu với Matplotlib và Seaborn
description: Matplotlib và Seaborn là hai thư viện trực quan hoá dữ liệu phổ biến trong Python. Cả hai thư viện này cung cấp các công cụ mạnh mẽ giúp tạo ra các biểu đồ đẹp mắt, dễ đọc và dễ hiểu. Sự kết hợp giữa Matplotlib và Seaborn giúp người dùng có thể tận dụng được ưu điểm của cả hai thư viện để trực quan hoá dữ liệu một cách hiệu quả.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/banner.jpeg
tags: [python]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về thư viện Matplotlib và Seaborn

Matplotlib là một thư viện vẽ đồ thị rất phổ biến trong Python, được sử dụng để tạo ra các biểu đồ và hình ảnh trực quan chất lượng cao.
Thư viện này cung cấp nhiều công cụ giúp dễ dàng trực quan hóa dữ liệu – tức biến dữ liệu số khô khan thành các biểu đồ sinh động để phân tích và trình bày.
Nói cách khác, nếu bạn dự định làm việc với dữ liệu trong Python, Matplotlib là một công cụ không thể thiếu giúp bạn nhìn thấy xu hướng và ý nghĩa của dữ liệu thay vì chỉ nhìn những con số.

Seaborn là một thư viện trực quan hóa dữ liệu (data visualization) mạnh mẽ trong Python, được xây dựng trên nền tảng Matplotlib.
Mục tiêu của Seaborn là cung cấp một giao diện cấp cao (high-level interface) giúp tạo ra các biểu đồ thống kê phức tạp một cách dễ dàng với ít mã lệnh hơn so với Matplotlib thuần túy.
Thư viện này đặc biệt tối ưu cho các biểu đồ thống kê (như biểu đồ phân phối, hồi quy, biểu đồ cho dữ liệu phân loại...) và có sẵn các định dạng hiển thị đẹp mắt với các chủ đề và bảng màu mặc định hài hòa.

Một điểm nổi bật khác là Seaborn tích hợp chặt chẽ với Pandas DataFrame.
Các hàm vẽ của Seaborn có thể nhận trực tiếp DataFrame và tên cột để biểu diễn trục X, Y, nhóm màu sắc,... mà không cần lấy thủ công từng mảng dữ liệu – điều này giúp việc xử lý và trực quan hóa dữ liệu dạng bảng trở nên thuận tiện và trực quan hơn.
Nói cách khác, Seaborn “hiểu” cấu trúc dữ liệu Pandas và tự động ánh xạ các cột vào biểu đồ phù hợp, giúp bạn viết code ngắn gọn hơn so với Matplotlib.

#### Cài đặt và sử dụng Matplotlib và Seaborn

Để sử dụng Matplotlib và Seaborn, trước hết ta cần cài đặt các thư viện này.
Nếu chưa cài, ta có thể dùng pip:

```bash
pip install matplotlib
pip install seaborn
```

hoặc nếu dùng Anaconda:

```bash
conda install matplotlib
conda install seaborn
```

Sau khi cài đặt, ta có thể import NumPy trong mã Python như sau:

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

Matplotlib có hai module chính để vẽ biểu đồ là `pyplot` và `pylab`:
- `pyplot`: Đây là module phổ biến và được sử dụng rộng rãi nhất trong Matplotlib. `pyplot` cung cấp giao diện theo dạng state-machine, tương tự như MATLAB, giúp bạn dễ dàng tạo và tuỳ chỉnh các biểu đồ.
- `pylab`: Đây là một module kết hợp giữa `pyplot` và NumPy. Tuy nhiên, module này rất ít được sử dụng trong thực tế.

## 2. Các biểu đồ cơ bản với Matplotlib và Seaborn

### 2.1. Biểu đồ đường (Line Plot)

#### Matplotlib

Ví dụ vẽ biểu đồ đường đơn giản với Matplotlib gồm trục X là các tháng và trục Y là doanh số bán hàng:

```python
import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = [250, 300, 400, 350, 500]

plt.plot(months, sales, marker='o')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid()
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/line_plot_matplotlib.jpeg" style="width: 550px;"/>

#### Seaborn

Ví dụ vẽ biểu đồ đường với Seaborn:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Sales': [250, 300, 400, 350, 500]
}
df = pd.DataFrame(data)

sns.lineplot(data=df, x='Month', y='Sales', marker='o')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/line_plot_seaborn.jpeg" style="width: 550px;"/>

### 2.2. Biểu đồ cột (Bar Plot)

#### Matplotlib

Ví dụ vẽ biểu đồ cột với Matplotlib gồm trục X là các tháng và trục Y là doanh số bán hàng:

```python
import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = [250, 300, 400, 350, 500]

plt.bar(months, sales, color='skyblue')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/bar_plot_matplotlib.jpeg" style="width: 550px;"/>

Hoặc ta có thể vẽ biểu đồ cột nhưng nằm ngang:

```python
import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = [250, 300, 400, 350, 500]

plt.barh(months, sales, color='skyblue')
plt.title('Monthly Sales')
plt.xlabel('Sales')
plt.ylabel('Month')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/barh_plot_matplotlib.jpeg" style="width: 550px;"/>

#### Seaborn

Ví dụ vẽ biểu đồ cột với Seaborn:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Sales': [250, 300, 400, 350, 500]
}
df = pd.DataFrame(data)

sns.barplot(data=df, x='Month', y='Sales', color='skyblue')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/bar_plot_seaborn.jpeg" style="width: 550px;"/>

Hoặc ta có thể vẽ biểu đồ cột nhưng nằm ngang:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Sales': [250, 300, 400, 350, 500]
}
df = pd.DataFrame(data)

sns.barplot(data=df, y='Month', x='Sales', color='skyblue')
plt.title('Monthly Sales')
plt.xlabel('Sales')
plt.ylabel('Month')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/barh_plot_seaborn.jpeg" style="width: 550px;"/>

### 2.3. Biểu đồ phân tán (Scatter Plot)

#### Matplotlib

Ví dụ vẽ biểu đồ phân tán với Matplotlib gồm trục X là chiều cao và trục Y là cân nặng của một nhóm người:

```python
import matplotlib.pyplot as plt
import numpy as np

height = np.random.randint(150, 200, 50)  # Chiều cao giả lập
weight = np.random.randint(50, 100, 50)   # Cân nặng giả lập

plt.scatter(height, weight, color='green')
plt.title('Height vs Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.grid()
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/scatterplot_matplotlib.jpeg" style="width: 550px;"/>

#### Seaborn

Ví dụ vẽ biểu đồ phân tán với Seaborn:

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = {
    'Height': np.random.randint(150, 200, 50),  # Chiều cao giả lập
    'Weight': np.random.randint(50, 100, 50)    # Cân nặng giả lập
}
df = pd.DataFrame(data)
sns.scatterplot(data=df, x='Height', y='Weight', color='green')
plt.title('Height vs Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/scatterplot_seaborn.jpeg" style="width: 550px;"/>

### 2.4. Biểu đồ tròn (Pie Plot)

#### Matplotlib

Ví dụ vẽ biểu đồ tròn với Matplotlib gồm các phần trăm thị phần của các sản phẩm:

```python
import matplotlib.pyplot as plt

labels = ['Product A', 'Product B', 'Product C', 'Product D']
sizes = [30, 25, 25, 20]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Market Share of Products')
plt.axis('equal')  # Đảm bảo biểu đồ tròn
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/pie_plot_matplotlib.jpeg" style="width: 550px;"/>

#### Seaborn

Seaborn không hỗ trợ trực tiếp biểu đồ tròn.

### 2.6. Biểu đồ phân phối (Distribution Plot)

#### Matplotlib

Ví dụ vẽ biểu đồ phân phối với Matplotlib gồm dữ liệu điểm số của một lớp học:

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.normal(70, 10, 100)  # Dữ liệu điểm số giả lập

plt.hist(data, bins=10, color='purple', alpha=0.7)
plt.title('Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.grid()
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/histogram_plot_matplotlib.jpeg" style="width: 550px;"/>

#### Seaborn

Ví dụ vẽ biểu đồ phân phối với Seaborn:

```python
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = np.random.normal(70, 10, 100)  # Dữ liệu điểm số giả lập
sns.histplot(data, bins=10, color='purple', kde=True)
plt.title('Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/histogram_plot_seaborn.jpeg" style="width: 550px;"/>

## 3. Một số câu lệnh chức năng bổ sung trong Matplotlib

### 3.1. Chú thích biểu đồ (Legend)

Trong Matplotlib, ta có thể thêm chú thích cho biểu đồ bằng cách sử dụng hàm `plt.legend()`.
Ví dụ:

```python
import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales_A = [250, 300, 400, 350, 500]
sales_B = [200, 250, 300, 400, 450]

plt.plot(months, sales_A, marker='o', label='Product A')
plt.plot(months, sales_B, marker='s', label='Product B')
plt.title('Monthly Sales Comparison')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()  # Hiển thị chú thích
plt.grid()
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/legend_matplotlib.jpeg" style="width: 550px;"/>

### 3.2. Lưu biểu đồ (Save Figure)

Để lưu biểu đồ dưới dạng file ảnh, ta sử dụng hàm `plt.savefig()`.
Ví dụ:

```python
import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = [250, 300, 400, 350, 500]

plt.plot(months, sales, marker='o')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid()
plt.savefig('monthly_sales.jpeg')  # Lưu biểu đồ dưới dạng file PNG
plt.show()
```

### 3.3. Tuỳ chỉnh kích thước biểu đồ (Figure Size)

Để tuỳ chỉnh kích thước biểu đồ, ta sử dụng hàm `plt.figure(figsize=(width, height))`.
Ví dụ:

```python
import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = [250, 300, 400, 350, 500]

plt.figure(figsize=(10, 6))  # Kích thước biểu đồ 10x6 inch
plt.plot(months, sales, marker='o')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid()
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/figsize_matplotlib.jpeg" style="width: 750px;"/>

### 3.4. Tuỳ chỉnh màu sắc và kiểu dáng (Color and Style)

Trong Matplotlib, ta có thể tuỳ chỉnh màu sắc và kiểu dáng của biểu đồ bằng cách sử dụng các tham số trong hàm vẽ.
Ví dụ:

```python
import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = [250, 300, 400, 350, 500]

plt.plot(months, sales, marker='o', color='orange', linestyle='--', linewidth=2, markersize=8)
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid()
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/color_style_matplotlib.jpeg" style="width: 550px;"/>

### 3.5. Hiển thị nhiều biểu đồ trong cùng một hình (Subplots)

Để hiển thị nhiều biểu đồ trong cùng một hình, ta sử dụng hàm `plt.subplot()`.
Ví dụ:
```python
import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales_A = [250, 300, 400, 350, 500]
sales_B = [200, 250, 300, 400, 450]

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)  # Biểu đồ đầu tiên trong 1 hàng, 2 cột
plt.plot(months, sales_A, marker='o', color='blue')
plt.title('Product A Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid()

plt.subplot(1, 2, 2)  # Biểu đồ thứ hai trong 1 hàng, 2 cột
plt.plot(months, sales_B, marker='s', color='red')
plt.title('Product B Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid()

plt.tight_layout()  # Tự động điều chỉnh khoảng cách giữa các biểu đồ
plt.suptitle('Monthly Sales Comparison', fontsize=16)  # Tiêu đề chung cho toàn bộ hình
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/subplots_matplotlib.jpeg" style="width: 750px;"/>

## 4. Các biểu đồ nâng cao với Seaborn

### 4.1. Tham số `hue` trong các biểu đồ

Nhiều hàm vẽ của Seaborn hỗ trợ tham số `hue`, cho phép phân loại dữ liệu theo một biến khác.
Ví dụ, ta có thể vẽ biểu đồ cột với phân loại theo giới tính:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = {
    'Height': [150, 160, 170, 180, 190, 155, 165, 175, 185, 195],
    'Weight': [55, 50, 65, 80, 90, 60, 70, 75, 85, 95],
    'Gender': ['F', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'M'],
    'Age': [20, 22, 23, 24, 25, 21, 23, 24, 26, 27]
}
df = pd.DataFrame(data)

sns.lineplot(data=df, x='Height', y='Weight', hue='Gender', marker='o')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/hue_seaborn.jpeg" style="width: 550px;"/>

### 4.2. Biểu đồ phân tán (Scatter Plot) nâng cao với Seaborn

Ta có thể nâng cấp biểu đồ phân tán trong Seaborn với đường hồi quy (regression line) thành biểu đồ `regplot`.
Ví dụ:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = {
    'Height': [150, 160, 170, 180, 190, 155, 165, 175, 185, 195],
    'Weight': [55, 50, 65, 80, 90, 60, 70, 75, 85, 95],
    'Gender': ['F', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'M'],
    'Age': [20, 22, 23, 24, 25, 21, 23, 24, 26, 27]
}
df = pd.DataFrame(data)

sns.regplot(data=df, x='Height', y='Weight', scatter_kws={'s':100, 'color':'blue'}, line_kws={'color':'red'})
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight with Regression Line')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/regplot_seaborn.jpeg" style="width: 550px;"/>

Ta cũng có thể bổ sung thêm histogram cho trục X và Y bằng cách sử dụng `jointplot`:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = {
    'Height': [150, 160, 170, 180, 190, 155, 165, 175, 185, 195],
    'Weight': [55, 50, 65, 80, 90, 60, 70, 75, 85, 95],
    'Gender': ['F', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'M'],
    'Age': [20, 22, 23, 24, 25, 21, 23, 24, 26, 27]
}
df = pd.DataFrame(data)

sns.jointplot(data=df, x='Height', y='Weight', hue='Gender', kind='scatter')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight with Gender')
plt.tight_layout()
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/jointplot_seaborn.jpeg" style="width: 550px;"/>

Ta còn có thể thêm đường hồi quy vào biểu đồ `jointplot`:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = {
    'Height': [150, 160, 170, 180, 190, 155, 165, 175, 185, 195],
    'Weight': [55, 50, 65, 80, 90, 60, 70, 75, 85, 95],
    'Gender': ['F', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'M'],
    'Age': [20, 22, 23, 24, 25, 21, 23, 24, 26, 27]
}
df = pd.DataFrame(data)

sns.jointplot(data=df, x='Height', y='Weight', kind='reg', scatter_kws={'s':100, 'color':'blue'}, line_kws={'color':'red'})
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight with Regression Line')
plt.tight_layout()
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/jointplot_reg_seaborn.jpeg" style="width: 550px;"/>

### 4.3. Biểu đồ hộp (Box Plot) và biểu đồ violin (Violin Plot)

Biểu đồ hộp (box plot) và biểu đồ violin (violin plot) là hai loại biểu đồ thống kê phổ biến để hiển thị phân phối của dữ liệu số.
Ví dụ vẽ biểu đồ hộp:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = {
    'Height': [150, 160, 170, 180, 190, 155, 165, 175, 185, 195],
    'Weight': [55, 50, 65, 80, 90, 60, 70, 75, 85, 95],
    'Gender': ['F', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'M'],
    'Age': [20, 22, 23, 24, 25, 21, 23, 24, 26, 27]
}
df = pd.DataFrame(data)

sns.boxplot(x='Weight', data=df, color='lightblue')
plt.xlabel('Weight (kg)')
plt.title('Box Plot of Weight')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/boxplot_seaborn.jpeg" style="width: 550px;"/>

Hoặc vẽ biểu đồ violin:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = {
    'Height': [150, 160, 170, 180, 190, 155, 165, 175, 185, 195],
    'Weight': [55, 50, 65, 80, 90, 60, 70, 75, 85, 95],
    'Gender': ['F', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'M'],
    'Age': [20, 22, 23, 24, 25, 21, 23, 24, 26, 27]
}
df = pd.DataFrame(data)

sns.violinplot(x='Weight', data=df, color='lightgreen')
plt.xlabel('Weight (kg)')
plt.title('Violin Plot of Weight')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/violinplot_seaborn.jpeg" style="width: 550px;"/>

#### Giải thích thêm về biểu đồ hộp và biểu đồ violin

- **Biểu đồ hộp (box plot)** hiển thị các đặc trưng thống kê của dữ liệu như giá trị tối thiểu, giá trị tối đa, trung vị (median), và các phần tư (quartiles).
Nó giúp ta nhanh chóng nhận biết phân phối, độ lệch và các giá trị ngoại lai (outliers) trong tập dữ liệu.
- **Biểu đồ violin (violin plot)** kết hợp giữa biểu đồ hộp và biểu đồ mật độ (density plot).
Nó không chỉ hiển thị các đặc trưng thống kê như biểu đồ hộp mà còn thể hiện phân phối của dữ liệu thông qua hình dạng của "violin".
Điều này giúp ta hiểu rõ hơn về cách dữ liệu phân bố, đặc biệt là khi có nhiều điểm dữ liệu hoặc phân phối không đồng đều.

### 4.4. Biểu đồ nhiệt (Heatmap)

Biểu đồ nhiệt (heatmap) là một công cụ trực quan hóa dữ liệu dạng bảng, trong đó các giá trị được biểu diễn bằng màu sắc.
Ví dụ vẽ biểu đồ nhiệt:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(style="darkgrid")

data = np.random.rand(10, 10)  # Dữ liệu ngẫu nhiên 10x10
df = pd.DataFrame(data, columns=[f'Col {i}' for i in range(10)], index=[f'Row {i}' for i in range(10)])

sns.heatmap(df, annot=True, fmt=".2f", cmap='YlGnBu')
plt.title('Heatmap Example')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/heatmap_seaborn.jpeg" style="width: 550px;"/>

### 4.5. Biểu đồ đôi (Pair Plot)

Biểu đồ đôi (pair plot) là một công cụ trực quan hóa mối quan hệ giữa các biến trong một DataFrame.
Ví dụ vẽ biểu đồ đôi:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data = {
    'Height': [150, 160, 170, 180, 190, 155, 165, 175, 185, 195],
    'Weight': [55, 50, 65, 80, 90, 60, 70, 75, 85, 95],
    'Age': [20, 22, 23, 24, 25, 21, 23, 24, 26, 27],
    'Gender': ['F', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'M']
}
df = pd.DataFrame(data)

sns.pairplot(df, hue='Gender', markers=['o', 's'])
plt.suptitle('Pair Plot of Height, Weight, and Age')
plt.tight_layout()
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/pairplot_seaborn.jpeg" style="width: 750px;"/>

## 5. Một số tính năng hữu ích khác của Matplotlib và Seaborn

### 5.1. Một số bộ dữ liệu mẫu (Sample Datasets) trong Seaborn

Seaborn cung cấp một số bộ dữ liệu mẫu (sample datasets) để người dùng có thể thực hành và thử nghiệm các hàm vẽ biểu đồ.
Ví dụ, ta có thể sử dụng bộ dữ liệu `tips` để vẽ biểu đồ:

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

tips = sns.load_dataset('tips')

sns.scatterplot(data=tips, x='total_bill', y='tip', hue='day', style='time')
plt.title('Total Bill vs Tip')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.show()
```

Ta thu được biểu đồ như sau:

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/3-matplotlib-seaborn/data_seaborn.jpeg" style="width: 550px;"/>
