---
time: 03/02/2023
title: Dự báo chuỗi thời gian Time Series Forecasting
description: Dự báo chuỗi thời gian là bài toán phổ biến mang tính ứng dụng cao của machine learning. Bài toán dự báo với dữ liệu Time series là lĩnh vực rộng với lịch sử lâu đời. Đối tượng cơ bản trong quá trình dự báo là chuỗi thời gian, là tập hợp các quan sát dữ liệu được thu thập theo thời gian.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/banner.jpeg
tags: [machine-learning, deep-learning, time-series]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về Chuỗi thời gian Time Series

Chuỗi thời gian - Time Series là tập hợp các giá trị được thu thập theo thứ tự thời gian đều đặn, như nhiệt độ hàng ngày, giá cổ phiếu hàng giờ, hay GDP theo năm.
Khác với dữ liệu độc lập thông thường, các quan sát thời gian thường có tính tự tương quan cao (giá trị tại thời điểm $t$ thường phụ thuộc vào các giá trị trước đó).

Đặc điểm quan trọng của dữ liệu chuỗi thời gian là mỗi quan sát có thứ tự tự nhiên theo thời gian, và giá trị hiện tại thường phụ thuộc vào các giá trị trong quá khứ.
Vì vậy, bài toán chính với dữ liệu chuỗi thời gian là dự báo (forecasting) – tức dự đoán các giá trị tương lai dựa trên lịch sử đã có, thay vì bài toán dự đoán thông thường.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/forecasting.jpeg" style="width: 600px;"/>

Dự báo chuỗi thời gian là một trong những kỹ thuật được ứng dụng rộng rãi trong kinh doanh, tài chính, chuỗi cung ứng, và nhiều lĩnh vực khác.
Ví dụ: dự báo doanh số bán hàng tuần tới dựa trên doanh số các tuần trước, dự báo nhiệt độ ngày mai dựa trên nhiệt độ các ngày trước đó.
Mục tiêu là học từ dữ liệu lịch sử (đầu vào) một mô hình có khả năng dự báo các giá trị trong tương lai một cách chính xác.

## 2. Các khái niệm cơ bản trong Time series

### 2.1. Timestamps và periods

Timestamp là một điểm thời gian cụ thể, xác định duy nhất một thời khắc trên dòng thời gian.
Timestamps thường biểu diễn một thời điểm chính xác như: ngày, giờ, phút, giây, hoặc thậm chí là microsecond.

Ví dụ: "2023-03-01 12:00:00" là một timestamp xác định thời điểm chính xác.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/timestamp_period.jpeg" style="width: 800px;"/>

Period là một khoảng thời gian có độ dài cố định, giữa hai timestamp như: một ngày, tháng, năm, hoặc quý.
Nó biểu thị một khoảng thời gian liền kề, thay vì một điểm thời gian cụ thể như timestamp.

Ví dụ: "2023-03-01 đến 2023-03-31" là một period xác định khoảng thời gian từ đầu tháng 3 đến cuối tháng 3 năm 2023.

Timestamps và periods có thể được chuyển đổi qua lại với nhau.

### 2.2. Resampling dữ liệu Time series

Nếu như trong dữ liệu hình ảnh, resampling dữ liệu là việc thay đổi kích thước của ảnh, thì trong Time series, resampling dữ liệu là việc thay đổi tần suất lấy mẫu của dữ liệu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/upsampling.jpeg" style="width: 1000px;"/>

Resampling dữ liệu Time series là việc thay đổi tần suất lấy mẫu của dữ liệu, từ tần suất cao xuống tần suất thấp hoặc ngược lại.
Có hai cách để lấy mẫu dữ liệu thời gian trong Time series:
- **Upsampling:** lấy mẫu tăng tần số, từ tần số thấp lên tần số cao
    - Ví dụ: từ lấy mẫu dữ liệu hàng tháng thành lấy mẫu thành dữ liệu hàng ngày.
    - Phương pháp này đòi hỏi ta phải có cách để làm đầy (nội suy) những khoảng giá trị bị khuyết thiếu.
- **Downsampling:** lấy mẫu giảm tần số, từ tần số cao xuống tần số thấp
    - Ví dụ: từ lấy mẫu dữ liệu hàng ngày thành lấy mẫu thành dữ liệu hàng tháng.
    - Phương pháp này đòi hỏi ta phải có cách để tổng hợp dữ liệu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/downsampling.jpeg" style="width: 1000px;"/>

Ví dụ: Giả sử ta có dữ liệu về doanh thu hàng tháng trong một năm như sau:

| Tháng      | Doanh thu |
|------------|-----------|
| Tháng 1    | 1000      |
| Tháng 2    | 1200      |
| Tháng 3    | 1500      |
| Tháng 4    | 1300      |
| Tháng 5    | 1400      |
| Tháng 6    | 1500      |
| Tháng 7    | 1600      |
| Tháng 8    | 1700      |
| Tháng 9    | 1400      |
| Tháng 10   | 1900      |
| Tháng 11   | 1500      |
| Tháng 12   | 1900      |

Nếu ta muốn downsample dữ liệu này xuống tần suất hàng quý, ta có thể tính tổng doanh thu của mỗi quý:

| Quý       | Doanh thu |
|-----------|-----------|
| Quý 1     | 3700      |
| Quý 2     | 4200      |
| Quý 3     | 4700      |
| Quý 4     | 5300      |

Nếu ta muốn upsample dữ liệu này lên tần suất hàng tuần, ta có thể nội suy để điền giá trị doanh thu cho từng tuần trong tháng.

Ta có thể giả định doanh thu trong mỗi tháng được phân bổ tăng dần theo tuần và có tổng bằng doanh thu của tháng đó.

| Tuần              | Doanh thu |
|-------------------|-----------|
| Tuần 1 tháng 1    | 175       |
| Tuần 2 tháng 1    | 225       |
| Tuần 3 tháng 1    | 275       |
| Tuần 4 tháng 1    | 325       |
| Tuần 1 tháng 2    | 225       |
| Tuần 2 tháng 2    | 275       |
| Tuần 3 tháng 2    | 325       |
| Tuần 4 tháng 2    | 375       |
| ...               | ...       |

### 2.3. Xu hướng - Trend

Xu hướng - Trend thể hiện sự thay đổi ổn định kéo dài của chuỗi thời gian, thể hiện sự thay đổi tổng thể của dữ liệu theo thời gian (tăng, giảm hoặc ổn định) và là phần không mang tính chu kỳ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/trend.jpeg" style="width: 1000px;"/>

Xu hướng có thể là ngắn hạn (ví dụ: tăng trưởng doanh thu trong một quý) hoặc dài hạn (ví dụ: tăng trưởng doanh thu trong nhiều năm).

### 2.4. Thời vụ - Seasonality và Chu kỳ - Cycle

Thời vụ - Seasonality thể hiện sự lặp đi lặp lại có quy luật trong chuỗi thời gian theo một chu kỳ ngắn hạn cố định, thường là theo quý, tháng, tuần, hoặc ngày.

Tính thời vụ thường xảy ra dựa theo sự tuần hoàn của thế giới tự nhiên hoặc của thói quen của xã hội, văn hóa, như thời tiết, kỳ nghỉ lễ, năm học,...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/seasonality.jpeg" style="width: 1000px;"/>

Chu kỳ - Cycle thể hiện sự dao động của chuỗi thời gian xảy ra theo khoảng thời gian dài hơn và không nhất thiết lặp lại theo chu kỳ cố định, thường liên quan đến các yếu tố kinh tế vĩ mô, như suy thoái và phục hồi kinh tế.

Thời gian chu kỳ thường dài (vài năm) như chu kỳ kinh tế kéo dài 5–10 năm với các giai đoạn mở rộng và suy thoái.

<!-- Độ dừng - Stationarity
Periodogram
Tính phụ thuộc nối tiếp - Serial dependence trong Time series
https://www.kaggle.com/code/ryanholbrook/time-series-as-features -->


## 3. Time-step features và Lag features

### 3.1. Time-step features

Time-step features (đặc trưng thời điểm) là đặc trưng được trích xuất trực tiếp từ thông tin thời gian (timestamp) của chuỗi thời gian.
Chúng mô tả vị trí thời gian của mỗi quan sát (observation) mà không sử dụng giá trị lịch sử.
Time-step features cung cấp thông tin ngữ cảnh thời gian cho mỗi điểm dữ liệu.

Ví dụ: Xét chuỗi thời gian về nhiệt độ trung bình hàng tháng trong một năm, ta có thể suy luận được rằng nhiệt độ vào tháng 7 sẽ cao hơn tháng 1 mà không cần quan tâm đến giá trị nhiệt độ của các tháng trước đó.

### 3.2. Lag features

Lag features (đặc trưng trễ) là đặc trưng mà ta có thể lấy được từ những điểm dữ liệu trong quá khứ.
Lag features có thể được xây dựng bằng việc dịch chuyển các quan sát của dữ liệu đi một số mốc thời gian nhất định.
Nếu ta dịch chuyển các quan sát của dữ liệu đi một mốc thời gian, ta được 1-step lag feature, tương tự với n-step lag feature.

Lag features là thể hiện tính phụ thuộc của dữ liệu vào dữ liệu quá khứ, còn gọi là serial dependence.
Cụ thể hơn, đối với lag feature, ta không quan trọng về mặt tuyệt đối của thời gian, ta chỉ quan tâm về tính tương đối giữa các dữ liệu xảy ra trước và các dữ liệu xảy ra sau.

## 4. Feature engineering xử lý đặc trưng trong Time series

### 4.1. Chia bộ dữ liệu train, validation, test

Với các bài toán mà mỗi điểm dữ liệu được xem xét là độc lập với một điểm dữ liệu khác, ta có thể chia bộ dữ liệu thành các tập train, validation, test một cách hoàn toàn ngẫu nhiên.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/train_val_test.jpeg" style="width: 500px;"/>

Tuy nhiên, với các bài toán Time series, ta cần phải chia bộ dữ liệu theo thứ tự thời gian, để đảm bảo rằng các điểm dữ liệu trong tập validation và test không được sử dụng trong quá trình huấn luyện mô hình.

Ví dụ: Xét bộ dữ liệu về doanh thu hàng tháng trong 5 năm từ tháng 1 năm 2018 đến tháng 12 năm 2022, tức là ta có 60 điểm dữ liệu. Ta có thể chia bộ dữ liệu thành các tập như sau:
- **Tập train:** từ tháng 1 năm 2018 đến tháng 12 năm 2020 (36 điểm dữ liệu)
- **Tập validation:** từ tháng 1 năm 2021 đến tháng 12 năm 2021 (12 điểm dữ liệu)
- **Tập test:** từ tháng 1 năm 2022 đến tháng 12 năm 2022 (12 điểm dữ liệu)

### 4.2. Chuẩn hoá (normalization) và tiêu chuẩn hoá (standardization) dữ liệu

Chuẩn hoá (normalization) và tiêu chuẩn hoá (standardization) dữ liệu là các bước quan trọng trong quá trình tiền xử lý dữ liệu nói chung và dữ liệu time series nói riêng phục vụ cho machine learning, đặc biệt đối với các mô hình nhạy cảm với khoảng cách và độ lớn của dữ liệu.

Có hai vấn đề của dữ liệu khiến việc chuẩn hoá và tiêu chuẩn hoá trở nên cần thiết:
- **Độ lớn tuyệt đối của dữ liệu:** 
Nhiều thuật toán học máy dựa vào gradient phụ thuộc vào độ lớn tuyệt đối của dữ liệu.
Khi dữ liệu có giá trị quá lớn có thể gây ra hiện tượng tràn số (overflow) hoặc exploding gradients.
Khi dữ liệu có giá trị quá nhỏ có thể dẫn đến vanishing gradients.
- **Thang đo khác nhau của các đặc trưng:**
Khi các đặc trưng có thang đo khác nhau, thuật toán có thể hội tụ chậm hoặc không hội tụ do gradient bị lệch về một đặc trưng nhất định có giá trị lớn hơn.
Ta đưa tất cả các đặc trưng về cùng một thang đo, từ đó cải thiện hiệu quả của quá trình huấn luyện mô hình.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/normalization_standardization.jpeg" style="width: 600px;"/>

Chuẩn hoá dữ liệu (normalization) là quá trình biến đổi dữ liệu về một khoảng giá trị nhất định, thường là từ 0 đến 1.
Chuẩn hoá thường được sử dụng khi dữ liệu không có giả định phân phối chuẩn; dữ liệu rải rác, phi tuyến tính, hoặc có các giá trị ngoại lai (outliers).

Tiêu chuẩn hoá dữ liệu (standardization) là quá trình biến đổi dữ liệu sao cho có trung bình bằng 0 và độ lệch chuẩn bằng 1, hay nói cách khác là đưa dữ liệu về phân phối chuẩn.
Tiêu chuẩn hoá thường được sử dụng khi dữ liệu có giả định phân phối chuẩn; dữ liệu tuyến tính, có phân phối đồng nhất, hoặc không có giá trị ngoại lai.

### 4.3. Tạo đặc trưng trễ - Lag features

Để huấn luyện mô hình dự báo chuỗi thời gian, ta thường chuyển đổi chuỗi thời gian thành dạng giám sát bằng cách tạo các đặc trưng là các giá trị trễ (lag features) của chuỗi.
Khi dữ liệu đã được biến đổi như vậy, bất cứ mô hình hồi quy nào cũng có thể được huấn luyện để dự đoán bước tiếp theo

Nếu dự báo giá trị tại thời điểm $t+1$, ta có thể dùng giá trị tại các bước trước $t, t-1, ..., t-p$ làm các biến đầu vào.
Trong đó, $p$ là số bước trễ mà ta muốn sử dụng làm đặc trưng, thường được gọi là window size.

Ví dụ: Xét bộ dữ liệu về doanh thu hàng tháng trong 5 năm từ tháng 1 năm 2018 đến tháng 12 năm 2022, tức là ta có 60 điểm dữ liệu.
Ta xây dựng các đặc trưng trễ với $p=10$ như sau:

| Dữ liệu thứ | Dữ liệu input                    | Nhãn mục tiêu           |
|-------------|----------------------------------|-------------------------|
| 1           | doanh thu từ 01/2018 đến 10/2018 | doanh thu 11/2018       |
| 2           | doanh thu từ 02/2018 đến 11/2018 | doanh thu 12/2018       |
| 3           | doanh thu từ 03/2018 đến 12/2018 | doanh thu 01/2019       |
| 4           | doanh thu từ 04/2018 đến 01/2019 | doanh thu 02/2019       |
| 5           | doanh thu từ 05/2018 đến 02/2019 | doanh thu 03/2019       |
| 6           | doanh thu từ 06/2018 đến 03/2019 | doanh thu 04/2019       |
| 7           | doanh thu từ 07/2018 đến 04/2019 | doanh thu 05/2019       |
| 8           | doanh thu từ 08/2018 đến 05/2019 | doanh thu 06/2019       |
| 9           | doanh thu từ 09/2018 đến 06/2019 | doanh thu 07/2019       |
| 10          | doanh thu từ 10/2018 đến 07/2019 | doanh thu 08/2019       |
| ...         | ...                              | ...                     |

### 4.4. Tạo đặc trưng thời điểm - Time-step features

Ngoài các thông tin về vị trí tương đối giữa các quan sát trong chuỗi thời gian từ đặc trưng trễ, ta cũng cần các thông tin về vị trí tuyệt đối của các quan sát trong chuỗi thời gian thông qua các đặc trưng thời điểm (time-step features).

Ta có thể thêm thông tin chu kỳ như giờ-trong-ngày, ngày-trong-tuần, tháng-trong-năm… để mô hình học được tính lặp lại theo thời gian.
Kỹ thuật này giúp nắm bắt các quy luật theo thời gian một cách hiệu quả.

Các đặc trưng thời điểm có thể thể hiện tính thời vụ (seasonality) của chuỗi thời gian, giúp mô hình nhận biết được các chu kỳ lặp lại trong dữ liệu.

#### 4.4.1. Seasonal plot và seasonal indicators

Seasonal plot chia chuỗi thời gian thành khoảng thời gian nhất định, mỗi khoảng thời gian là một chu kỳ nhất định.

Ví dụ: Đối với dữ liệu doanh thu theo ngày của một nhà hàng trong năm 2023, ta có thể chia chuỗi thời gian thành các khoảng 7 ngày (theo tuần) để quan sát sự thay đổi doanh thu trong từng ngày trong tuần.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/seasonal_time_series.jpeg" style="width: 1000px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/seasonal_plot.jpeg" style="width: 1000px;"/>

Seasonal indicators mã hoá các mốc thời gian trong time series về dạng one-hot theo chu kỳ thời vụ mà ta lựa chọn.
Seasonal indicators phù hợp với những thời vụ ngắn và có ít quan sát (ví dụ như tính thời vụ theo tuần với các quan sát dữ liệu hàng ngày).

Ví dụ: Đối với dữ liệu doanh thu theo ngày của một nhà hàng trong năm 2023, sau khi ta chia chuỗi thời gian thành các khoảng 7 ngày (theo tuần), ta có thể mã hoá các ngày trong tuần thành các vector one-hot.
- Ngày 1 tháng 1 năm 2023 là chủ nhật, ta mã hoá thành [0, 0, 0, 0, 0, 0, 1]
- Ngày 2 tháng 1 năm 2023 là thứ hai, ta mã hoá thành [1, 0, 0, 0, 0, 0, 0]
- Ngày 3 tháng 1 năm 2023 là thứ ba, ta mã hoá thành [0, 1, 0, 0, 0, 0, 0]
- Ngày 4 tháng 1 năm 2023 là thứ tư, ta mã hoá thành [0, 0, 1, 0, 0, 0, 0]
- Ngày 5 tháng 1 năm 2023 là thứ năm, ta mã hoá thành [0, 0, 0, 1, 0, 0, 0, 0]
- Ngày 6 tháng 1 năm 2023 là thứ sáu, ta mã hoá thành [0, 0, 0, 0, 1, 0, 0]
- Ngày 7 tháng 1 năm 2023 là thứ bảy, ta mã hoá thành [0, 0, 0, 0, 0, 1, 0]
- Ngày 8 tháng 1 năm 2023 là chủ nhật, ta mã hoá thành [0, 0, 0, 0, 0, 0, 1]
- ...

Với việc mã hoá seasonal indicators, ta có thể mô phỏng được tính thời vụ dưới dạng hàm số.

#### 2.4.2. Fourier features

Fourier features phù hợp với những thời vụ dài và có nhiều quan sát (ví dụ như tính thời vụ theo năm với các quan sát dữ liệu hàng ngày).
Fourier features hướng đến việc mô phỏng lại hình dáng của tính thời vụ trong time series thông qua cặp hai đường cong sin và cos.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/fourier_basis.jpeg" style="width: 1000px;"/>

Trong thực tế, để tính toán Fourier features, ta sẽ khởi tạo một số cặp đường cong sin và cos nhất định, sau đó sử dụng Linear Regression để tính toán ra các trọng số của đường cong sin và cos của mỗi cặp sao cho ta có thể mô phỏng được gần nhất đường cong thể hiện tính thời vụ của time series.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/fourier_approximation.jpeg" style="width: 1000px;"/>

### 4.5. Tạo đặc trưng thể hiện xu hướng

Để dễ dàng quan sát được xu hướng tăng hay giảm, ta thường sử dụng Moving average.
Moving average là cách tính trung bình của dữ liệu chuỗi thời gian trong một khoảng thời gian lân cận.

Moving average giúp giảm bớt được sự dao động của dữ liệu trong quãng thời gian ngắn, tạo điều kiện giúp ta quan sát xu hướng trong quãng thời gian dài chính xác và dễ dàng hơn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/moving_average.gif" style="width: 1000px;"/>

## 5. Các nhóm mô hình dự báo chuỗi thời gian

Các mô hình dự báo chuỗi thời gian có thể được chia thành một số nhóm chính như sau:
- **Mô hình thống kê**: Sử dụng các phương pháp thống kê truyền thống để phân tích và dự báo chuỗi thời gian, như ARIMA, SARIMA, Exponential Smoothing.
- **Mô hình học máy**: Sử dụng các thuật toán học máy để học từ dữ liệu quá khứ và dự báo tương lai, như Random Forest, Gradient Boosting, Support Vector Regression.
- **Mô hình học sâu**: Sử dụng các kiến trúc mạng nơ ron để học từ dữ liệu chuỗi thời gian, như LSTM, GRU, Temporal Convolutional Networks (TCN) hay mới đây là Transformer.
- **Mô hình lai**: Kết hợp các mô hình thống kê và học máy để tận dụng ưu điểm của cả hai, như Prophet của Facebook.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/8_time_series/images/1-time-series/models.jpeg" style="width: 800px;"/>
