---
slug: time-series
time: 11/09/2024
title: Giới thiệu chung về Chuỗi thời gian - Time Series
description:
author: Nguyễn Hữu Minh
banner_url: 
tags: [machine-learning, deep-learning]
is_highlight: false
is_published: false
---

# Time series

Dự báo là bài toán phổ biến mang tính ứng dụng cao của machine learning.
Bài toán dự báo với dữ liệu Time series là lĩnh vực rộng với lịch sử lâu đời.

## 1. Giới thiệu chung về Time series

Đối tượng cơ bản trong quá trình dự báo là chuỗi thời gian, là tập hợp các quan sát dữ liệu được thu thập theo thời gian.
Cụ thể, các quan sát thường được thu thập theo một tần suất thời gian như hàng ngày, hàng giờ hay hàng tháng ...

### 1.1. Timestamps và periods trong Time series

Timestamps được sử dụng để đại diện cho một mốc thời gian.
Periods đại diện cho khoảng thời gian.
Periods được sử dụng để kiểm tra xem một sự kiện nào đó có được diễn ra trong một chu kỳ cho trước hay không.
Timestamps và periods có thể được chuyển đổi qua lại với nhau.

### 1.2. Resampling dữ liệu Time series

Có hai cách để lấy mẫu dữ liệu thời gian trong Time series, hai cách lấy mẫu dữ liệu: Upsampling và Downsampling
- Upsampling: là cách lấy mẫu tăng tần số, từ tần số thấp lên tần số cao (ví dụ từ dữ liệu hàng tháng, lấy mẫu thành dữ liệu hàng ngày). Phương pháp này đòi hỏi ta phải có cách để làm đầy (nội suy) những khoảng giá trị bị khuyết thiếu.
- Downsampling: là cách lấy mẫu giảm tần số, từ tần số cao xuống tần số thấp (ví dụ từ dữ liệu hàng ngày, lấy mẫu thành dữ liệu hàng tháng). Phương pháp này đòi hỏi ta phải có cách để tổng hợp dữ liệu.

## 2. Đặc trưng dữ liệu trong Time series

Có hai kiểu đặc trưng dữ liệu trong Time series là time-step features and lag features.

### 2.1. Time-step features

Time-step features là những đặc trưng mà ta có thể lấy được trực tiếp từ thời gian.

Time-step feature cơ bản nhất là time dummy, đánh dấu mỗi mốc thời gian trên bộ dữ liệu là một time index lần lượt từ đầu đến hết các mốc.

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2122706&authkey=%21ABaYv_AV1wRdTSQ" style="width: 1000px;"/>

Time-step features thể hiện tính phụ thuộc của dữ liệu vào thời gian.
Hay nói cách khác, chuỗi dữ liệu phụ thuộc vào thời gian có nghĩa là giá trị của dữ liệu có thể được dự đoán từ thời điểm mà dữ liệu đó xảy ra.

Hai time-step features được sử dụng nhiều là tính xu hướng - trend và tính thời vụ - seasonality.

### 2.2. Lag features

Lag features là những đặc trưng mà ta có thể lấy được từ dữ liệu trong quá khứ.

Lag features có thể được xây dựng bằng việc dịch chuyển các quan sát của dữ liệu đi một số mốc thời gian nhất định.
Nếu ta dịch chuyển các quan sát của dữ liệu đi một mốc thời gian, ta được 1-step lag feature, tương tự với n-step lag feature.

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2122707&authkey=%21AGdDhbwaY885KHc" style="width: 500px;"/>

Lag features là thể hiện tính phụ thuộc của dữ liệu vào dữ liệu quá khứ, còn gọi là serial dependence.
Cụ thể hơn, đối với lag feature, ta không quan trọng về mặt tuyệt đối của thời gian, ta chỉ quan tâm về tính tương đối giữa các dữ liệu xảy ra trước và các dữ liệu xảy ra sau.

Tóm lại, việc chuẩn bị dữ liệu để giải quyết bài toán Time series là việc chúng ta feature engineering giữa Time-step features (đặc trưng phụ thuộc thời gian) và Lag features (đặc trưng phụ thuộc vào chuỗi dữ liệu trong quá khứ).

### 2.3. Tính xu hướng - Trend trong Time series

Tính xu hướng - trend đại diện cho sự thay đổi ổn định kéo dài trong chuỗi.
Sự thay đổi ở đây có thể là tăng dần hoặc giảm dần theo thời gian.

<img src="https://storage.googleapis.com/kaggle-media/learn/images/KFYlgGm.png" style="width: 1000px;"/>

Để dễ dàng quan sát được xu hướng tăng hay giảm, ta thường sử dụng moving average.
Moving average là cách tính trung bình của dữ liệu chuỗi thời gian trong một khoảng thời gian lân cận.

Moving average giúp giảm bớt được sự dao động của dữ liệu trong quãng thời gian ngắn, tạo điều kiện giúp ta quan sát xu hướng trong quãng thời gian dài chính xác và dễ dàng hơn.

<img src="https://storage.googleapis.com/kaggle-media/learn/images/EZOXiPs.gif" style="width: 1000px;"/>

### 2.4. Tính thời vụ - Seasonality trong Time series

Tính thời vụ - seasonality đại diện cho sự lặp đi lặp lại có chu kỳ của một đặc điểm nào đó của chuỗi thời gian tại một thời điểm nào đó nhất định trong dòng thời gian.
Tính thời vụ thường xảy ra dựa theo sự tuần hoàn của thế giới tự nhiên hoặc của thói quen của xã hội.

<img src="https://storage.googleapis.com/kaggle-media/learn/images/ViYbSxS.png" style="width: 1200px;"/>

Có hai loại đặc trưng của tính thời vụ: Indicators và Fourier features

#### 2.4.1. Seasonal plot và seasonal indicators

Tương tự với Moving average plot được sử dụng để giúp quan sát xu hướng của time series, seasonal plot được sử dụng để giúp quan sát tính thời vụ của time series.

Seasonal plot chia time series thành các phân đoạn theo chu kỳ thời gian nhất định, mỗi chu kỳ là một thời vụ mà ta muốn quan sát.

<img src="https://storage.googleapis.com/kaggle-media/learn/images/bd7D4NJ.png" style="width: 1000px;"/>

Seasonal indicators mã hoá các mốc thời gian trong time series về dạng one-hot theo chu kỳ thời vụ mà ta lựa chọn.

Ví dụ, ta có các mốc thời gian là các ngày trong khoảng từ ngày 1 tháng 5 năm 2023 đến 14 tháng 5 năm 2023.
Từ đó, ta tạo ra các seasonal indicators theo chu kỳ hàng tuần bằng việc mã hoá mỗi mốc thời gian bằng một vector one-hot có độ dài là 7 phần tử, trong đó:
- ngày 1 tháng 5 năm 2023 là [1, 0, 0, 0, 0, 0, 0]
- ngày 2 tháng 5 năm 2023 là [0, 1, 0, 0, 0, 0, 0]
- ngày 3 tháng 5 năm 2023 là [0, 0, 1, 0, 0, 0, 0]
- ngày 4 tháng 5 năm 2023 là [0, 0, 0, 1, 0, 0, 0]
- ngày 5 tháng 5 năm 2023 là [0, 0, 0, 0, 1, 0, 0]
- ngày 6 tháng 5 năm 2023 là [0, 0, 0, 0, 0, 1, 0]
- ngày 7 tháng 5 năm 2023 là [0, 0, 0, 0, 0, 0, 1]
- ...
- ngày 13 tháng 5 năm 2023 là [0, 0, 0, 0, 0, 1, 0]
- ngày 14 tháng 5 năm 2023 là [0, 0, 0, 0, 0, 0, 1]

<img src="https://storage.googleapis.com/kaggle-media/learn/images/hIlF5j5.png" style="width: 1000px;"/>

Với việc mã hoá seasonal indicators, ta có thể mô phỏng được tính thời vụ dưới dạng hàm số.

Seasonal indicators phù hợp với những thời vụ ngắn và có ít quan sát (ví dụ như tính thời vụ theo tuần với các quan sát dữ liệu hàng ngày).

#### 2.4.2. Fourier features

Fourier features phù hợp với những thời vụ dài và có nhiều quan sát (ví dụ như tính thời vụ theo năm với các quan sát dữ liệu hàng ngày).
Mã hoá và tạo ra các đặc trưng cho từng quan sát (từng ngày) sẽ khiến cho mô hình cần nhiều tham số để mô phỏng được tính thời vụ.
Thay vì vậy, fourier features hướng đến việc mô phỏng lại hình dáng của tính thời vụ trong time series thông qua một vài đặc trưng.

<img src="https://storage.googleapis.com/kaggle-media/learn/images/NJcaEdI.png" style="width: 1000px;"/>

Fourier features là cặp hai đường cong sin và cos. Hình dưới là biểu đồ ví dụ cho hai cặp fourier features, hình trên là 1 chu kỳ mỗi năm, hình dưới là 2 chu kỳ mỗi năm (A-DEC là "anually end at Dec").

<img src="https://storage.googleapis.com/kaggle-media/learn/images/bKOjdU7.png" style="width: 700px;"/>

Trong thực tế, ta sẽ khởi tạo một số cặp fourier features nhất định, sau đó sử dụng Linear Regression để tính toán ra các trọng số của đường cong sin và cos của mỗi cặp sao cho ta có thể mô phỏng được gần nhất đường cong thể hiện tính thời vụ của time series.

<img src="https://storage.googleapis.com/kaggle-media/learn/images/mijPhko.png" style="width: 900px;"/>

<!-- Periodogram -->

<!-- ## 5. Tính phụ thuộc nối tiếp - Serial dependence trong Time series

https://www.kaggle.com/code/ryanholbrook/time-series-as-features -->
