---
time: 03/18/2022
title: "[CASE STUDY] Thực hành phân tích bộ dữ liệu M5"
description: Bài thực hành này được thực hiện trên bộ dữ liệu được lấy từ cuộc thi M5 Forecasting - Accuracy trên Kaggle.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/2-pandas/banner.jpeg
tags: [python, data-analysis, case-study]
is_highlight: false
is_published: true
---

## 1. Giới thiệu về bài toán

Bài toán phân tích dữ liệu bán hàng là một bài toán quan trọng trong việc ứng dụng phân tích dữ liệu trong kinh doanh.
Bài toán này giúp các doanh nghiệp hiểu rõ hơn về hoạt động bán hàng của mình, từ đó đưa ra các chiến lược kinh doanh phù hợp.

Theo một báo cáo từ McKinsey & Company:
"Việc sử dụng phân tích dữ liệu để tối ưu hóa hoạt động bán hàng có thể giúp các doanh nghiệp tăng doanh thu lên đến 20% và giảm chi phí vận hành lên đến 30%. Các công ty hàng đầu trong việc ứng dụng phân tích dữ liệu bán hàng thường có hiệu suất kinh doanh vượt trội so với các đối thủ cạnh tranh."

## 2. Mô tả bộ dữ liệu

Bộ dữ liệu này được lấy từ cuộc thi [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) trên Kaggle.

```bibtex
@misc{m5-forecasting-accuracy,
    author = {Addison Howard and inversion and Spyros Makridakis and vangelis},
    title = {M5 Forecasting - Accuracy},
    year = {2020},
    howpublished = {\url{https://kaggle.com/competitions/m5-forecasting-accuracy}},
    note = {Kaggle}
}
```

Bạn được cung cấp bộ dữ liệu gồm 3 tệp `csv`, mô tả tệp và mô tả các trường dữ liệu như sau:
- **Mô tả tệp**:
    - `sales.csv`: Chứa dữ liệu lịch sử về số lượng sản phẩm bán ra hằng ngày theo từng sản phẩm và cửa hàng từ ngày D1 đến D1913.
    - `calendar.csv`: Chứa thông tin về các ngày mà sản phẩm được bán.
    - `sell_prices.csv`: Chứa thông tin về giá bán của sản phẩm theo từng cửa hàng và từng ngày.
- **Mô tả các trường dữ liệu**:
    - `sales.csv`:
        - `id`: Mã định danh duy nhất của chuỗi doanh số, ghép từ mã sản phẩm và mã cửa hàng.
        - `item_id`: Mã sản phẩm.
        - `dept_id`: Mã ngành hàng con (department) của sản phẩm.
        - `cat_id`: Mã nhóm hàng (category) của sản phẩm (`FOODS`, `HOBBIES`, `HOUSEHOLD`).
        - `store_id`: Mã cửa hàng.
        - `state_id`: Mã bang nơi cửa hàng đặt trụ sở (`CA`, `TX`, `WI`).
        - `d_1`, `d_2`, ..., `d_1913`: Số lượng sản phẩm bán ra trong từng ngày, tính từ ngày D1 đến D1913.
    - `calendar.csv`:
        - `date`: Ngày theo định dạng yyyy-mm-dd.
        - `wm_yr_wk`: Mã tuần (dùng để nối với `sell_prices.csv`).
        - `weekday`: Tên thứ trong tuần.
        - `wday`: Thứ tự ngày trong tuần (1 là thứ Bảy).
        - `month`: Tháng.
        - `year`: Năm.
        - `d`: Mã ngày (`d_1`, `d_2`, ...) tương ứng với các cột của `sales.csv`.
        - `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2`: Tên và loại sự kiện (nếu có) diễn ra trong ngày. Một ngày có thể có tối đa hai sự kiện.
        - `snap_CA`, `snap_TX`, `snap_WI`: Cho biết ngày đó cửa hàng ở bang tương ứng có cho phép mua hàng bằng trợ cấp SNAP hay không (1: có, 0: không).
    - `sell_prices.csv`:
        - `store_id`: Mã cửa hàng.
        - `item_id`: Mã sản phẩm.
        - `wm_yr_wk`: Mã tuần.
        - `sell_price`: Giá bán của sản phẩm tại cửa hàng trong tuần đó.
## 3. Câu hỏi phân tích
### 3.1. Câu 1: Dữ liệu thiếu và dữ liệu bằng 0
- Bộ dữ liệu `sales.csv` có giá trị thiếu hay không?
- Với mỗi chuỗi doanh số, tỷ lệ ngày không có lượt bán nào là bao nhiêu? Tỷ lệ này là bao nhiêu nếu chỉ tính từ ngày sản phẩm bắt đầu có doanh số?
### 3.2. Câu 2: Chuỗi thời gian doanh số
- Vẽ chuỗi thời gian tổng doanh số và doanh số trung bình trên toàn bộ sản phẩm.
- Vẽ chuỗi doanh số của một sản phẩm bất kỳ.
- Vẽ chuỗi doanh số theo từng bang, từng nhóm hàng, từng cửa hàng và từng ngành hàng con, sau đó tổng hợp theo tháng.
- Nhóm hàng nào bán nhiều nhất và ít nhất?
### 3.3. Câu 3: Tính mùa vụ
- Doanh số thay đổi như thế nào theo thứ trong tuần và theo tháng trong năm (heatmap) ở từng bang?
### 3.4. Câu 4: Lịch và sự kiện
- Có bao nhiêu ngày có 0, 1 hoặc 2 sự kiện?
- Mỗi loại sự kiện (`Religious`, `National`, `Cultural`, `Sporting`) xuất hiện bao nhiêu lần?
- Mỗi bang có bao nhiêu ngày cho phép dùng SNAP?
### 3.5. Câu 5: Giá bán
- Phân phối giá bán thay đổi như thế nào giữa các bang, nhóm hàng và ngành hàng con?
- Phân phối giá bán thay đổi như thế nào theo năm?
### 3.6. Câu 6: Mối liên hệ giữa doanh số, sự kiện và SNAP
- Doanh số theo nhóm hàng và theo bang vào ngày có sự kiện và ngày không có sự kiện khác nhau thế nào?
- Doanh số theo bang vào ngày cho phép dùng SNAP và ngày không cho phép khác nhau thế nào?
## 4. Lời giải tham khảo

Bạn có thể tham khảo lời giải chi tiết trong [notebook này](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/2_data_analysis_with_python/notebook/6-case-study-m5/m5.ipynb).
