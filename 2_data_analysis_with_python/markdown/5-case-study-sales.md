---
time: 03/15/2022
title: "[CASE STUDY] Thực hành phân tích bộ dữ liệu 1C Company Sales"
description: Bài thực hành này được thực hiện trên bộ dữ liệu này được lấy từ cuộc thi Predict Future Sales trên Kaggle và qua một số chỉnh sửa để phù hợp với mục đích phân tích và thực hành sử dụng thư viện Pandas.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/2-pandas/banner.png
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

Bộ dữ liệu này được lấy từ cuộc thi [Predict Future Sales](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales) trên Kaggle và qua một số chỉnh sửa để phù hợp với mục đích phân tích và thực hành sử dụng thư viện Pandas.

```bibtex
@misc{competitive-data-science-predict-future-sales,
    author = {Alexander Guschin and Dmitry Ulyanov and inversion and Mikhail Trofimov and utility and Μαριος Μιχαηλιδης KazAnova},
    title = {Predict Future Sales},
    year = {2018},
    howpublished = {\url{https://kaggle.com/competitions/competitive-data-science-predict-future-sales}},
    note = {Kaggle}
}
```

Bạn được cung cấp dữ liệu bán hàng lịch sử hàng ngày. Bộ dữ liệu gồm 5 tệp `.csv`, mô tả tệp và mô tả các trường dữ liệu như sau:
- **Mô tả tệp**:
    - `sales.csv`: Dữ liệu lịch sử bán hàng hằng ngày từ tháng 01/2013 đến tháng 10/2015.
    - `items_list_1.csv`: Thông tin cần thiết về các sản phẩm trong danh sách 1.
    - `items_list_2.csv`: Thông tin cần thiết về các sản phẩm trong danh sách 2.
    - `item_categories.csv`: Thông tin cần thiết về các danh mục sản phẩm.
    - `shops.csv`: Thông tin cần thiết về các cửa hàng.
- **Mô tả các trường dữ liệu**:
    - `shop_id`: Mã định danh duy nhất của cửa hàng.
    - `item_id`: Mã định danh duy nhất của sản phẩm.
    - `item_category_id`: Mã định danh duy nhất của danh mục sản phẩm.
    - `item_cnt_day`: Số lượng sản phẩm bán ra trong ngày.
    - `item_price`: Giá hiện tại của sản phẩm.
    - `date`: Ngày bán theo định dạng dd/mm/yyyy.
    - `date_block_num`: Số thứ tự tháng liên tiếp, dùng để tiện phân tích. Tháng 01/2013 là 0, tháng 02/2013 là 1, ..., tháng 10/2015 là 33.
    - `item_name`: Tên sản phẩm.
    - `shop_name`: Tên cửa hàng.
    - `item_category_name`: Tên danh mục sản phẩm.

## 3. Câu hỏi phân tích

### 3.1. Câu 1:

- Có bao nhiêu sản phẩm:
    - trong danh sách 1?
    - trong danh sách 2?
    - trong chỉ danh sách 1 (Liệt kê tên các sản phẩm đó)?
    - trong chỉ danh sách 2 (Liệt kê tên các sản phẩm đó)?
    - trong cả hai danh sách (Liệt kê tên các sản phẩm đó)?
- Tạo tệp csv mới chỉ chứa các sản phẩm duy nhất từ hai danh sách (Đặt tên tệp là items.csv).

### 3.2. Câu 2:

- Có bao nhiêu sản phẩm trong tệp items.csv?
- Trong số đó có bao nhiêu sản phẩm chứa chữ số trong tên?
- Có bao nhiêu sản phẩm là trò chơi bóng đá FIFA (tên chứa "FIFA")?

### 3.3. Câu 3:

- Có bao nhiêu danh mục sản phẩm trong bộ dữ liệu?
- Danh mục nào chứa nhiều sản phẩm nhất?
- Danh mục nào chứa ít sản phẩm nhất?
- Liệt kê toàn bộ sản phẩm theo từng danh mục.
- Tính số lượng trung bình sản phẩm trong mỗi danh mục.

### 3.4. Câu 4:

- Sản phẩm nào có giá cao nhất trong mỗi năm?
- Sản phẩm nào có giá thấp nhất trong mỗi năm?
- Tính giá trung bình của từng sản phẩm trong mỗi năm.

### 3.5. Câu 5:

- Sản phẩm nào có doanh số cao nhất trong mỗi năm?
- Sản phẩm nào có doanh số thấp nhất trong mỗi năm?
- Tính doanh số trung bình của từng sản phẩm trong mỗi năm.

## 4. Lời giải tham khảo

Bạn có thể tham khảo lời giải chi tiết trong [notebook này](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/2_data_analysis_with_python/notebook/5-case-study-sales/sales.ipynb).
