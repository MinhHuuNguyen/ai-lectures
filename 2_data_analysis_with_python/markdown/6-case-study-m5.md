---
time: 03/18/2022
title: "[CASE STUDY] Thực hành phân tích bộ dữ liệu M5"
description: Bài thực hành này được thực hiện trên bộ dữ liệu này được lấy từ cuộc thi M5 Forecasting - Accuracy trên Kaggle.
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

## 3. Câu hỏi phân tích

## 4. Lời giải tham khảo

Bạn có thể tham khảo lời giải chi tiết trong [notebook này](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/2_data_analysis_with_python/notebook/6-case-study-m5/m5.ipynb).
