---
time: 03/11/2022
title: "[CASE STUDY] Phân tích Phân khúc khách hàng của ngân hàng (Bank Customer Segmentation)"
description: Hầu hết các ngân hàng có một lượng lớn khách hàng với các đặc điểm khác nhau về độ tuổi, thu nhập, giá trị, lối sống và nhiều yếu tố khác. Phân khúc khách hàng là quá trình chia tập dữ liệu khách hàng thành các nhóm cụ thể dựa trên những đặc điểm chung.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/2-pandas/banner.png
tags: [python, data-analysis, case-study]
is_highlight: false
is_published: true
---

## 1. Giới thiệu về bài toán

Bài toán Phân khúc khách hàng của ngân hàng là bài toán quan trọng trong việc ứng dụng phân tích dữ liệu trong ngành ngân hàng. Bài toán này giúp ngân hàng hiểu rõ hơn về khách hàng của mình, từ đó đưa ra các chiến lược kinh doanh phù hợp.

Theo một báo cáo từ Ernst & Young:
"Việc hiểu sâu sắc hơn về khách hàng không còn là một yếu tố tùy chọn, mà đã trở thành một yêu cầu chiến lược và mang tính cạnh tranh đối với các nhà cung cấp dịch vụ ngân hàng. Việc thấu hiểu khách hàng nên trở thành một phần thiết yếu trong hoạt động kinh doanh hàng ngày, với những thông tin chi tiết làm nền tảng cho toàn bộ hoạt động ngân hàng."

## 2. Mô tả bộ dữ liệu

Bộ dữ liệu này được lấy từ cuộc thi [Bank Customer Segmentation](https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation) trên Kaggle.

Bộ dữ liệu này bao gồm 1 tệp `.csv` chứa hơn 1 triệu giao dịch từ hơn 800.000 khách hàng của một ngân hàng tại Ấn Độ.
Mô tả các trường dữ liệu như sau:
- `TransactionID`: Mã giao dịch
- `CustomerID`: Mã khách hàng
- `CustomerDOB`: Ngày sinh khách hàng
- `CustGender`: Giới tính khách hàng
- `CustLocation`: Địa chỉ khách hàng
- `CustAccountBalance`: Số dư tài khoản khách hàng
- `TransactionDate`: Ngày giao dịch
- `TransactionTime`: Thời gian giao dịch (có dạng hh:mm:ss)
- `TransactionAmount (INR)`: Số tiền giao dịch (đơn vị: INR - Rupee Ấn Độ)

## 3. Câu hỏi phân tích

## 4. Lời giải tham khảo

Bạn có thể tham khảo lời giải chi tiết trong [notebook này](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/2_data_analysis_with_python/notebook/7-case-study-bank-transactions/bank_transactions.ipynb).
