---
time: 03/22/2022
title: "[CASE STUDY] Phân tích và dự báo với bộ dữ liệu Credit Scoring"
description: Bài thực hành này được thực hiện trên bộ dữ liệu này được lấy từ cuộc thi Give Me Some Credit trên Kaggle và qua một số chỉnh sửa để phù hợp với mục đích phân tích và thực hành sử dụng thư viện Pandas.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/2_data_analysis_with_python/images/2-pandas/banner.jpeg
tags: [python, data-analysis, case-study]
is_highlight: false
is_published: true
---

## 1. Giới thiệu về bài toán

Ngân hàng đóng vai trò then chốt trong các nền kinh tế thị trường. Họ quyết định ai có thể được cấp tín dụng, với các điều kiện như thế nào, và có thể làm nên hoặc phá vỡ các quyết định đầu tư. Để thị trường và xã hội vận hành, cả cá nhân và doanh nghiệp đều cần có khả năng tiếp cận tín dụng.

Các thuật toán chấm điểm tín dụng, vốn ước lượng xác suất vỡ nợ, là phương pháp mà ngân hàng sử dụng để xác định liệu một khoản vay có nên được chấp thuận hay không. Cuộc thi này yêu cầu người tham gia cải thiện những phương pháp hiện có trong chấm điểm tín dụng, bằng cách dự đoán xác suất một cá nhân sẽ gặp khó khăn tài chính trong vòng hai năm tới.

Mục tiêu của cuộc thi là xây dựng một mô hình mà người vay có thể sử dụng để đưa ra những quyết định tài chính tốt nhất.

## 2. Mô tả bộ dữ liệu

Bộ dữ liệu này được lấy từ cuộc thi [Give Me Some Credit](https://www.kaggle.com/competitions/give-me-some-credit) trên Kaggle và qua một số chỉnh sửa để phù hợp với mục đích phân tích và thực hành sử dụng thư viện Pandas.

```bibtex
@misc{give-me-some-credit,
    author = {Daniel Ortiz},
    title = {Give Me Some Credit},
    year = {2021},
    howpublished = {\url{https://kaggle.com/competitions/give-me-some-credit}},
    note = {Kaggle}
}
```

Bộ dữ liệu gồm 2 tệp `.csv`, tệp `train.csv` chứa dữ liệu huấn luyện và tệp `test.csv` chứa dữ liệu kiểm tra. Mỗi tệp gồm các trường dữ liệu như sau:
- `SeriousDlqin2yrs`: Người vay từng bị quá hạn thanh toán từ 90 ngày trở lên hoặc tệ hơn. **Giá trị: Có/Không (Y/N)**.
- `RevolvingUtilizationOfUnsecuredLines`: Tổng số dư trên thẻ tín dụng và các hạn mức tín dụng cá nhân (không bao gồm bất động sản và các khoản trả góp) **Đơn vị: tỷ lệ phần trăm**
- `age`: Tuổi của người vay (tính theo năm) **Kiểu dữ liệu: số nguyên**
- `NumberOfTime30-59DaysPastDueNotWorse`: Số lần người vay bị quá hạn thanh toán từ 30–59 ngày nhưng không nghiêm trọng hơn trong vòng 2 năm qua **Kiểu dữ liệu: số nguyên**
- `DebtRatio`: Tỷ lệ nợ: tổng chi phí trả nợ hàng tháng, tiền cấp dưỡng, chi phí sinh hoạt chia cho tổng thu nhập gộp hàng tháng **Đơn vị: tỷ lệ phần trăm**
- `MonthlyIncome`: Thu nhập hàng tháng **Kiểu dữ liệu: số thực**
- `NumberOfOpenCreditLinesAndLoans`: Số lượng khoản vay trả góp đang mở (ví dụ: vay mua xe, vay thế chấp) và số lượng hạn mức tín dụng đang mở (ví dụ: thẻ tín dụng) **Kiểu dữ liệu: số nguyên**
- `NumberOfTimes90DaysLate`: Số lần người vay bị quá hạn từ 90 ngày trở lên **Kiểu dữ liệu: số nguyên**
- `NumberRealEstateLoansOrLines`: Số khoản vay bất động sản và vay thế chấp, bao gồm cả hạn mức tín dụng thế chấp nhà **Kiểu dữ liệu: số nguyên**
- `NumberOfTime60-89DaysPastDueNotWorse`: Số lần người vay bị quá hạn từ 60–89 ngày nhưng không nghiêm trọng hơn trong vòng 2 năm qua **Kiểu dữ liệu: số nguyên**
- `NumberOfDependents`: Số người phụ thuộc trong gia đình (không tính bản thân người vay), bao gồm vợ/chồng, con cái, v.v. **Kiểu dữ liệu: số nguyên**

## 3. Câu hỏi phân tích

## 4. Lời giải tham khảo

Bạn có thể tham khảo lời giải chi tiết trong [notebook này](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/2_data_analysis_with_python/notebook/8-case-study-credit-scoring/credit_scoring.ipynb).
