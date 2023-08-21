---
layout: "post"
title:  "Google Cloud Storage"
author: "Nguyễn Hữu Minh"
permalink: "/cloud-computing/google-cloud-platform/google-cloud-storage"
parent: "Google Cloud Platform"
grand_parent: "Cloud computing"
nav_order: 1
---

# Các dịch vụ lưu trữ và database của Google Cloud Platform

## 1. Google Cloud Storage

### 1.1. Lợi ích và tính năng chính của Google Cloud Storage

- Độ tin cậy và bảo mật:
Dữ liệu được lưu trữ trên hạ tầng đám mây bảo mật và đáng tin cậy của Google.
Bạn có thể áp dụng các cơ chế bảo mật như quản lý quyền truy cập và mã hóa dữ liệu.
- Hỗ trợ đa dạng định dạng:
Google Cloud Storage hỗ trợ nhiều định dạng dữ liệu khác nhau như văn bản, hình ảnh, âm thanh và video.
Điều này giúp bạn lưu trữ các loại dữ liệu khác nhau một cách linh hoạt.
- Lưu trữ lớn và linh hoạt:
Bạn có thể lưu trữ lượng dữ liệu lớn mà không cần lo lắng về việc mở rộng cơ sở hạ tầng.
Google Cloud Storage cho phép bạn mở rộng tài nguyên lưu trữ theo nhu cầu.
- Lưu trữ dữ liệu phân tán:
Dữ liệu có thể được lưu trữ ở nhiều vị trí khác nhau trên toàn cầu, giúp cải thiện khả năng truy cập và tốc độ tải dữ liệu.

### 1.2. Cách sử dụng Google Cloud Storage

- Tạo Bucket:
Bạn bắt đầu bằng cách tạo "bucket" - đơn vị lưu trữ cơ bản trong Google Cloud Storage.
Bucket giống như một thư mục chứa các đối tượng dữ liệu.
- Tải Lên Và Tải Xuống Dữ Liệu:
Bạn có thể tải lên và tải xuống dữ liệu vào bucket thông qua giao diện dòng lệnh hoặc các giao diện quản lý như GCP Console.
- Quản Lý Quyền Truy Cập:
Google Cloud Storage cho phép bạn quản lý quyền truy cập đối với các bucket và đối tượng.
Bạn có thể chia sẻ dữ liệu với người dùng cụ thể hoặc công khai chúng nếu cần.

### 1.3. Ứng Dụng của Google Cloud Storage

- Lưu trữ dữ liệu đa phương tiện:
Bạn có thể lưu trữ hình ảnh, video, và âm thanh trong Google Cloud Storage, rất phù hợp cho các ứng dụng liên quan đến phương tiện truyền thông.
- Dự án lớn:
Google Cloud Storage thích hợp cho các dự án lớn, như lưu trữ và phân tích dữ liệu lớn.
- Backup và khôi phục dữ liệu:
Bạn có thể sử dụng Google Cloud Storage để sao lưu dữ liệu quan trọng và khôi phục chúng khi cần.

### 1.4. Một số loại cấu hình storage

#### 1.4.1. Standard Storage:

- Loại: Standard
- Mô tả: Dịch vụ lưu trữ chuẩn với thời gian truy cập thấp và khả năng sẵn sàng cao.
- Sử dụng: Lưu trữ dữ liệu thường xuyên truy cập, dự án lớn với yêu cầu bảo mật và hiệu suất cao.

#### 1.4.2. Nearline Storage:

- Loại: Nearline
- Mô tả: Dịch vụ lưu trữ dành cho dữ liệu ít truy cập, có thời gian truy cập từ vài giây đến phút.
- Sử dụng: Lưu trữ dữ liệu lưu trữ dài hạn như backup, lưu trữ lịch sử.

#### 1.4.2. Coldline Storage:

- Loại: Coldline
- Mô tả: Dịch vụ lưu trữ dành cho dữ liệu cần bảo quản lâu dài, với thời gian truy cập từ vài phút đến giờ.
- Sử dụng: Lưu trữ dữ liệu lưu trữ rất lâu dài, như dữ liệu pháp lý.

#### 1.4.3. Archive Storage:

- Loại: Archive
- Mô tả: Dịch vụ lưu trữ dành cho dữ liệu cần lưu trữ siêu lâu dài với thời gian truy cập từ vài giờ đến một ngày.
- Sử dụng: Lưu trữ dữ liệu ít truy cập, cần bảo quản theo yêu cầu quy định hoặc pháp lý.

#### 1.4.4. Multi-Regional Storage:

- Loại: Multi-Regional
- Mô tả: Dịch vụ lưu trữ đa vùng, đảm bảo sẵn sàng cao và thời gian truy cập thấp.
- Sử dụng: Dữ liệu được truy cập toàn cầu, ứng dụng có yêu cầu sẵn sàng và hiệu suất cao.

#### 1.4.5. Regional Storage:

- Loại: Regional
- Mô tả: Dịch vụ lưu trữ trong một vùng cụ thể, với sẵn sàng cao và thời gian truy cập thấp.
- Sử dụng: Dữ liệu được truy cập trong khu vực cụ thể, giảm thời gian truy cập.

#### 1.4.6. Durable Reduced Availability Storage:

- Loại: Durable Reduced Availability (DRA)
- Mô tả: Dịch vụ lưu trữ với sự sẵn sàng hạn chế hơn, giá thấp hơn so với các loại khác.
- Sử dụng: Dữ liệu ít truy cập, giảm chi phí lưu trữ.

### 1.5. Chi phí của Google Cloud Storage

- Loại cấu hình lưu trữ:
Chi phí sẽ thay đổi tùy theo loại cấu hình lưu trữ bạn chọn, ví dụ: Standard, Nearline, Coldline, Archive, Multi-Regional, Regional, Durable Reduced Availability.
- Dung lượng lưu trữ:
Giá lưu trữ dựa trên lượng dữ liệu bạn lưu trữ.
Càng nhiều dữ liệu thì càng tăng chi phí.
- Thời gian lưu trữ:
Dữ liệu lưu trữ trong thời gian dài sẽ tạo ra chi phí lưu trữ liên quan.
- Thời gian truy cập:
Dịch vụ lưu trữ có thời gian truy cập khác nhau (ví dụ: Standard có thời gian truy cập thấp hơn Archive).
Thời gian truy cập ảnh hưởng đến giá.
- Vùng địa lý:
Giá lưu trữ có thể thay đổi tùy theo vùng địa lý bạn chọn.
Vùng địa lý sẽ ảnh hưởng đến giá lưu trữ và truy cập.
- Số lượng yêu cầu API:
Nếu bạn sử dụng API để truy cập dữ liệu từ Google Cloud Storage, có thể có phí liên quan đến số lượng yêu cầu API.
- Chuyển dữ liệu và băng thông:
Nếu bạn di chuyển dữ liệu ra/vào Google Cloud Storage hoặc sử dụng băng thông lớn, chi phí có thể tăng.
- Quản lý dữ liệu:
Có thể có chi phí phụ thuộc vào việc quản lý và xử lý dữ liệu như xóa, di chuyển, sao lưu, phục hồi.

<!-- 
## 2. Google Cloud SQL

### 2.1. Google Cloud Vision

### 2.2. Google Cloud Speech

### 2.3. Google Cloud Translation

## 3. Google Cloud BigQuery -->
