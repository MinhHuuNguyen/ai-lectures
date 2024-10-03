---
slug: model-compression
time: 11/09/2024
title: Nén mô hình mạng nơ ron - Model Compression
description:
author: Nguyễn Hữu Minh
banner_url: 
tags: [deep-learning]
is_highlight: false
---

# Model compression

## 1. Giới thiệu về Model compression

- Deep learning phát triển nhanh chóng, kéo theo việc nén mô hình nhận được sự chú ý đặc biệt.
- Nhu cầu triển khai mô hình trên các thiết bị có tài nguyên hạn chế (mobile device) ngày càng tăng, việc nén các mô hình học sâu lớn và phức tạp mà không làm suy giảm hiệu suất trở nên quan trọng.
- Giá trị cốt lõi mà Model Compression mang lại là giúp tạo ra mô hình mới chạy nhanh hơn, tốn ít tài nguyên hơn và duy trì được độ chính xác tương đương.

## 2. Các phương pháp Model compression

### 2.1. Quantization (Lượng tử hóa)

- Quantization là phương pháp giảm số lượng bit cần lưu trữ cho các trọng số của mô hình.
- Thay vì lưu trữ các trọng số dưới dạng số thực 32-bit, ta có thể lưu trữ chúng dưới dạng số nguyên 8-bit hoặc 16-bit.
- Việc giảm số lượng bit giúp giảm dung lượng mô hình, tăng tốc độ tính toán và giảm tài nguyên cần thiết cho việc triển khai mô hình.

Phân loại Quantization:
- **Dynamic Quantization**: Lượng tử hoá các trọng số 
- **Static Quantization**: Số lượng bit cần lưu trữ cho trọng số không thay đổi.
- **Quantization-aware training**: Huấn luyện mô hình với một số lượng bit cố định cho trọng số.

<img src="https://apple.github.io/coremltools/docs-guides/_images/quantization-technique.png" style="width: 700px;"/>

### 2.2. Pruning (Cắt tỉa)

- Pruning là phương pháp loại bỏ các trọng số không quan trọng khỏi mô hình.
- Các trọng số không quan trọng thường có giá trị gần bằng 0 hoặc rất nhỏ.
- Việc loại bỏ các trọng số không quan trọng giúp giảm dung lượng mô hình, tăng tốc độ tính toán và giảm tài nguyên cần thiết cho việc triển khai mô hình.

<img src="https://developer.nvidia.com/blog/wp-content/uploads/2019/03/remove_neuron.png" style="width: 600px;"/>

### 2.3. Knowledge distillation (Truyền thụ kiến thức)

- Knowledge distillation là phương pháp huấn luyện một mô hình mới (student model) sao cho nó học từ một mô hình mạnh hơn (teacher model).
- Mô hình mạnh hơn thường có kích thước lớn và phức tạp, trong khi mô hình mới cần có kích thước nhỏ và tốc độ tính toán nhanh.
- Knowledge distillation giúp mô hình mới học được kiến thức từ mô hình mạnh hơn, giảm kích thước mô hình và tăng tốc độ tính toán.

<img src="https://editor.analyticsvidhya.com/uploads/30818Knowledge%20Distillation%20Flow%20Chart%201.2.jpg" style="width: 800px;"/>


### 2.4. Low-rank approximation (Xấp xỉ ma trận hạng thấp)

- Low-rank approximation là phương pháp xấp xỉ ma trận trọng số của mô hình bằng cách phân rã ma trận trọng số thành tích của hai ma trận có hạng thấp hơn.
- Việc xấp xỉ ma trận trọng số giúp giảm dung lượng mô hình, tăng tốc độ tính toán và giảm tài nguyên cần thiết cho việc triển khai mô hình.

<img src="https://dustinstansbury.github.io/theclevermachine/assets/images/svd-data-compression/low-rank-approximation.png" style="width: 600px;"/>
