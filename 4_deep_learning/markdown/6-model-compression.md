---
time: 10/18/2022
title: Nén mô hình mạng nơ ron Model Compression
description: Các mô hình mạng nơ ron có kích thước lớn và phức tạp có thể đạt được độ chính xác cao trên các bài toán khác nhau. Tuy nhiên, việc triển khai các mô hình này trên các thiết bị có tài nguyên hạn chế như các thiết bị di động hoặc các thiết bị IoT là một thách thức lớn. Nén mô hình mạng nơ ron là một giải pháp giúp giảm kích thước mô hình, tăng tốc độ tính toán và giảm tài nguyên cần thiết cho việc triển khai mô hình mà không làm suy giảm hiệu suất quá nhiều.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/6-model-compression/banner.png
tags: [machine-learning, deep-learning]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về Model Compression

- Deep learning phát triển nhanh chóng, kéo theo việc nén mô hình nhận được sự chú ý đặc biệt.
- Nhu cầu triển khai mô hình trên các thiết bị có tài nguyên hạn chế (mobile device) ngày càng tăng, việc nén các mô hình học sâu lớn và phức tạp mà không làm suy giảm hiệu suất trở nên quan trọng.
- Giá trị cốt lõi mà Model Compression mang lại là giúp tạo ra mô hình mới chạy nhanh hơn, tốn ít tài nguyên hơn và duy trì được độ chính xác tương đương.

## 2. Model Quantization (Lượng tử hóa mô hình)

- Quantization là phương pháp giảm số lượng bit cần lưu trữ cho các trọng số của mô hình.
- Thay vì lưu trữ các trọng số dưới dạng số thực 32-bit, ta có thể lưu trữ chúng dưới dạng số nguyên 8-bit hoặc 16-bit.
- Việc giảm số lượng bit giúp giảm dung lượng mô hình, tăng tốc độ tính toán và giảm tài nguyên cần thiết cho việc triển khai mô hình.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/6-model-compression/quantization.png" style="width: 900px;"/>

#### Uniform quantization và Non-uniform quantization

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/6-model-compression/uniform_non_uniform_quantization.png" style="width: 900px;"/>

#### Symmetric quantization và Asymmetric quantization

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/6-model-compression/symmetric_asymmetric_quantization.png" style="width: 900px;"/>

### 2.1. Post-Training Quantization (PTQ)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/6-model-compression/ptq.png" style="width: 900px;"/>

#### Dynamic Quantization (Stochastic Quantization)

#### Static Quantization (Deterministic Quantization)

### 2.2. Quantization Aware Training (QAT)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/6-model-compression/qat_ptq.png" style="width: 900px;"/>

## 3. Model Pruning (Cắt tỉa mô hình)

- Pruning là phương pháp loại bỏ các trọng số không quan trọng khỏi mô hình.
- Các trọng số không quan trọng thường có giá trị gần bằng 0 hoặc rất nhỏ.
- Việc loại bỏ các trọng số không quan trọng giúp giảm dung lượng mô hình, tăng tốc độ tính toán và giảm tài nguyên cần thiết cho việc triển khai mô hình.


## 4. Knowledge distillation (Truyền thụ kiến thức)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/6-model-compression/distillation.png" style="width: 900px;"/>

- Knowledge distillation là phương pháp huấn luyện một mô hình mới (student model) sao cho nó học từ một mô hình mạnh hơn (teacher model).
- Mô hình mạnh hơn thường có kích thước lớn và phức tạp, trong khi mô hình mới cần có kích thước nhỏ và tốc độ tính toán nhanh.
- Knowledge distillation giúp mô hình mới học được kiến thức từ mô hình mạnh hơn, giảm kích thước mô hình và tăng tốc độ tính toán.


## 5. Matrix low-rank approximation (Xấp xỉ ma trận hạng thấp)

- Low-rank approximation là phương pháp xấp xỉ ma trận trọng số của mô hình bằng cách phân rã ma trận trọng số thành tích của hai ma trận có hạng thấp hơn.
- Việc xấp xỉ ma trận trọng số giúp giảm dung lượng mô hình, tăng tốc độ tính toán và giảm tài nguyên cần thiết cho việc triển khai mô hình.

