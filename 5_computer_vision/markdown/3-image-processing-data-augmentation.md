---
time: 05/07/2022
title: Kỹ thuật xử lý hình ảnh Image processing và tăng cường dữ liệu hình ảnh Image data augmentation
description: Xử lý ảnh là một lĩnh vực rộng lớn trong Thị giác máy tính, cung cấp các công cụ và thuật toán để thao tác và phân tích hình ảnh kỹ thuật số. Nó là nền tảng cho nhiều ứng dụng từ chỉnh sửa ảnh đơn giản đến phức tạp. Tăng cường dữ liệu ảnh là một kỹ thuật không thể thiếu trong việc xây dựng các mô hình Thị giác máy tính dựa trên mạng nơ ron hiện đại. Bằng cách sử dụng các phép xử lý ảnh để tạo ra dữ liệu huấn luyện đa dạng, chúng ta có thể xây dựng các mô hình AI chính xác hơn, mạnh mẽ hơn và có khả năng khái quát hóa tốt hơn trong thế giới thực.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/3-image-processing-data-augmentation/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: true
---

## 1. Xử lý ảnh (Image Processing)

### 1.1. Giới thiệu chung về Image Processing

Image Processing (Xử lý ảnh) là một lĩnh vực trong khoa học máy tính và công nghệ thông tin dùng để xử lý và thay đổi hình ảnh hoặc video để trích xuất thông tin, cải thiện chất lượng, và thực hiện các nhiệm vụ khác liên quan đến hình ảnh.

- Đầu vào: Một hình ảnh (ví dụ: ảnh chụp, ảnh vệ tinh, ảnh y tế ...).
- Đầu ra: Có thể là một hình ảnh khác (đã được cải thiện) hoặc một tập hợp các đặc trưng, thông tin về hình ảnh đó.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/3-image-processing-data-augmentation/image_processing.jpeg" style="width: 1000px;"/>

Để máy tính có thể "hiểu" và xử lý, hình ảnh được biểu diễn dưới dạng một ma trận các con số. Mỗi phần tử trong ma trận được gọi là một pixel (điểm ảnh).

- Ảnh xám (Grayscale): Mỗi pixel có một giá trị duy nhất biểu thị cường độ sáng (thường từ 0 - đen đến 255 - trắng).
- Ảnh màu (RGB): Mỗi pixel được biểu diễn bởi một bộ 3 giá trị, tương ứng với cường độ của ba kênh màu cơ bản: Đỏ (Red), Lục (Green), và Lam (Blue).

### 1.2. Các cấp độ của Xử lý ảnh

- **Xử lý ảnh cấp thấp (Low-level Processing)**: Thực hiện các thao tác cơ bản trực tiếp trên các pixel với cả đầu vào và đầu ra đều là hình ảnh.
Ví dụ:
    - Giảm nhiễu (Noise Reduction): Loại bỏ các điểm ảnh nhiễu không mong muốn.
    - Tăng cường độ tương phản (Contrast Enhancement): Làm cho hình ảnh rõ nét hơn.
    - Làm sắc nét (Sharpening): Tăng cường các cạnh và chi tiết.

- **Xử lý ảnh cấp trung (Mid-level Processing)**: Trích xuất các thuộc tính từ ảnh, phân nhóm các pixel thành các đối tượng với đầu vào là hình ảnh, đầu ra là các thuộc tính (ví dụ: các cạnh, đường viền, vùng).
Ví dụ:
    - Phân vùng ảnh (Image Segmentation): Chia hình ảnh thành nhiều vùng hoặc đối tượng có ý nghĩa.
    - Trích xuất đặc trưng (Feature Extraction): Tìm các đặc điểm quan trọng như góc, cạnh, kết cấu (texture).

- **Xử lý ảnh cấp cao (High-level Processing)**: Hiểu và phân tích nội dung của hình ảnh, gần với cách con người nhận thức với đầu vào là hình ảnh, đầu ra là các hiểu biết về hình ảnh đó.
Đây là nơi áp dụng các mô hình deep learning vào computer Vision.
Ví dụ:
    - Nhận dạng đối tượng (Object Recognition): Xác định trong ảnh có chứa đối tượng ("ô tô", "con người", "con mèo" ...)
    - Phân loại ảnh (Image Classification): Gán nhãn cho toàn bộ bức ảnh ("cảnh bãi biển", "phòng khách" ...)
    - Mô tả ảnh (Image Captioning): Mô tả nội dung của ảnh bằng văn bản.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/3-image-processing-data-augmentation/rotate_flip_crop_resize.jpeg" style="width: 1000px;"/>

### 1.3. Một số thao tác phổ biến trong Xử lý ảnh

#### Biến đổi không gian: Rotate, Flip, Crop, Resize

- Rotate (Xoay): Xoay hình ảnh một góc độ xác định.
- Flip (Lật hình): Lật hình ảnh theo trục ngang hoặc trục dọc.
- Crop (Cắt): Cắt ra một phần của hình ảnh và sử dụng nó.
- Resize (Phóng to và thu nhỏ): Thay đổi kích thước hình ảnh.

#### Biến đổi chất lượng: Color, Brightness, Contrast, Saturation, Blur, Noise

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/3-image-processing-data-augmentation/color_bright_constrast_saturation.jpeg" style="width: 1000px;"/>

- Color: Thay đổi màu sắc của hình ảnh.
- Brightness: Thay đổi độ sáng của hình ảnh.
- Contrast: Thay đổi độ tương phản của hình ảnh. Constrast là mức độ 
- Saturation: Thay đổi độ bão hoà của hình ảnh. Saturation là mức độ 
- Blur / De-blur: Tăng hoặc giảm mờ của hình ảnh.
- Noise / De-noise: Tăng hoặc giảm nhiễu của hình ảnh.

#### Trích xuất thông tin: Edge detection, Corner detection, Image segmentation

- Edge detection:
- Corner detection:
- Image segmentation

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/3-image-processing-data-augmentation/edge_corner_segmentation.jpeg" style="width: 1000px;"/>

#### Cắt dán hình ảnh

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/4-object-detection/cut_mix.jpeg" style="width: 500px;"/>

## 2. Tăng cường dữ liệu ảnh (Image data augmentation)

### 2.1. Vấn đề thiếu dữ liệu và hiện tượng overfitting

Overfitting là hiện tượng xảy ra khi mô hình học thuộc phần lớn (hoặc thậm chí toàn bộ) bộ dữ liệu train.
Lúc này, mô hình không còn khả năng khái quát hoá bộ dữ liệu nữa mà chỉ ghi nhớ các điểm dữ liệu trong bộ train.

Nguyên nhân dẫn đến hiện tượng overfit là do mô hình có độ phức tạp quá cao (mô hình quá phức tạp) so với bộ dữ liệu train quá đơn giản.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/3-image-processing-data-augmentation/data_augmentation_reason.jpeg" style="width: 500px;"/>

Chi tiết hơn về hiện tượng overfit đã được chia sẻ trong bài viết [Hiện tượng Overfit và Underfit](/blog/hien-tuong-overfit-va-underfit).

### 2.2. Giới thiệu chung về Image data augmentation

Data Augmentation (Tăng cường dữ liệu) là một kỹ thuật quan trọng trong Machine Learning và Deep Learning, đặc biệt là trong việc xử lý dữ liệu hình ảnh Image data augmentation.

Kỹ thuật này bao gồm việc tạo ra các biến thể của dữ liệu huấn luyện bằng cách áp dụng các biến đổi đơn giản lên trên dữ liệu huấn luyện để tạo ra dữ liệu mới mà không thay đổi tính năng quan trọng của dữ liệu gốc.
Đối với Image data augmentation, ta có thể sử dụng các kỹ thuật image processing như Rotate, Flip, Crop, Resize, Noise/Denoise, Blur/Deblur, Color Jitter ... 

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/3-image-processing-data-augmentation/data_augmentation.jpeg" style="width: 500px;"/>

Vai trò chính của data augmentation là giúp tăng kích thước dữ liệu huấn luyện bằng cách tạo ra các biến thể của dữ liệu gốc mà không cần thu thập thêm dữ liệu mới.
Điều này đặc biệt hữu ích khi bạn có ít dữ liệu.

Từ việc tăng kích thước của bộ dữ liệu huấn luyện, ta thu được lợi ích đối với quá trình huấn luyện mô hình machine learning: **Giảm overfitting**.

### 2.3. So sánh image processing và data augmentation
