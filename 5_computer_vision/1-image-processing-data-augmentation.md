---
layout: "post"
title:  "Image processing và data augmentation"
author: "Nguyễn Hữu Minh"
permalink: "/deep-learning/computer-vision/image-processing-data-augmentation"
parent: "Computer vision"
grand_parent: "Deep learning"
nav_order: 2
---

# Image processing và data augmentation

## 1. Giới thiệu chung về image processing và data augmentation

### 1.1. Giới thiệu về image processing

Image Processing (Xử lý ảnh) là một lĩnh vực trong khoa học máy tính và công nghệ thông tin dùng để xử lý và thay đổi hình ảnh hoặc video để trích xuất thông tin, cải thiện chất lượng, và thực hiện các nhiệm vụ khác liên quan đến hình ảnh.

Các nhiệm vụ trong Image Processing bao gồm:
- Làm sáng hoặc làm tối:
Điều chỉnh độ sáng hoặc độ tối của hình ảnh để cải thiện hiển thị hoặc tạo ra hiệu ứng mong muốn.
- Lọc ảnh:
Áp dụng bộ lọc để loại bỏ nhiễu (noise) khỏi hình ảnh hoặc làm nổi bật các đặc điểm quan trọng.
- Biến đổi hình dáng:
Thay đổi kích thước, xoay, thu nhỏ, phóng to hoặc biến đổi hình dáng của hình ảnh.
- Phát hiện và nhận dạng:
Nhận biết các đối tượng, khuôn mặt, chữ viết, hoặc biểu đồ trong hình ảnh.
- Trích xuất đặc trưng:
Trích xuất các đặc trưng quan trọng từ hình ảnh, như cạnh, màu sắc, hình dáng, vị trí, v.v.
- Biến đổi màu sắc:
Thay đổi màu sắc của hình ảnh hoặc chuyển đổi hình ảnh sang không gian màu khác nhau.
- Biểu đồ và tạo hiệu ứng hình ảnh:
Tạo hiệu ứng đồ họa, cắt, ghép ảnh, thay đổi màu sắc, v.v.
- Phân đoạn:
Phân tách hình ảnh thành các vùng riêng biệt dựa trên màu sắc, độ sáng hoặc đặc trưng khác.

### 1.2. Giới thiệu về data augmentation

Data Augmentation (Tăng cường dữ liệu) là một kỹ thuật quan trọng trong Machine Learning và Deep Learning, đặc biệt là trong việc xử lý dữ liệu hình ảnh.

Kỹ thuật này bao gồm việc tạo ra các biến thể của dữ liệu huấn luyện bằng cách áp dụng các biến đổi đơn giản như xoay, phóng to, thu nhỏ, lật hình, thay đổi độ sáng, tăng cường nhiễu ... để tạo ra dữ liệu mới mà không thay đổi tính năng quan trọng của dữ liệu gốc.

Vai trò của data augmentation:
- Tăng kích thước dữ liệu huấn luyện:
    - Data Augmentation giúp tăng kích thước dữ liệu bằng cách tạo ra các biến thể của dữ liệu gốc mà không cần thu thập thêm dữ liệu mới.
    - Điều này đặc biệt hữu ích khi bạn có ít dữ liệu.
- Từ việc tăng kích thước của bộ dữ liệu huấn luyện, ta thu được các lợi ích đối với mô hình deep learning:
    - Giảm overfitting:
        - Overfitting xảy ra khi mô hình học cụ thể cho dữ liệu huấn luyện mà không thể tổng quát hóa cho dữ liệu mới.
        - Khi bạn có ít dữ liệu huấn luyện, mô hình có nguy cơ overfitting cao.
        - Sử dụng Data Augmentation giúp mô hình được đào tạo trên các biến thể của dữ liệu, giúp giảm thiểu nguy cơ overfitting.
    - Cải thiện khả năng tổng quát hóa:
        - Data Augmentation giúp mô hình trở nên tốt hơn trong việc tổng quát hóa và đối phó với dữ liệu thực tế.
        - Mô hình được trang bị để xử lý nhiễu và biến đổi trong dữ liệu.
        - Mô hình đã được đào tạo trên dữ liệu tăng cường có khả năng hoạt động tốt hơn khi tái sử dụng cho các ứng dụng khác hoặc tại các bài toán mới.

### 1.3. So sánh image processing và data augmentation

- Điểm giống nhau:
    - Liên quan đến hình ảnh:
    Cả hai đều liên quan đến việc làm việc với hình ảnh.
    - Sử dụng các phép biến đổi hình ảnh:
    Cả hai đều sử dụng các thao tác như lọc, cắt, xoay, thay đổi kích thước, làm sáng tối, chuyển đổi màu sắc, và nhiều phép biến đổi khác...
- Điểm khác nhau:
    - Image Processing:
        - Mục tiêu nhằm cải thiện hoặc thay đổi hình ảnh để giúp con người hoặc máy tính hiểu rõ hơn nội dung hình ảnh, loại bỏ nhiễu, cải thiện chất lượng, hoặc trích xuất thông tin cụ thể từ hình ảnh.
        - Cần tinh chỉnh các giá trị để đạt được chính xác kết quả mong muốn.
    - Data Augmentation:
        - Mục tiêu nhằm tạo ra các biến thể của dữ liệu huấn luyện để cải thiện hiệu suất mô hình.
        Không phải là để cải thiện hình ảnh ban đầu, mà để mô hình học được từ sự đa dạng và tránh overfitting.
        - Không cần tinh chỉnh các giá trị để đạt được kết quả chính xác, chỉ cần tạo ra các biến thể của dữ liệu gốc mà không thay đổi đặc tính quan trọng của dữ liệu.

## 2. Các kỹ thuật data augmentation cơ bản

### 2.1. Rotate (Xoay):

Xoay hình ảnh một góc độ xác định. Điều này giúp mô hình học cách nhận dạng đối tượng ở các góc độ khác nhau.

### 2.2. Flip (Lật hình):

Lật hình ảnh theo trục ngang hoặc trục dọc. Điều này tạo ra hình ảnh đối xứng và cải thiện khả năng tổng quát hóa.

### 2.3. Crop (Cắt):

Cắt ra một phần của hình ảnh và sử dụng nó. Điều này giúp mô hình tập trung vào các phần quan trọng của hình ảnh.

### 2.4. Resize (Phóng to và thu nhỏ):

Thay đổi kích thước hình ảnh. Điều này giúp mô hình học cách nhận dạng đối tượng ở các kích thước khác nhau.

### 2.5. Thay đổi độ sáng, độ tương phản, độ bão hoà:

Tăng hoặc giảm độ sáng của hình ảnh. Điều này giúp mô hình học cách xử lý các điều kiện ánh sáng khác nhau.

### 2.6. Thay đổi màu sắc:

Thay đổi màu sắc của hình ảnh, chẳng hạn như chuyển đổi sang không gian màu khác, thay đổi độ bão hòa, độ sáng, hoặc độ tương phản.

### 2.7. Thêm nhiễu (noise):

Thêm nhiễu như nhiễu Gaussian, nhiễu Salt-and-Pepper, hoặc nhiễu khác vào hình ảnh. Điều này giúp mô hình trở nên bền vững hơn đối với nhiễu trong dữ liệu thực tế.

### 2.8. Thêm mờ (blur):

Thêm mờ vào hình ảnh. Điều này giúp mô hình trở nên bền vững hơn đối với nhiễu trong dữ liệu thực tế.

### 2.9. Thêm độ nghiêng (shear):

Thêm độ nghiêng vào hình ảnh. Điều này giúp mô hình học cách nhận dạng đối tượng ở các góc độ khác nhau.

<!-- ### 2.10. Thêm biến dạng (distort):

Occlusion -->


### 2.10. Kết hợp các biến đổi:

Kết hợp nhiều biến đổi lại với nhau để tạo ra sự đa dạng trong dữ liệu. Ví dụ, có thể xoay và phóng to hình ảnh cùng lúc.

## 3. Các kỹ thuật data augmentation nâng cao

### 3.1. Mix Up:

- Mục tiêu của Mix Up là tạo ra các hình ảnh mới bằng cách kết hợp thông tin từ hai hình ảnh gốc.
- Quá trình Mix Up:
    - Chọn ngẫu nhiên hai hình ảnh gốc và một hệ số α thuộc khoảng [0, 1].
    - Kết hợp hai hình ảnh này bằng cách tính trung bình có trọng số của các pixel từ hai hình ảnh theo công thức: new_image = α * image1 + (1 - α) * image2.
    - Tính trung bình có trọng số tương ứng cho nhãn của hai hình ảnh.
- Mix Up giúp mô hình học được tính toàn cầu của các đối tượng và giảm thiểu overfitting bằng cách tạo ra đa dạng trong dữ liệu huấn luyện.

<img src="https://www.researchgate.net/profile/Zihan-Yang-4/publication/357823193/figure/fig4/AS:1169970583080960@1655954132026/Sample-images-using-the-Mixup-Cutout-and-Cutmix-augmentation-methods-40.ppm" style="width: 800px;"/>

### 3.2. Cut Out:

- Mục tiêu của Cut Out là ẩn đi một phần ngẫu nhiên của hình ảnh bằng một hình chữ nhật đen.
- Quá trình Cut Out:
    - Chọn ngẫu nhiên một vùng hình vuông có kích thước và vị trí ngẫu nhiên trên hình ảnh.
    - Thay vùng này bằng một hình vuông đen.
- Cut Out tạo ra một hiệu ứng giúp mô hình phải học cách xử lý đối tượng khi một phần của nó bị ẩn đi.
Điều này giúp tăng khả năng tổng quát hóa và giảm khả năng overfitting.

### 3.3. Cut Mix:

- Cut Mix là một biến thể phức tạp hơn của Cut Out, nó kết hợp hai hình ảnh lại với nhau để tạo ra một hình ảnh mới.
- Quá trình Cut Mix:
    - Chọn ngẫu nhiên một vùng hình vuông có kích thước và vị trí ngẫu nhiên trên một hình ảnh gốc.
    - Chọn một hình ảnh khác và chọn một vùng tương tự.
    - Thay vùng hình vuông trên hình ảnh gốc bằng vùng tương tự từ hình ảnh khác.
    - Tính trung bình có trọng số của hai nhãn tương ứng với vùng hình vuông.
- Cut Mix tạo ra hình ảnh mới chứa thông tin từ cả hai hình ảnh gốc và giảm bớt thông tin về đối tượng ban đầu.
Điều này đặt ra một thách thức học hỏi cho mô hình.
