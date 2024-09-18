---
slug: machine-learning
time: "11/09/2024"
title: "Machine learning"
description: "Machine Learning (ML) là một phần của trí tuệ nhân tạo (AI) mà chúng ta dùng để xây dựng các mô hình hoặc chương trình máy tính có khả năng tự học từ dữ liệu."
author: "Nguyễn Hữu Minh"
banner_url: "https://tenten.vn/tin-tuc/wp-content/uploads/2023/08/1cG6U1qstYDijh9bPL42e-Q.jpg"
tags:
---

# CNN model visualization

## 1. Giới thiệu về model visualization

Khi phát triển các mô hình machine learning, chúng ta mong muốn biết được model đang học tốt ở phần nào và chưa tốt ở phần nào.
Một phương pháp quan trọng để hiểu mô hình là thông qua trực quan hóa mô hình.

Vai trò chính của model visualization là giúp chúng ta hiểu rõ quyết định của mô hình, giúp giải thích được lý do tại sao mô hình lại đưa ra quyết định đó.
Từ đó, ta sẽ xác định được đâu là những điểm mạnh mà mô hình đã đạt được và đâu là những điểm yếu của mô hình và cải thiện từ đó.

Có 2 kiểu model visualization chính:
- **Model visualization during training**: Trực quan hóa mô hình trong quá trình huấn luyện
    - **Scalars**:
        - Giúp hiểu trend học của model.
        - Ví dụ: loss, accuracy, learning rate, ...
    - **Histograms**:
        - Trực quan hoá sự thay đổi của phân phối của các trọng số trong model
        - Ví dụ: weight, bias, gradient, ...
- **Model visualization after training**: Trực quan hóa mô hình sau khi huấn luyện
    - **Feature map**: Trực quan hoá các feature map của các layer trong model
    - **Kernel**: Trực quan hoá các kernel của các layer Conv trong model CNN
    - **Activation**: Trực quan hoá các activation map của các layer trong model
    - **Image**: Trực quan hoá input dữ liệu và output dự đoán của model

## 2. Feature map và kernel visualization

Việc trực quan hoá kernel và feature map giúp ta hiểu cách các kernels hoạt động trong việc trích xuất các đặc trưng từ ảnh và xem xét cách các feature map được tạo ra từ các kernels khác nhau và như thế nào chúng cùng đóng góp vào quá trình nhận dạng đối tượng.

- Trực quan hoá kernel:
    - Hiển thị các kernels dưới dạng hình ảnh, trong đó mỗi kernel sẽ là một ma trận nhỏ, và các giá trị mô tả cường độ của mỗi pixel trong kernel.
    - Giúp ta quan sát được cách kernel nhận dạng các đặc trưng cụ thể như cạnh, góc, hoặc kết cấu.
- Trực quan hoá feature map:
    - Hiển thị để xem xét cách chúng phản ánh các đặc trưng trên ảnh gốc sau khi áp dụng bộ lọc.
    - Có thể thấy cách các feature map biểu thị đặc trưng ở các mức độ khác nhau trong mạng CNN.

## 3. Activation visualization

### 3.1. CAM

### 3.2. Grad-CAM
