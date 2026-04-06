---
time: 06/04/2022
title: Bài toán phân đoạn ảnh image segmentation
description: Phân đoạn ảnh (Image Segmentation) là một trong những nhiệm vụ cốt lõi và quan trọng nhất của lĩnh vực thị giác máy tính. Mục tiêu của nó là phân chia một hình ảnh kỹ thuật số thành nhiều vùng hoặc đối tượng khác nhau. Việc phân nhóm các mô hình phân đoạn ảnh giúp chúng ta hiểu rõ hơn về cách tiếp cận, ưu nhược điểm và ứng dụng của từng loại. Các mô hình này có thể được phân thành hai nhóm chính là Phương pháp truyền thống và Phương pháp dựa trên Deep Learning. Hiện nay, các mô hình Deep Learning chiếm ưu thế tuyệt đối về độ chính xác và hiệu quả trong các bài toán phức tạp.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/7-image-segmentation/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về image segmentation

### 1.1. Nhóm các mô hình semantic segmentation

**Semantic segmentation** là một dạng khác của bài toán classification, trong đó, mô hình, thay vì phân lớp trên cả ảnh, sẽ phân lớp từng pixel trên ảnh thuộc lớp nào

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/semantic_segmentation.jpeg" style="width: 1000px;"/>

### 1.2. Nhóm các mô hình instance segmentation

**Instance segmentation:** là một phiên bản cao hơn của semantic segmentation, bên cạnh việc phân lớp các pixel trên ảnh thuộc lớp nào, đối với những pixel thuộc cùng một lớp, mô hình cần phải phân lớp rõ pixel đó thuộc đối tượng nào

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/instance_segmentation.jpeg" style="width: 1000px;"/>

## 3. Các metrics trong image segmentation
