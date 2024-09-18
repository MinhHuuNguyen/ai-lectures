---
slug: machine-learning
time: "11/09/2024"
title: "Machine learning"
description: "Machine Learning (ML) là một phần của trí tuệ nhân tạo (AI) mà chúng ta dùng để xây dựng các mô hình hoặc chương trình máy tính có khả năng tự học từ dữ liệu."
author: "Nguyễn Hữu Minh"
banner_url: "https://tenten.vn/tin-tuc/wp-content/uploads/2023/08/1cG6U1qstYDijh9bPL42e-Q.jpg"
tags:
---

# Object detection

## 1. Giới thiệu chung bài toán object detection
Bài toán object detection là một bài toán rất phổ biến trong computer vision và được coi là một trong số các bài toán machine learning kinh điển.

Tính ứng dụng của bài toán object detection trong thực tiễn là rất lớn trong nhiều ngành nghề khác nhau.
Object detection được sử dụng trong y tế giúp xác định vị trí bị bệnh trong cơ thể, trong bảo mật giúp định vị vị trí của con người trong những khu vực cấm, trong nông nghiệp giúp xác định số lượng nông sản, trong hệ thống xe tự hành ...

<img src="https://i.imgur.com/zsla1eq.png" style="width: 700px;"/>

Bài toán object detection là sự tổng hợp của hai bài toán con: object localization và image classification. 
- Object localization là bài toán định vị vị trí của object trong ảnh: nhận đầu vào là một ảnh và trả đầu ra là một hoặc nhiều bbox của từng đối tượng.
- Image classification là bài toán phân lớp ảnh: nhận đầu vào là một ảnh và trả đầu ra là lớp của đối tượng đó.
Bài toán object detection kết hợp cả hai bài toán trên, yêu cầu mô hình vừa định vị vị trí của một hoặc nhiều đối tượng trong ảnh vừa xác định lớp của từng đối tượng đó.

## 2. Khái quát các mô hình giải quyết bài toán object detection

### 2.1. Nhóm các mô hình two-stage

Các mô hình thuộc nhóm two-stage ra đời khá sớm từ năm 2014 đến 2017. Nhóm này có đặc điểm chung về kiến trúc gồm hai phần:
- Region proposals module: module nhận đầu vào là ảnh ban đầu và trả đầu ra là các khu vực trên ảnh mà có khả năng chứa đối tượng.
- Feature extraction module: module nhận đầu vào là các region từ Region proposals module giúp xác định chính xác đối tượng trong khu vực đó là đối tượng nào và tinh chỉnh toạ độ của khu vực chính xác hơn.

<img src="https://www.researchgate.net/publication/353284602/figure/fig3/AS:1046072046673927@1626414419841/Two-stage-vs-one-stage-object-detection-models.ppm" style="width: 700px;"/>

### 2.2. Nhóm các mô hình single-stage

Các mô hình thuộc nhóm single-stage ra đời muộn hơn từ năm 2016 đến nay, tuy nhiên lại đang nhận được sự quan tâm rất lớn của giới nghiên cứu trong thời gian trở lại đây vì tính ứng dụng trong thực tiễn cao của chúng.

Các mô hình single-stage đều dựa vào động lực trong việc loại bỏ Region proposals module nhằm giảm khối lượng tính toán, qua đó tăng tốc độ và đưa mô hình đến gần hơn với khả năng chạy real-time.
