---
time: 08/16/2022
title: Mô hình DBSCAN
description: Khác với K-means Clustering, mô hình phân cụm DBSCAN Clustering không yêu cầu số lượng cụm cần phân chia trước. Trong bài viết này, chúng ta sẽ tìm hiểu về mô hình DBSCAN Clustering, mô hình giúp phân chia dữ liệu thành các cụm dựa trên mật độ của chúng.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/11-kmeans/banner.png
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Tổng quan

Phân cụm (clustering) là bài toán học máy không giám sát nhằm nhóm các điểm dữ liệu thành những cụm sao cho các điểm trong cùng cụm tương đồng với nhau hơn so với các điểm ở cụm khác.

Khác với K-means Clustering, mô hình phân cụm DBSCAN Clustering không yêu cầu số lượng cụm cần phân chia trước.
Thay vào đó, mô hình này sử dụng mật độ của các điểm dữ liệu để xác định các cụm.

| Kết quả với DBSCAN  | Kết quả với K-Means |
|---------------------|---------------------|
| <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/example_blobs_dbscan.gif" width="400"/> | <img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/12-dbscan/example_blobs_kmeans.gif" width="400"/> |

DBSCAN là viết tắt của **Density-Based Spatial Clustering of Applications with Noise** - tạm dịch là Phân cụm không gian dựa trên mật độ với nhiễu.
DBSCAN là một trong những thuật toán phân cụm phổ biến nhất trong học máy không giám sát và được sử dụng nhiều trong các bài toán Định danh khuôn mặt (Face Recognition) hay Nhận diện bất thường (Anomaly Detection).

## 2. Các khái niệm được định nghĩa trong mô hình DBSCAN

### 2.1. Epsilon Neighborhood

### 2.2. Core Point

### 2.3. Directly Density Reachable

### 2.4. Density Reachable

### 2.5. Density Connected

### 2.6. Cluster

### 2.7. Noise

## 3. Các bước của thuật toán

## 4. Ưu và nhược điểm của mô hình

## 5. Các biến thể nâng cấp của mô hình

