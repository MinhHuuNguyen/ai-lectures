---
time: 03/04/2023
title: seq2seq và mô hình Transformer
description: Mô hình seq2seq (sequence-to-sequence) gồm Encoder (mã hóa) và Decoder (giải mã) là kiến trúc mạng nơ-ron được sử dụng để chuyển đổi một chuỗi đầu vào thành một chuỗi đầu ra. Tuy nhiên, mô hình seq2seq truyền thống gặp khó khăn trong việc xử lý các chuỗi dài do phụ thuộc vào RNN/LSTM. Năm 2017, Transformer ra mắt đã giải quyết vấn đề này bằng cách sử dụng cơ chế Attention hoàn toàn, cho phép mô hình học được mối quan hệ giữa các từ trong chuỗi mà không cần tuần tự.
banner_url:
tags: [deep-learning, natural-language-processing]
is_highlight: false
is_published: true
---

## 1. Mô hình Encoder - Decoder truyền thống
