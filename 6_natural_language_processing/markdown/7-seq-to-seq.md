---
time: 06/24/2022
title:
description:
banner_url:
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: false
---

BLEU (Bilingual Evaluation Understudy):

Ứng dụng chính: Dịch máy.
Ý nghĩa: Đo lường sự tương đồng giữa câu do máy dịch và một hoặc nhiều câu tham chiếu (reference) của con người bằng cách tính toán độ chính xác của các n-gram (cụm từ gồm n từ). BLEU càng cao, bản dịch càng được cho là tốt.
Hạn chế: Không bắt được ý nghĩa, từ đồng nghĩa hoặc cấu trúc ngữ pháp khác nhau nhưng vẫn đúng.

Nguồn gốc: Ban đầu được tạo ra để đánh giá chất lượng của các hệ thống dịch máy (Machine Translation).
Ý tưởng chính: Đo lường độ chính xác (precision) của các cụm từ (gọi là n-gram) trong câu do mô hình tạo ra so với các câu tham chiếu (reference captions) do con người viết.
Cách hoạt động:
Nó đếm số lượng các n-gram (cụm 1 từ, 2 từ, 3 từ, 4 từ, v.v.) trong câu dự đoán xuất hiện trong bất kỳ câu tham chiếu nào.
Điểm số càng cao nếu câu dự đoán có nhiều cụm từ trùng khớp với các câu tham chiếu.
BLEU cũng có một "hình phạt cho câu quá ngắn" (brevity penalty) để tránh trường hợp mô hình tạo ra những câu rất ngắn nhưng có độ chính xác cao (ví dụ: chỉ dự đoán "a dog") và được điểm cao một cách không công bằng.
Ưu điểm:
Đơn giản, nhanh chóng để tính toán.
Là một tiêu chuẩn được sử dụng rộng rãi, dễ dàng so sánh với các nghiên cứu trước đây.
Nhược điểm:
Không quan tâm đến ngữ nghĩa: Hai câu có thể có nghĩa giống hệt nhau nhưng dùng từ khác nhau và sẽ bị điểm BLEU thấp. Ví dụ: "a man is walking his dog" và "a guy is strolling with his pet".
Không quan tâm đến thứ tự từ: Với các n-gram ngắn (n=1), nó chỉ quan tâm đến sự xuất hiện của từ mà không cần đúng thứ tự.
Thường không tương quan tốt với đánh giá của con người về chất lượng tổng thể của chú thích.


ROUGE (Recall-Oriented Understudy for Gisting Evaluation):

Ứng dụng chính: Tóm tắt văn bản.
Ý nghĩa: Tương tự BLEU nhưng dựa trên Recall. Nó đo lường số lượng n-gram trong bản tóm tắt tham chiếu xuất hiện trong bản tóm tắt do máy tạo ra.
Các biến thể: ROUGE-N (n-gram), ROUGE-L (dựa trên chuỗi con chung dài nhất - Longest Common Subsequence), ROUGE-S (skip-bigram).

Nguồn gốc: Ban đầu được thiết kế cho bài toán tóm tắt văn bản (Text Summarization).
Ý tưởng chính: Ngược lại với BLEU, ROUGE tập trung vào độ bao phủ (recall). Nó đo lường xem có bao nhiêu n-gram trong các câu tham chiếu của con người xuất hiện trong câu do mô hình tạo ra.
Cách hoạt động:
ROUGE-N: Tương tự BLEU-N nhưng tính theo recall.
ROUGE-L: Đo lường chuỗi con chung dài nhất (Longest Common Subsequence - LCS) giữa câu dự đoán và câu tham chiếu. Điều này giúp đánh giá sự tương đồng về cấu trúc câu mà không cần các từ phải liền kề nhau.
Ưu điểm:
ROUGE-L có thể nắm bắt sự tương đồng về cấu trúc câu tốt hơn BLEU.
Nhược điểm:
Vẫn gặp các vấn đề tương tự BLEU về mặt ngữ nghĩa.


METEOR (Metric for Evaluation of Translation with Explicit ORdering):

Ứng dụng chính: Dịch máy.
Ý nghĩa: Một phiên bản cải tiến của BLEU, nó xem xét cả từ đồng nghĩa, gốc từ (stemming) và sắp xếp từ. METEOR thường tương quan tốt hơn với đánh giá của con người so với BLEU.
Nguồn gốc: Cũng từ lĩnh vực dịch máy, được tạo ra để khắc phục một số nhược điểm của BLEU.
Ý tưởng chính: Đo lường sự tương đồng dựa trên việc so khớp các unigram (từ đơn) một cách linh hoạt hơn.
Cách hoạt động:
So khớp linh hoạt: Không chỉ so khớp từ chính xác, METEOR còn xem xét các từ đồng nghĩa (synonyms) và các từ cùng gốc (stemming). Ví dụ, "walking" và "walked" sẽ được coi là trùng khớp.
Xem xét thứ tự: Nó có một hình phạt cho các "khối" (chunks) từ không được sắp xếp đúng thứ tự so với câu tham chiếu, giúp đánh giá sự trôi chảy của câu.
Ưu điểm:
Tương quan với đánh giá của con người tốt hơn so với BLEU và ROUGE.
Linh hoạt hơn trong việc so khớp từ.
Nhược điểm:
Phức tạp hơn để tính toán.
Yêu cầu các tài nguyên ngôn ngữ như WordNet để tìm từ đồng nghĩa.



Perplexity (PPL):

Ứng dụng chính: Đánh giá các mô hình ngôn ngữ (Language Models).
Ý nghĩa: Đo lường mức độ "ngạc nhiên" của mô hình khi nó gặp một chuỗi văn bản. Perplexity càng thấp, mô hình càng dự đoán tốt chuỗi văn bản đó, tức là mô hình ngôn ngữ càng tốt.


BERTScore:

Ứng dụng: Dịch máy, tóm tắt, sinh văn bản.
Ý nghĩa: Một metric hiện đại sử dụng các embedding từ BERT để đo lường sự tương đồng về mặt ngữ nghĩa giữa câu được tạo ra và câu tham chiếu. Nó khắc phục được nhiều nhược điểm của các metric dựa trên n-gram như BLEU và ROUGE.