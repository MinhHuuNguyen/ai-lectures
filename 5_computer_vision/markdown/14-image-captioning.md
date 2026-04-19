---
time: 06/24/2022
title:
description:
banner_url:
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: false
---


CIDEr (Consensus-based Image Description Evaluation)
Nguồn gốc: Được thiết kế đặc biệt cho bài toán Image Captioning. Đây là một trong những metric quan trọng nhất hiện nay.
Ý tưởng chính: Đo lường mức độ tương đồng của một câu chú thích với "sự đồng thuận" (consensus) của tập hợp các câu mô tả của con người. Một chú thích tốt là chú thích nắm bắt được những gì mà hầu hết mọi người đều đồng ý mô tả về bức ảnh.
Cách hoạt động:
Sử dụng phương pháp trọng số TF-IDF (Term Frequency-Inverse Document Frequency) trên các n-gram.
Các n-gram xuất hiện thường xuyên trong các câu tham chiếu của một ảnh cụ thể (TF cao) nhưng lại hiếm khi xuất hiện trong toàn bộ tập dữ liệu (IDF cao) sẽ được coi là quan trọng và có trọng số cao hơn.
Điều này giúp đánh giá xem mô hình có nắm bắt được các chi tiết đặc trưng và độc đáo của bức ảnh hay không.
Ưu điểm:
Được chứng minh là có độ tương quan rất cao với đánh giá của con người.
Thưởng cho các mô hình tạo ra các mô tả độc đáo và chi tiết, thay vì các câu chung chung.
Nhược điểm:
Có thể không ổn định trên các tập dữ liệu nhỏ.


5. SPICE (Semantic Propositional Image Caption Evaluation)
Nguồn gốc: Cũng được thiết kế riêng cho Image Captioning, tập trung hoàn toàn vào ý nghĩa ngữ nghĩa (semantics).
Ý tưởng chính: Đánh giá xem câu chú thích có mô tả đúng các đối tượng, thuộc tính và mối quan hệ giữa chúng trong ảnh hay không.
Cách hoạt động:
Nó phân tích câu dự đoán và các câu tham chiếu thành một "đồ thị cảnh" (scene graph).
Một đồ thị cảnh bao gồm các nút (đối tượng, ví dụ: "người đàn ông", "con chó") và các cạnh (mối quan hệ, ví dụ: "đang dắt"; thuộc tính, ví dụ: "màu nâu").
Sau đó, nó tính toán điểm F1-score (kết hợp giữa precision và recall) trên sự trùng khớp của các bộ ba (object, relation, attribute) trong các đồ thị cảnh này.
Ưu điểm:
Là metric tốt nhất hiện nay để đánh giá sự chính xác về mặt ngữ nghĩa. Nó thực sự "hiểu" được nội dung câu mô tả.
Nhược điểm:
Phức tạp và chậm để tính toán vì yêu cầu một bộ phân tích cú pháp (parser) mạnh mẽ.
Không đánh giá sự trôi chảy hay ngữ pháp của câu. Một câu có thể đúng về mặt ngữ nghĩa nhưng sai ngữ pháp vẫn có thể đạt điểm SPICE cao.

