---
time:
title: Giới thiệu chung về Trí tuệ nhân tạo Artificial Intelligence
description: Trí tuệ nhân tạo - Artificial Intelligence là một lĩnh vực nghiên cứu và ứng dụng các phương pháp máy tính để mô phỏng và mở rộng khả năng tư duy của con người. Trong những năm gần đây, AI đã phát triển mạnh mẽ và đóng góp rất nhiều ứng dụng từ hỗ trợ con người đến tự động hóa các công việc. Bài giới thiệu này sẽ giới thiệu chung về Trí tuệ nhân tạo - AI, Máy học - Machine Learning và Học sâu - Deep Learning.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/ai_vs_ml_vs_dl.png
tags: [python, series]
is_highlight: false
is_published: true
---

## 1. Một số ví dụ nổi tiếng về Trí tuệ nhân tạo

### 1.1. ChatGPT - OpenAI

Dưới đây là một đoạn hội thoại ngắn giữa ChatGPT và một người dùng về nội dung liên quan đến lịch sử và ta có thể thấy rằng ChatGPT có khả năng tương tác với con người khá tốt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/example_chatgpt.png" style="width: 1200px;"/>

Đầu tiên, câu trả lời của ChatGPT có tính cấu trúc đồng nhất, gồm các phần: số thứ tự, tên viết bằng tiếng việt có dấu, chữ tượng hình tiếng trung trong ngoặc, dấu gạch ngang, đoạn giới thiệu ngắn bằng tiếng anh.

Thứ hai, trong câu trả lời trên, ChatGPT có khả năng hoạt động đa ngôn ngữ.

Cuối cùng và quan trọng nhất, tính đúng đắn của câu trả lời là khá cao, không có lỗi chính tả, ngữ pháp, cũng như thông tin lịch sử.

Kể từ thời điểm ra đời đến nay, ChatGPT nói riêng và các công cụ tương tự nói chung đã được cải thiện rất nhiều và có thể tương tác với con người một cách tự nhiên hơn, thông tin được cung cấp cũng chính xác hơn. Và dần dần, chúng đang trở thành một trợ thủ, một phần không thể thiếu trong cuộc sống hàng ngày của chúng ta, giúp chúng ta tăng tốc trong các công việc hàng ngày.

### 1.2. AlphaGo - Google DeepMind

AlphaGo là một hệ thống trí tuệ nhân tạo phát triển bởi Google DeepMind, được thiết kế để chơi cờ vây. Nó đã gây tiếng vang lớn trong cộng đồng khoa học máy tính và cờ vây khi năm 2016, AlphaGo đã đánh bại Lee Sedol - một trong những kỳ thủ hàng đầu thế giới - trong một trận đấu cờ vây trực tiếp.

AlphaGo không chỉ đơn thuần là một chương trình máy tính chơi cờ vây, mà còn ứng dụng kỹ thuật học sâu và học tăng cường để phân tích và đưa ra quyết định trong trò chơi.
Hệ thống này đã đạt được một khả năng đánh cờ đáng kinh ngạc, thậm chí khi đối mặt với những nước đi sáng tạo và phức tạp mà trước đây được xem là khó khăn đối với máy tính.

<video width="1000" controls>
    <source src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/example_alphago.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

Sự thành công của AlphaGo đã thể hiện sự tiến bộ đáng kinh ngạc trong lĩnh vực trí tuệ nhân tạo và Machine Learning, đồng thời mở ra nhiều cơ hội mới trong việc áp dụng trí tuệ nhân tạo vào các lĩnh vực khác nhau.

### 1.3. Autopilot - Tesla

AutoPilot của Tesla là một hệ thống lái tự động được tích hợp vào các xe điện sản xuất bởi Tesla, một công ty đổi mới trong lĩnh vực ô tô tự lái.
AutoPilot cho phép xe tự động thực hiện nhiều tác vụ lái xe, như duy trì làn đường, duyệt qua giao lộ, và tự động điều khiển tốc độ, giữ khoảng cách an toàn.

<video width="1000" controls>
    <source src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/example_tesla_autopilot.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

Hệ thống AutoPilot của Tesla sử dụng nhiều cảm biến, radar, camera và máy tính để thu thập thông tin về môi trường xung quanh và hướng dẫn xe điều khiển an toàn.
Mặc dù được gọi là "lái tự động," AutoPilot vẫn cần sự giám sát của người lái và có thể yêu cầu họ can thiệp trong một số tình huống.

### 1.4. MidJourney

Hình ảnh dưới đây là một hình ảnh được sinh ra bởi MidJourney, một hệ thống trí tuệ nhân tạo và hơn nữa, hình ảnh này đã dành chiến thắng trong một cuộc thi về ảnh.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/example_midjourney.png" style="width: 1200px;"/>

Thẩm mỹ hình ảnh là một lĩnh vực khá trừu tượng và mang tính cá nhân hoá cao.
Nên việc MidJourney nói riêng hay AI nói chung chiến thắng con người trong một cuộc thi như vậy là một bước tiến lớn trong việc ứng dụng trí tuệ nhân tạo vào các lĩnh vực khác nhau cũng như gây ra nhiều tranh cãi về vai trò của con người trong tương lai.

### 1.5. Siri - Apple

Siri là trợ lý ảo phát triển bởi Apple, được tích hợp trên các thiết bị của họ như iPhone, iPad, Mac và các sản phẩm khác.
Siri hoạt động dựa trên trí tuệ nhân tạo để hiểu và thực hiện các lệnh và câu hỏi của người dùng thông qua giọng nói.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/example_siri.jpeg" style="width: 600px;"/>

Siri có khả năng thực hiện nhiều nhiệm vụ, bao gồm gửi tin nhắn, thực hiện cuộc gọi, kiểm tra thời tiết, dự báo giao thông, tìm kiếm thông tin trên mạng, mở ứng dụng, đặt báo thức, và thậm chí điều khiển các thiết bị trong nhà thông qua HomeKit.

Với các phiên bản cập nhật liên tục, Siri ngày càng được cải tiến về khả năng hiểu và phản hồi tự nhiên, từ việc nhận dạng giọng nói đến khả năng hiểu ngữ cảnh và mục đích của người dùng.
Siri đã trở thành một phần quan trọng trong việc tương tác với các thiết bị của Apple, mang lại sự tiện ích và tương tác trực quan cho người dùng.

### 1.6. Social Credit Score - China government

Hệ thống điểm tín dụng xã hội (Social Credit Score) của Chính phủ Trung Quốc là một chương trình theo dõi và đánh giá hành vi của công dân và doanh nghiệp dựa trên nhiều tiêu chí khác nhau.
Mục tiêu chính của chương trình là tạo ra một hệ thống xác định mức độ đáng tin cậy của các cá nhân và tổ chức trong xã hội, từ đó ảnh hưởng đến quyền lợi và ưu đãi mà họ có được.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/example_social_credit_score.jpeg" style="width: 800px;"/>

Hệ thống này sử dụng thông tin từ nhiều nguồn khác nhau như hành vi mua sắm, thanh toán tài chính, việc thực hiện nghĩa vụ công dân, và hoạt động trực tuyến để tính toán điểm tín dụng của mỗi cá nhân và doanh nghiệp.
Những người có điểm cao có thể được hưởng ưu đãi như vay tiền dễ dàng hơn, du lịch dễ dàng hơn, và nhiều quyền lợi khác.
Ngược lại, những người có điểm thấp có thể gặp khó khăn trong việc nhận vay hoặc thậm chí bị hạn chế trong việc di chuyển và hoạt động kinh doanh.

Hệ thống điểm tín dụng xã hội của Trung Quốc đã nhận nhiều ý kiến trái chiều, với một số người cho rằng nó có thể đảm bảo tính trật tự và đạo đức trong xã hội, trong khi những người khác lo ngại về việc xâm phạm quyền riêng tư và nguy cơ rơi vào việc kiểm soát quá mức từ phía chính quyền.

### 1.7. Các ứng dụng trong các lĩnh vực khác nhau

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/example_others.png" style="width: 1200px;"/>

## 2. Trí tuệ nhân tạo - Artificial Intelligence, Máy học - Machine Learning và Học sâu - Deep Learning

### 2.1. Trí tuệ nhân tạo - Artificial Intelligence

Trí tuệ nhân tạo (AI) là một lĩnh vực nghiên cứu và ứng dụng các phương pháp máy tính để mô phỏng và mở rộng khả năng tư duy của con người.

Trí tuệ nhân tạo được quan tâm từ những năm 1950, tuy nhiên, chỉ trong những năm gần đây, AI đã phát triển mạnh mẽ và đóng góp rất nhiều ứng dụng từ hỗ trợ con người đến tự động hóa các công việc.

Trí tuệ nhân tạo giống như là một đích đến cuối cùng, một mục tiêu lớn mà các nhà nghiên cứu và kỹ sư máy tính hướng tới.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/ai_vs_ml_vs_dl.png" style="width: 1200px;"/>

### 2.2. Máy học - Machine Learning

Máy học (Machine Learning) là một phần của trí tuệ nhân tạo, mà mục tiêu là phát triển các kỹ thuật giúp máy tính học từ dữ liệu.
Machine Learning không yêu cầu lập trình cụ thể cho từng tác vụ, mà thay vào đó, Machine Learning sử dụng dữ liệu để học và cải thiện hiệu suất theo thời gian.

Machine Learning nói chung bao gồm nhiều phương pháp và thuật toán được xây dựng từ các nền tảng lý thuyết khác nhau như đại số tuyến tính, xác suất thống kê, mô hình dạng cây, hình học giải tích ...

### 2.3. Học sâu - Deep Learning

Học sâu (Deep Learning) là một phương pháp nằm trong Machine Learning, dựa trên mô hình mạng nơ-ron nhân tạo.
Nói cách khác, Machine Learning sử dụng mạng nơ-ron để học được gọi là Deep Learning.

Deep Learning trong những năm gần đây đã trở thành một trong những lĩnh vực nghiên cứu và ứng dụng phổ biến nhất và thành công nhất của trí tuệ nhân tạo.
Những ứng dụng phổ biến nhất của Deep Learning có thể kể đến như Chat GPT, AlphaGo, Autopilot, và nhiều ứng dụng khác.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/deep_learning_reasons.png" style="width: 1200px;"/>

Về lý thuyết, Deep learning là một phần con của Machine Learning, tuy nhiên, trong thực tế hiện nay, chúng ta có thể chia Machine Learning thành 2 nhóm: Machine Learning truyền thống và Deep Learning.
Lý do mà Deep Learning mạnh mẽ và phổ biến hiện nay nhờ **khả năng hấp thụ dữ liệu lớn** của mô hình mạng nơ ron.

Cụ thể, khi ta tăng lượng dữ liệu lên rất nhiều, thì mô hình Deep learning có khả năng tăng tiến độ chính xác một cách đáng kể, so sánh với các mô hình Machine Learning truyền thống.
Khả năng này của Deep learning được hậu thuẫn bởi sự phát triển của các công nghệ hiện đại về phần cứng như CPU hay GPU nhanh hơn, RAM hay ổ cứng nhanh hơn và nhiều hơn, các phần cứng mới như TPU, ...
Phần cứng máy tính nói chung, hiện nay, có hiệu năng rất tốt trong khi giá thành rất rẻ, giúp cho việc huấn luyện mô hình Deep learning trở nên dễ dàng hơn.

## 3. So sánh giữa Lập trình truyền thống và Machine Learning

### 3.1. Lập trình truyền thống

Lập trình truyền thống là quá trình viết mã máy tính để thực hiện một loạt các hành động cụ thể để giải quyết một vấn đề.
Trong lập trình truyền thống, chúng ta cần xác định từng bước cụ thể để giải quyết vấn đề, từ việc thu thập dữ liệu, xử lý dữ liệu, đưa ra quyết định, và thực hiện hành động.

Lập trình truyền thống thường yêu cầu kiến thức chuyên sâu về ngôn ngữ lập trình, thuật toán, cấu trúc dữ liệu, và nhiều kỹ năng khác.
Nó cũng đòi hỏi thời gian và công sức lớn để viết và duy trì mã nguồn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/traditional_programming_vs_machine_learning.png" style="width: 1200px;"/>

### 3.2. Machine Learning

Machine learning hướng đến việc xây dựng các mô hình hoặc chương trình máy tính có khả năng tự học từ dữ liệu.
Thay vì viết cụ thể từng bước giải quyết vấn đề, Machine Learning cho phép máy tính "học" từ dữ liệu và cải thiện hiệu suất theo thời gian.
Machine learning hướng đến việc xây dựng một giải pháp khái quát và để cho máy tính tự học từ dữ liệu, thay vì viết cụ thể từng bước giải quyết vấn đề.

Machine learning thường được chia làm 2 giai đoạn: Training (Huấn luyện) và Inference (Dự đoán).
- Trong giai đoạn training, chúng ta cung cấp dữ liệu cho mô hình để nó học cách thực hiện dự đoán chính xác.
    - Giai đoạn này có thể mất nhiều thời gian và công sức để huấn luyện mô hình.
    - Các công việc cụ thể trong giai đoạn này bao gồm: thu thập dữ liệu, xử lý dữ liệu, chọn mô hình, huấn luyện mô hình, đánh giá mô hình, phân tích lỗi và điều chỉnh mô hình.
    - Giai đoạn training sẽ kết thúc khi mô hình đạt độ chính xác tốt nhất (tuỳ theo từng bài toán mà ta sẽ có định nghĩa tốt nhất là như thế nào) và sẵn sàng để sử dụng cho giai đoạn inference.
- Trong giai đoạn inference, mô hình được sử dụng để dự đoán kết quả cho dữ liệu mới và triển khai trong môi trường thực tế.
    - Công việc cụ thể trong giai đoạn này bao gồm: thu thập dữ liệu mới, xử lý dữ liệu mới, dự đoán kết quả, và triển khai mô hình.
    - Ngoài ra, trong giai đoạn này, chúng ta cũng cần theo dõi hiệu suất của mô hình và điều chỉnh nếu cần.
    - Hiện nay, đặc biệt là trong các doanh nghiệp, giai đoạn inference cũng được dành nhiều sự quan tâm, vì nó ảnh hưởng trực tiếp đến việc cung cấp dịch vụ cho khách hàng và chi phí mà doanh nghiệp phải bỏ ra.

## 4. Xây dựng Trí tuệ nhân tạo như thế nào?

Để xây dựng mô hình trí tuệ nhân tạo, hiện nay, ngôn ngữ lập trình Python là một trong những ngôn ngữ phổ biến nhất được sử dụng.
Python có nhiều thư viện hỗ trợ mạnh mẽ cho:
- Machine Learning: NumPy, Pandas, Matplotlib, Scikit-learn ...
- Deep Learning: TensorFlow, PyTorch, Keras ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/tools_libraries.png" style="width: 1200px;"/>

Ngoài ngôn ngữ lập trình Python và các thư viện nói trên, với sự quan tâm dành cho Deep learning, những Kỹ sư Trí tuệ nhân tạo cũng có thể cần sử dụng đến ngôn ngữ lập trình khác như C++, CUDA ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/0-ai-introduction/skills_knowledge.jpeg" style="width: 1200px;"/>

Bên cạnh việc sử dụng các ngôn ngữ lập trình và thư viện, để xây dựng mô hình trí tuệ nhân tạo, chúng ta cần có kiến thức về:
- Toán học: Đại số tuyến tính, xác suất thống kê, giải tích, ...
- Lập trình: Ngôn ngữ lập trình Python, database, điện toán đám mây ...
- Kiến thức chuyên ngành: Tài chính, kinh doanh, y học, ...
