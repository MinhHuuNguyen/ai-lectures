---
time: 06/15/2025
title: Tác tử Trí tuệ nhân tạo AI Agent
description:
banner_url:
tags: [deep-learning, generative-ai, ai-agent]
is_highlight: true
is_published: true
---

## 1. Các khái niệm trong AI Agent

### 1.1. Tác tử Trí tuệ nhân tạo - AI Agent

AI Agent là một thực thể phần mềm AI tự chủ, được thiết kế để thực hiện một nhiệm vụ cụ thể hoặc giải quyết một vấn đề trong phạm vi xác định mà không cần người giám sát liên tục.
Một AI agent có thể tiếp nhận đầu vào, xử lý thông tin, và thực hiện hành động để đạt mục tiêu đề ra.

Khác với các hệ thống tự động hoá thông thường, AI Agent có mức độ "linh hoạt" cao hơn, đưa ra các output dựa tương ứng với các input và ngữ cảnh cụ thể mà không cần phải lập trình cứng (hard-code) cho từng tình huống.
Chính vì sự linh hoạt này, AI Agent có thể được áp dụng trong nhiều lĩnh vực khác nhau như trợ lý ảo, chatbot, hệ thống tự động hoá quy trình (RPA), và nhiều ứng dụng AI khác.

Mặc dù có thể hoạt động độc lập, mỗi AI agent thường chỉ đảm trách một nhiệm vụ hẹp, trong phạm vi hay domain cụ thể đã được định trước.


Dưới góc độ kỹ thuật của deep learning, khái niệm Agent không phải là mới.
Khái niệm Agent đã xuất hiện trong lĩnh vực Reinforcement Learning (RL) từ những năm 1990, nơi các tác tử (agents) học cách tương tác với môi trường (environment) để tối đa hoá phần thưởng (reward).

Và quan trọng hơn hết, một yếu tố quan trọng của AI Agent có khả năng tự học hỏi, tự sửa lỗi và tự cải thiện hiệu suất của mình theo thời gian.

### 1.2. Agentic AI

Agentic AI là các hệ thống AI có tính “agentic” mạnh mẽ, nghĩa là có khả năng tự đưa ra quyết định và hành động để đạt được mục tiêu tổng quát, thường phối hợp nhiều "tools - công cụ" và nhiều "agent - tác tử" với nhau một cách linh hoạt.

Nói cách khác, Agentic AI là bước phát triển cao hơn của AI agent truyền thống, vượt khỏi các tác vụ hẹp để xử lý những mục tiêu lớn và phức tạp.
Nhờ sự phối hợp này, Agentic AI hoạt động chủ động và linh hoạt hơn so với một AI agent đơn lẻ.
Nó có thể tự đề xuất mục tiêu mới, tự chia nhỏ nhiệm vụ, tự chọn công cụ và điều chỉnh kế hoạch khi gặp tình huống bất ngờ, thay vì chỉ phản ứng thụ động theo kịch bản lập sẵn.

Agentic AI được đặc trưng bởi bốn năng lực chính:
- Ra quyết định tự động (autonomous decision-making): phân tích tình huống và hành động độc lập không chỉ dựa trên các luật cố định.
- Hành động hướng mục tiêu (goal-driven): biết lập kế hoạch và thực thi chuỗi hành động nhiều bước để đạt mục tiêu đề ra.
- Học hỏi và thích nghi (learning and adaptation): tự cải thiện hiệu suất dựa trên kinh nghiệm và điều chỉnh chiến lược theo thời gian thực.
- Lý luận nâng cao (advanced reasoning): có khả năng phối hợp nhiều hệ thống, công cụ, cơ sở dữ liệu để giải quyết những quy trình phức tạp một cách tự chủ.

| **Khía cạnh**           | **Hệ thống AI truyền thống**                             | **Agentic AI (AI tác nhân tự chủ)**                                              |
| ----------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Phạm vi nhiệm vụ**    | Hẹp, cụ thể (một nhiệm vụ hoặc domain nhất định)         | Rộng, phức tạp (nhiều nhiệm vụ liên kết hoặc mục tiêu lớn)                       |
| **Tính tự chủ**         | Thấp – làm theo kịch bản cố định, chờ đầu vào            | Cao – tự đề ra bước hành động, chủ động theo đuổi mục tiêu                       |
| **Học hỏi thích nghi**  | Thụ động – cần con người cập nhật để cải thiện           | Chủ động – tự học từ phản hồi, thích nghi thời gian thực                         |
| **Kỹ năng & công cụ**   | Thường một kỹ năng chính (vd. chỉ NLP hoặc chỉ thị giác) | Kết hợp đa kỹ năng, sử dụng nhiều công cụ (tìm kiếm, API…)                       |
| **Lập luận & Kế hoạch** | Hạn chế, theo quy tắc sẵn có                             | Lập luận sâu, lên kế hoạch đa bước linh hoạt                                     |
| **Phối hợp tác nhân**   | Không (tác tử đơn lẻ)                                    | Có thể bao gồm nhiều tác tử phối hợp (multi-agent)                               |
| **Ví dụ**               | Chatbot FAQ cố định, hệ nhận diện khuôn mặt              | AutoGPT đa năng thực hiện dự án, hệ thống trợ lý doanh nghiệp tích hợp nhiều bot |

### 1.3. Mô hình Ngôn ngữ lớn - Large Language Model (LLM)

Mô hình Ngôn ngữ Lớn (Large Language Model – LLM) là các mô hình học sâu có quy mô cực lớn được huấn luyện trên lượng dữ liệu văn bản khổng lồ.
Các LLM thường có hàng chục đến hàng trăm tỷ tham số, học từ hàng triệu trang văn bản (toàn bộ Wikipedia, sách, web ...) để nắm bắt ngữ pháp, ý nghĩa ngôn ngữ và kiến thức thế giới, có khả năng hiểu và sinh ngôn ngữ tự nhiên.
LLM là nền tảng của GenAI trong xử lý ngôn ngữ tự nhiên – chúng được xem như bộ não của nhiều hệ thống AI hiện đại.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/llm.png" style="width: 600px;"/>

Trong hệ thống AI agent, LLM thường đảm nhiệm vai trò “bộ não” của tác tử, cung cấp khả năng lý luận (reasoning) và hiểu ngữ cảnh cho agent.
Nhờ có LLM, agent có thể phân tích yêu cầu dưới dạng ngôn ngữ tự nhiên của người dùng, suy diễn ra các bước cần làm, và sinh ra các hành động/ câu trả lời phù hợp.

LLM cho phép AI agent thoát khỏi cách điều khiển cứng (hard-code) thông thường, thay vào đó linh hoạt ứng phó với nhiều tình huống mới dựa vào ngữ cảnh và kiến thức đã học.

### 1.4. Công cụ - Tools

## 2. Một số kỹ thuật được sử dụng trong AI Agent

### 2.1. Function Calling

### 2.2. Chain of Thought (CoT)

### 2.3. Retrieval-Augmented Generation (RAG)

## 3. Kiến trúc tổng quát của AI Agent

### 3.1. Reasoning - Lập luận

### 3.2. Act - Hành động

### 3.3. Memory - Bộ nhớ

### 3.4. Tools use - Sử dụng công cụ

### 3.5. Kiến trúc tổng quát

### 3.6. Ví dụ minh hoạ

## 4. Công cụ xây dựng AI Agent

| Lớp trong “stack” Agentic AI *điển hình*     | Ví dụ sản phẩm                                               |
| -------------------------------------------- | ------------------------------------------------------------ |
| **1. Model / Foundation**                    | ChatGPT, Claude, Gemini                                      |
| **2. Agent Framework & Reasoning**           | LangChain, LlamaIndex, CrewAI, AutoGPT                       |
| **3. Low‑/No‑code Automation & Integration** | n8n, Zapier Agents, Make AI Agents, Microsoft Power Automate |
| **4. Ứng dụng / Domain Layer**               | Sản phẩm doanh nghiệp, giải pháp vertical                    |

### 4.1. Foundation models

### 4.2. Agent Framework và Reasoning

### 4.3. Low‑/No‑code Automation & Integration
