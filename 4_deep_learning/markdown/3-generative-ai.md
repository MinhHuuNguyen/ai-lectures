---
time: 03/20/2023
title: Trí tuệ nhân tạo tạo sinh Generative AI
description: Trí tuệ nhân tạo tạo sinh (Generative AI) là lĩnh vực nghiên cứu và ứng dụng trong trí tuệ nhân tạo nhằm tạo ra nội dung mới, bao gồm văn bản, hình ảnh, âm thanh và video. Các mô hình GenAI như GPT, DALL-E, Stable Diffusion đã đạt được những tiến bộ đáng kể trong việc tạo ra nội dung chất lượng cao, mở ra nhiều cơ hội mới trong các lĩnh vực sáng tạo, nghệ thuật và truyền thông. Tuy nhiên, nó cũng đặt ra những thách thức về đạo đức, bảo mật và quyền sở hữu trí tuệ cần được giải quyết.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/llm.jpeg
tags: [deep-learning, generative-ai]
is_highlight: true
is_published: true
---

***Note: Một số nội dung trong bài viết này được cập nhật trong thời gian gần đây.***

## 1. Giới thiệu chung về Generative AI

Generative AI - GenAI (trí tuệ nhân tạo tạo sinh) là một lĩnh vực của AI tập trung vào khả năng tạo ra nội dung mới dựa trên kiến thức học được từ dữ liệu có sẵn.
Khác với AI truyền thống chỉ phân tích hay dự đoán trên dữ liệu, AI tạo sinh có thể sinh ra dữ liệu mới ở nhiều định dạng như văn bản, hình ảnh, âm thanh, video, mã code, thậm chí mô hình 3D.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/gen_ai_predictive_ai.jpeg" style="width: 600px;"/>

Các mô hình GenAI khác với các mô hình Predictive AI (AI dự đoán) ở điểm:
- **Predictive AI** tìm kiếm mối quan hệ ẩn trong dữ liệu để dự đoán kết quả, dự đoán xu hướng, dự đoán giá trị ...
- **Generative AI** cũng tìm kiếm mối quan hệ ẩn trong dữ liệu nhưng hơn nữa, nó tìm cách mô hình hoá và kết hợp các mối quan hệ đó để tạo ra dữ liệu "mới" hoặc thực thi nhiệm vụ nào đó.
Chữ "mới" ở đây có thể hiểu là dữ liệu chưa từng xuất hiện trong tập huấn luyện, nhưng vẫn có cấu trúc và đặc trưng tương tự như dữ liệu đã học.

### 1.1. Lịch sử phát triển

Ý tưởng cho máy móc tạo ra nội dung mới đã xuất hiện từ sớm (ví dụ chatbot ELIZA vào thập niên 1960).
Tuy nhiên, phải đến những năm 2010 trở lại đây, GenAI mới có bước nhảy vọt nhờ sự phát triển của học sâu (deep learning).

Năm 2014 đánh dấu cột mốc quan trọng với sự ra đời của Generative Adversarial Network - GAN và Variational Autoencoder - VAE, những mô hình deep learning đầu tiên có khả năng tự học để tạo ra dữ liệu phức tạp như hình ảnh, thậm chí video.

Tiếp đó, năm 2017, kiến trúc Transformer được giới thiệu, mở đường cho các mô hình sinh ngôn ngữ hiện đại – GPT (Generative Pre-trained Transformer) đầu tiên ra đời năm 2018, rồi GPT-2 (2019), GPT-3 (2020), GPT-3.5 (2022) và GPT-4 (2023) của OpenAI.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/gpt_series.jpeg" style="width: 600px;"/>

Đến năm 2021, loạt mô hình tạo hình ảnh từ văn bản xuất hiện: DALL-E của OpenAI, kế đó Midjourney và Stable Diffusion, đánh dấu khả năng tạo ra tác phẩm nghệ thuật số chất lượng cao từ vài dòng mô tả.

Đặc biệt cuối năm 2022, sự ra mắt của ChatGPT (dựa trên GPT-3.5) đã tạo nên cơn sốt toàn cầu về AI tạo sinh, thu hút hàng triệu người dùng chỉ trong vài ngày.
Công cụ trò chuyện này cho thấy AI có thể tạo nội dung văn bản mạch lạc gần như con người, mở ra kỷ nguyên mới cho AI tạo sinh trong đời sống.

### 1.2. Nguyên lý hoạt động

Về mặt kỹ thuật, các mô hình GenAI thường sử dụng các kiến trúc mạng nơ-ron sâu (deep neural networks) để học từ dữ liệu đầu vào.
Các mô hình này được xây dựng dựa trên các mô hình ngôn ngữ, sau này, các mô hình này được mở rộng để xử lý nhiều loại dữ liệu khác nhau như hình ảnh, âm thanh, video.

Với nền tảng là lượng dữ liệu văn bản khổng lồ trên khắp internet, các mô hình GenAI học cách hiểu ngữ cảnh, cấu trúc ngôn ngữ và kiến thức của thế giới.
Sau đó, với các kỹ thuật để đồng nhất dữ liệu đầu vào, các mô hình LLM có thể nhận thêm hình ảnh hoặc âm thanh, video để tạo ra các đầu ra đa phương thức.
Các loại dữ liệu khác nhau có thể được kết hợp để nâng cao chất lượng đầu ra.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/gen_ai_tasks.jpeg" style="width: 600px;"/>

Bản chất các bài toán của GenAI là các bài toán như text-to-text, text-to-image, text-to-video, text-to-3D, text-to-task, image-to-image, text+image-to-image ...

Chi tiết về lịch sử ra đời của các mô hình GenAI và các kỹ thuật được sử dụng trong các mô hình đó được trình bày trong [bài viết này](/blog/mo-hinh-transformer).

### 1.3. Ứng dụng

Ngày nay GenAI hiện diện trong nhiều lĩnh vực với các ứng dụng phong phú.

Các ứng dụng của GenAI rất đa dạng và đang ngày càng mở rộng, một số ứng dụng nổi bật nhất của GenAI đã được chia sẻ trong [bài viết này](/blog/gioi-thieu-chung-ve-tri-tue-nhan-tao-artificial-intelligence).

## 2. Mô hình ngôn ngữ lớn (Large Language Model - LLM)

Mô hình Ngôn ngữ Lớn (Large Language Model – LLM) là các mô hình học sâu có quy mô cực lớn được huấn luyện trên lượng dữ liệu văn bản khổng lồ.
Các LLM thường có hàng chục đến hàng trăm tỷ tham số, học từ hàng triệu trang văn bản (toàn bộ Wikipedia, sách, web ...) để nắm bắt ngữ pháp, ý nghĩa ngôn ngữ và kiến thức thế giới, có khả năng hiểu và sinh ngôn ngữ tự nhiên.
LLM là nền tảng của GenAI trong xử lý ngôn ngữ tự nhiên – chúng được xem như bộ não của nhiều hệ thống AI hiện đại.

Kiến trúc nền tảng của đa số LLM hiện đại là Transformer và sử dụng các kỹ thuật huấn luyện đặc biệt như Masked Language Modeling (MLM), Standard Language Modeling (SLM), Reinforcement Learning from Human Feedback (RLHF) ...
Kết quả là LLM học được cách dự đoán từ ngữ tiếp theo trong một ngữ cảnh, LLM có khả năng sinh ra văn bản lưu loát, mạch lạc về nhiều chủ đề.

Nội dung chi tiết mô hình Transformer và các mô hình LLM có thể được tìm thấy trong [bài viết này](/blog/mo-hinh-transformer).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/llm.jpeg" style="width: 600px;"/>

Một số mô hình tiêu biểu như: **ChatGPT** của OpenAI, **Gemini** của Google, **Claude** của Anthropic, **LLaMA** của Meta, **DeepSeek** của DeepSeek AI Lab, **Mistral** của Mistral AI, **Grok** của xAI ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/llm2slm.jpeg" style="width: 600px;"/>

Trong thời gian gần đây, một hướng nghiên cứu mới cũng nhận được nhiều sự quan tâm là Small Language Model (SLM) - mô hình ngôn ngữ nhỏ.
SLM là các mô hình ngôn ngữ có kích thước nhỏ hơn LLM nhưng vẫn có khả năng sinh văn bản chất lượng cao.

Một số mô hình SLM tiêu biểu như: **LLaMA 2** của Meta, **Mistral 7B** của Mistral AI, **Falcon** của TII, **Qwen-7B** của Alibaba ...

## 3. Mô hình tạo ảnh (Generative Image Models) và mô hình ngôn ngữ hình ảnh (Vision Language Models - VLM)

Bên cạnh văn bản, GenAI còn có khả năng tạo ra hình ảnh mới – đây là lĩnh vực bùng nổ mạnh mẽ trong vài năm qua.
Hai phương pháp kỹ thuật nổi bật để sinh ảnh bằng AI hiện nay là mạng đối nghịch tạo sinh (GAN) và mô hình khuếch tán (Diffusion).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/vlm.jpeg" style="width: 600px;"/>

Những mô hình này đã chứng minh khả năng sáng tạo vô hạn về thị giác.
Từ một dòng prompt (mô tả) ngắn gọn, AI có thể vẽ ra chân dung, phong cảnh, tranh biếm họa, thiết kế sản phẩm ... theo phong cách tùy chọn.

Năm 2021, sự xuất hiện của DALL-E rồi Midjourney, Stable Diffusion đã đánh dấu thời kỳ nghệ thuật do AI tạo ra đạt chất lượng cao và rất chân thực chỉ từ các lời gợi ý văn bản

Một số mô hình tiêu biểu như: **DALL-E** của OpenAI, **Midjourney**, **Stable Diffusion** của Stability AI, **Imagen** của Google, **Adobe Firefly** của Adobe ...

## 4. Rủi ro và thách thức của Generative AI

### 4.1. Vấn đề ảo giác (hallucination)

Hallucination là hiện tượng mà mô hình GenAI tạo ra thông tin rất khó đọc, khó hiểu hoặc đưa ra thông tin sai lệch, gây hiểu lầm.
Đây là một trong những thách thức lớn nhất của GenAI.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/hallucination_problem.jpeg" style="width: 600px;"/>

Các mô hình GenAI hoạt động như “hộp đen” phức tạp.
Điều này dẫn đến khó khăn trong việc giải thích quyết định của AI và kiểm soát đầu ra.
Mô hình có thể “ảo tưởng” (hallucinate) tạo ra thông tin vô căn cứ mà trông rất có vẻ thuyết phục, gây hiểu lầm cho người dùng.

Một kỹ thuật thường được dùng để giảm thiểu hiện tượng này là **Retrieval-Augmented Generation (RAG)**, trong đó mô hình sẽ truy vấn cơ sở dữ liệu bên ngoài để lấy thêm thông tin trước khi tạo ra câu trả lời.

### 4.2. Sai lệch và định kiến (bias)

GenAI học từ dữ liệu do con người tạo ra từ trước đến nay, nên nó mang theo các định kiến, sai lệch trong dữ liệu đó.
Ví dụ, nếu dữ liệu huấn luyện chứa nhiều định kiến về giới tính, chủng tộc, mô hình có thể tạo ra nội dung phân biệt đối xử, phân biệt chủng tộc, hoặc thể hiện các định kiến xã hội khác.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/bias_problem.jpeg" style="width: 600px;"/>

Đã có trường hợp AI tạo sinh văn bản đưa ra câu trả lời phân biệt chủng tộc hay giới, hoặc AI vẽ hình ảnh với định kiến nghề nghiệp (như bác sĩ thường vẽ thành nam giới).
Việc mô hình khuếch đại định kiến xã hội là nguy hiểm, đòi hỏi phải có biện pháp lọc dữ liệu và điều chỉnh mô hình để giảm bias.

### 4.3. Chi phí tính toán và vấn đề môi trường

Đào tạo và vận hành các mô hình GenAI khổng lồ tiêu tốn nguồn lực tính toán rất lớn, có thể tốn hàng triệu đô la chi phí hạ tầng GPU và điện năng.
Việc tiêu thụ điện năng khổng lồ của các trung tâm dữ liệu để chạy AI đang trở thành vấn đề về phát triển bền vững.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/environment_problem.jpeg" style="width: 600px;"/>

Các mô hình càng lớn, năng lượng càng tăng theo cấp số nhân, gây phát thải carbon đáng kể.
Thách thức đặt ra là làm sao tối ưu mô hình để tiết kiệm năng lượng, hoặc sử dụng nguồn điện tái tạo, nhằm giảm tác động xấu đến môi trường.

### 4.4. Bảo mật, bản quyền và quyền riêng tư

Các mô hình AI lớn được huấn luyện trên dữ liệu internet, có thể bao gồm dữ liệu cá nhân nhạy cảm.
Có lo ngại rằng AI có thể vô tình tái tạo lại thông tin cá nhân hoặc bí mật đã thấy trong dữ liệu huấn luyện.

Quyền riêng tư của người dùng cũng bị đe dọa khi AI có khả năng tổng hợp và phân tích lượng lớn dữ liệu cá nhân.
Tuy nhiên, hiện nay, các công ty lớn phát triển mô hình GenAI đều cam kết tuân thủ các quy định về bảo vệ dữ liệu cá nhân (như GDPR ở châu Âu) và không sử dụng dữ liệu cá nhân mà không có sự đồng ý của người dùng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/privacy_security_problem.jpeg" style="width: 600px;"/>

Về bản quyền, nếu AI được huấn luyện trên hàng triệu hình ảnh hay bài hát có bản quyền, đầu ra của AI được xem là phái sinh từ những tác phẩm đó hay là hoàn toàn mới?

Nhiều nghệ sĩ lo ngại AI “học lỏm” phong cách của họ và tạo ra tác phẩm cạnh tranh.
Đã có các vụ kiện về việc dữ liệu huấn luyện vi phạm bản quyền của nghệ sĩ.
Hiện nay, các bộ luật và quy định về bản quyền liên quan đến AI vẫn đang trong quá trình hình thành và chưa rõ ràng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/responsible_ethical_ai.jpeg" style="width: 600px;"/>

Các hệ thống GenAI có thể được dùng để tự động hóa giao tiếp thay cho con người với chất lượng đáng kinh ngạc.
Tuy nhiên, điều này khiến cho GenAI có thể bị kẻ xấu lợi dụng để tạo nội dung giả mạo phục vụ ý đồ xấu.

Ví dụ: AI có thể tạo ra video deepfake ghép mặt một chính trị gia vào một phát ngôn gây sốc hoặc ảnh giả trong những tình huống không có thật, dẫn đến lan truyền tin giả (fake news) rất thuyết phục

### 4.5. Phương pháp đánh giá và tiếp tục phát triển

Sau khi đã huấn luyện mô hình với hầu như toàn bộ dữ liệu trên Internet, để tiếp tục phát triển GenAI, các nhà nghiên cứu cần phải giải quyết những thách thức mới liên quan đến dữ liệu huấn luyện đa dạng và chất lượng, (đặc biệt là dữ liệu có gán nhãn).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/evaluation_problem.jpeg" style="width: 600px;"/>

Bên cạnh đó, đánh giá chất lượng mô hình tạo sinh cũng không đơn giản (vì đầu ra có thể không cố định đúng/sai mà mang tính sáng tạo, chủ quan).
Người ta phải nghĩ ra các tiêu chí như tính đa dạng, độ chân thực, mức độ phù hợp ý định... để đánh giá mô hình

Việc kiểm thử và kiểm toán mô hình AI (AI audit) trở nên cần thiết để phát hiện lỗi và sai lệch trước khi triển khai rộng.

## 5. Cơ hội nghề nghiệp và tương lai của Generative AI

GenAI được dự đoán sẽ tiếp tục phát triển mạnh mẽ và sâu rộng hơn trong tương lai.
Những mô hình như GPT-4, Google Gemini... cho thấy AI đang tiến tới đa phương thức (multimodal): một mô hình thống nhất có thể hiểu và tạo ra nhiều dạng dữ liệu (văn bản, hình ảnh, âm thanh) cùng lúc.

Điều này mở ra khả năng xây dựng các trợ lý AI tổng quát giống trong phim khoa học viễn tưởng – có thể nghe, nhìn, trò chuyện, và hỗ trợ con người trong mọi tình huống.
Bên cạnh việc “to hơn”, các nghiên cứu cũng đang làm cho mô hình “thông minh hơn”: cải thiện khả năng suy luận logic, ghi nhớ lâu hơn, tương tác cảm xúc...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/reasoning_development.jpeg" style="width: 600px;"/>

Một xu hướng khác là phổ cập và thương mại hóa GenAI trong mọi ngóc ngách đời sống.
Sau thành công của ChatGPT, các “ông lớn” công nghệ đua nhau tích hợp AI tạo sinh vào sản phẩm của mình:
- Microsoft tích hợp GPT-4 vào **bộ Office** (Copilot cho Word, Excel, PowerPoint giúp soạn thảo và phân tích tự động)
- Google tích hợp AI vào **Gmail** và **Google Docs** (hỗ trợ viết email, tài liệu)
- Adobe thêm AI tạo hình ảnh vào **Photoshop** (tính năng Generative Fill)
- Amazon đưa AI vào **trợ lý nhà thông minh Alexa**

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/career.jpeg" style="width: 600px;"/>

GenAI cũng tạo ra nhiều cơ hội nghề nghiệp mới trong các lĩnh vực như:
- **Nhóm Kỹ thuật & nghiên cứu:**
    - **LLM/GenAI Engineer:** xây chatbot, tác tử (agents), hệ hỏi–đáp có RAG, tóm tắt, trích xuất cấu trúc.
    - **Applied ML/Research Engineer:** thử nghiệm mô hình, tinh chỉnh/adapter (LoRA), tối ưu suy luận (quantization, distillation), benchmarking.
    - **Evaluation/AI Quality Engineer:** thiết kế thước đo (factuality, faithfulness, toxicity), xây bộ test, A/B, canary.
    - **LLMOps / MLOps Engineer:** triển khai, giám sát, logging/redaction PII, versioning prompt/mô hình, autoscaling, cost/latency.
    - **Generative Media Engineer (Image/Audio/Video):** pipeline tạo ảnh (Stable Diffusion, ControlNet), TTS/ASR, dựng video ngắn, hậu kỳ.
- **Nhóm Sản phẩm, thiết kế, kinh doanh:**
    - **AI Product Manager:** xác định bài toán/ROI, chọn kiến trúc (RAG vs. fine-tune), lộ trình tính năng, tuân thủ.
    - **AI UX/Conversation Designer & UX Writer:** thiết kế hội thoại tự nhiên, kiểm soát kỳ vọng người dùng, microcopy, luồng fallback khi AI “không biết”.
    - **AI Solutions Architect / Customer Engineer (Pre‑sales):** thiết kế giải pháp GenAI theo nhu cầu doanh nghiệp, PoC, tính toán chi phí.
    - **AI Content/Creative Producer (Marketing/Studio/Game/VFX):** tạo asset ảnh/video/giọng nói, concept art, biến thể chiến dịch.
- **Nhóm Pháp lý, an toàn, quản trị:**
    - **AI Safety/Policy/Red Team:** stress‑test jailbreak/prompt‑injection, tiêu chuẩn an toàn, đánh giá rủi ro.
    - **Legal/Compliance (IP, dữ liệu cá nhân):** đánh giá license dữ liệu, xử lý yêu cầu gỡ/xóa, tư vấn bản quyền & PII.
- **Nhóm Chuyên gia miền (Domain Expert + AI):**
    - **Healthcare/Finance/Law + AI:** tri thức chuyên ngành + RAG/tự động hoá quy trình đặc thù.
    - **Giáo dục/Đào tạo/Enablement:** huấn luyện đội ngũ dùng AI, biên soạn guideline, workshop.

Một số kỹ năng cần thiết cho các vị trí trong lĩnh vực GenAI:
- **Nền tảng chung:** Tư duy dữ liệu, xác suất/bayesian cơ bản, đạo đức & riêng tư, viết tài liệu rõ ràng.
- **LLM & RAG:** embeddings, vector store, retrieval/rerank, tách đoạn (semantic/heading), trích dẫn nguồn, cache, guardrails.
- **Đánh giá:** bộ test kịch bản, chấm điểm bán tự động, human review, tiêu chí an toàn.
- **Vận hành:** CI/CD prompt & model, quan sát (latency, cost, error, toxicity, hallucination rate).
- **Thị giác/âm thanh:** pipeline diffusion (SD/ControlNet), TTS/ASR cơ bản.
- **Cloud & tích hợp:** AWS/GCP/Azure, API dịch vụ LLM, kho bí mật, RBAC/DLP.
- **Quản trị & pháp lý:** C2PA/provenance/watermark (khái niệm), license dữ liệu, quy trình phê duyệt.
