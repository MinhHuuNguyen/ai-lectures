---
time: 06/15/2025
title: Tác tử Trí tuệ nhân tạo AI Agent
description: Large Language Model (LLM) phát triển mạnh mẽ đã mở ra kỷ nguyên mới cho trí tuệ nhân tạo. Trong thời gian gần đây, AI Agent đã trở thành một trong những lĩnh vực nghiên cứu và ứng dụng nổi bật nhất trong AI, với khả năng tự động hoá các tác vụ phức tạp và tương tác linh hoạt với người dùng. AI Agent không chỉ là một chatbot hay trợ lý ảo đơn giản, mà là những hệ thống AI có khả năng tự học hỏi, tự lập kế hoạch và phối hợp nhiều công cụ để giải quyết các vấn đề phức tạp.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/banner.png
tags: [deep-learning, generative-ai, ai-agent]
is_highlight: true
is_published: true
---

## 1. Các khái niệm trong AI Agent

### 1.1. Mô hình Ngôn ngữ lớn - Large Language Model (LLM)

Mô hình Ngôn ngữ Lớn (Large Language Model – LLM) là các mô hình học sâu có quy mô cực lớn được huấn luyện trên lượng dữ liệu văn bản khổng lồ.
Các LLM thường có hàng chục đến hàng trăm tỷ tham số, học từ hàng triệu trang văn bản (toàn bộ Wikipedia, sách, web ...) để nắm bắt ngữ pháp, ý nghĩa ngôn ngữ và kiến thức thế giới, có khả năng hiểu và sinh ngôn ngữ tự nhiên.
LLM là nền tảng của GenAI trong xử lý ngôn ngữ tự nhiên – chúng được xem như bộ não của nhiều hệ thống AI hiện đại.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/3-generative-ai/llm.png" style="width: 600px;"/>

Kiến trúc nền tảng của đa số LLM hiện đại là Transformer và sử dụng các kỹ thuật huấn luyện đặc biệt như Masked Language Modeling (MLM), Standard Language Modeling (SLM), Reinforcement Learning from Human Feedback (RLHF) ...
Kết quả là LLM học được cách dự đoán từ ngữ tiếp theo trong một ngữ cảnh, LLM có khả năng sinh ra văn bản lưu loát, mạch lạc về nhiều chủ đề.

Nội dung chi tiết mô hình Transformer và các mô hình LLM có thể được tìm thấy trong [bài viết này](/blog/mo-hinh-transformer).

Một số mô hình tiêu biểu như: **ChatGPT** của OpenAI, **Gemini** của Google, **Claude** của Anthropic, **LLaMA** của Meta, **DeepSeek** của DeepSeek AI Lab, **Mistral** của Mistral AI, **Grok** của xAI ...

### 1.2. Tác tử Trí tuệ nhân tạo - AI Agent

AI Agent là một thực thể phần mềm AI tự chủ, được thiết kế để thực hiện một nhiệm vụ cụ thể hoặc giải quyết một vấn đề trong phạm vi xác định mà không cần người giám sát liên tục.
Một AI Agent có thể tiếp nhận đầu vào, xử lý thông tin, và thực hiện hành động để đạt mục tiêu đề ra.

Khác với các hệ thống tự động hoá thông thường, AI Agent có mức độ "linh hoạt" cao hơn, đưa ra các output dựa tương ứng với các input và ngữ cảnh cụ thể mà không cần phải lập trình cứng (hard-code) cho từng tình huống.
Chính vì sự linh hoạt này, AI Agent có thể được áp dụng trong nhiều lĩnh vực khác nhau như trợ lý ảo, chatbot, hệ thống tự động hoá quy trình (RPA), và nhiều ứng dụng AI khác.

Mặc dù có thể hoạt động độc lập, mỗi AI Agent thường chỉ đảm trách một nhiệm vụ hẹp, trong phạm vi hay domain cụ thể đã được định trước.

<img src="" style="width: 600px;"/>

Dưới góc độ kỹ thuật của deep learning, khái niệm Agent không phải là mới.
Khái niệm Agent đã xuất hiện trong lĩnh vực Reinforcement Learning (RL) từ những năm 1990, nơi các tác tử (agents) học cách tương tác với môi trường (environment) để tối đa hoá phần thưởng (reward).

Và quan trọng hơn hết, một yếu tố quan trọng của AI Agent có khả năng tự học hỏi, tự sửa lỗi và tự cải thiện hiệu suất của mình theo thời gian.

Trong hệ thống AI Agent, LLM thường đảm nhiệm vai trò “bộ não” của tác tử, cung cấp khả năng lý luận (reasoning) và hiểu ngữ cảnh cho agent.
Bên cạnh "bộ não" LLM, AI Agent còn có một danh sách các công cụ (tools) được coi như "cơ bắp" của tác tử, cho phép nó thực hiện các hành động cụ thể như tìm kiếm thông tin, truy cập cơ sở dữ liệu, hoặc tương tác với các hệ thống khác.

### 1.3. Agentic AI

Agentic AI là các hệ thống AI có tính “agentic” mạnh mẽ, nghĩa là có khả năng tự đưa ra quyết định và hành động để đạt được mục tiêu tổng quát, thường phối hợp nhiều "tools - công cụ" và nhiều "agent - tác tử" với nhau một cách linh hoạt. Hai hình ảnh dưới đây được lấy từ bài báo [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/metagpt_human_align.png" style="width: 1000px;"/>

Nói cách khác, Agentic AI là bước phát triển cao hơn của AI Agent, vượt khỏi các tác vụ hẹp để xử lý những mục tiêu lớn và phức tạp.
Nhờ sự phối hợp này, Agentic AI hoạt động chủ động và linh hoạt hơn so với một AI Agent đơn lẻ.
Nó có thể tự đề xuất mục tiêu mới, tự chia nhỏ nhiệm vụ, tự chọn công cụ và điều chỉnh kế hoạch khi gặp tình huống bất ngờ, thay vì chỉ phản ứng thụ động theo kịch bản lập sẵn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/metagpt_example.png" style="width: 1000px;"/>

Agentic AI được đặc trưng bởi bốn năng lực chính:
- **Ra quyết định tự động (autonomous decision-making):** phân tích tình huống và hành động độc lập không chỉ dựa trên các luật cố định.
- **Hành động hướng mục tiêu (goal-driven):** biết lập kế hoạch và thực thi chuỗi hành động nhiều bước để đạt mục tiêu đề ra.
- **Học hỏi và thích nghi (learning and adaptation):** tự cải thiện hiệu suất dựa trên kinh nghiệm và điều chỉnh chiến lược theo thời gian thực.
- **Lý luận nâng cao (advanced reasoning):** có khả năng phối hợp nhiều hệ thống, công cụ, cơ sở dữ liệu để giải quyết những quy trình phức tạp một cách tự chủ.

| **Khía cạnh**           | **Hệ thống AI truyền thống**                             | **Agentic AI (AI tác nhân tự chủ)**                        |
| ----------------------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| **Phạm vi nhiệm vụ**    | Hẹp, cụ thể (một nhiệm vụ hoặc domain nhất định)         | Rộng, phức tạp (nhiều nhiệm vụ liên kết hoặc mục tiêu lớn) |
| **Tính tự chủ**         | Thấp – làm theo kịch bản cố định, chờ đầu vào            | Cao – tự đề ra bước hành động, chủ động theo đuổi mục tiêu |
| **Học hỏi thích nghi**  | Thụ động – cần con người cập nhật để cải thiện           | Chủ động – tự học từ phản hồi, thích nghi thời gian thực   |
| **Kỹ năng & công cụ**   | Thường một kỹ năng chính (vd. chỉ NLP hoặc chỉ thị giác) | Kết hợp đa kỹ năng, sử dụng nhiều công cụ (tìm kiếm, API…) |
| **Lập luận & Kế hoạch** | Hạn chế, theo quy tắc sẵn có                             | Lập luận sâu, lên kế hoạch đa bước linh hoạt               |
| **Phối hợp tác nhân**   | Không (tác tử đơn lẻ)                                    | Có thể bao gồm nhiều tác tử phối hợp (multi-agent)         |

## 2. Một số kỹ thuật được sử dụng trong AI Agent

### 2.1. Function Calling

Kỹ thuật Function Calling, được giới thiệu bởi OpenAI, cho phép người dùng định nghĩa các hàm (functions) hoặc API, từ đó, mô hình LLM sẽ có thể sinh ra các output dưới dạng JSON để quyết định có nên gọi hàm hay không?, nên gọi hàm nào? và với các tham số gì?.
Đây là phương pháp để kết nối mô hình LLM với các hệ thống bên ngoài, cho phép mô hình thực hiện các hành động cụ thể dựa trên đầu ra của nó.

Hình ảnh dưới đây là ví dụ mà OpenAI giới thiệu về cách sử dụng kỹ thuật Function Calling trong mô hình LLM của họ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/function_calling_1.png" style="width: 1000px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/function_calling_2.png" style="width: 1000px;"/>

Ở thời điểm mà Function Calling ra đời, LLM được yêu cầu cần sinh ra output với cấu trúc chính xác để có thể xác định được hàm nào sẽ được gọi và các tham số cần thiết.
Tuy nhiên, với sự phát triển của các mô hình LLM hiện đại, hầu như tất cả các LLM hiện nay đều có thể sinh ra output với cấu trúc chính xác khi cần thiết.

### 2.2. Chain of Thought (CoT)

Chain of Thought (CoT) là một kỹ thuật được dùng khi sử dụng các mô hình ngôn ngữ lớn (LLM), giúp mô hình thực hiện các bước suy luận logic để giải quyết vấn đề phức tạp.

Hình ảnh dưới đây được lấy từ bài báo [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) mô tả kỹ thuật Chain of Thought (CoT) trong quá trình chuẩn bị prompt đầu vào cho mô hình LLM.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/cot_example_compare.png" style="width: 1000px;"/>

Với việc đưa ra ví dụ về các bước suy luận trong prompt, mô hình LLM có thể dựa vào các bước này để hiểu cách tiếp cận vấn đề và áp dụng logic tương tự cho các câu hỏi mới.
Kỹ thuật này đã chứng minh hiệu quả trong việc cải thiện độ chính xác của các mô hình LLM trong các tác vụ yêu cầu suy luận phức tạp, nhiều bước.

Chain of Thought (CoT) cũng là kỹ thuật thường được khuyên sử dụng khi prompting để cải thiện chất lượng đầu ra của các mô hình nổi tiếng như ChatGPT, Claude, Gemini ...
Tuy nhiên, để áp dụng được CoT, người dùng cần dành nhiều thời gian để chuẩn bị các ví dụ và hướng dẫn cụ thể cho mô hình và đưa vào prompt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/cot_example_results.png" style="width: 1000px;"/>

Chain of Thought (CoT) là nền tảng cho các kỹ thuật reasoning (lập luận) trong AI Agent, cho phép mô hình không chỉ trả lời câu hỏi mà còn giải thích quá trình suy nghĩ của nó, từ đó giúp người dùng hiểu rõ hơn về cách mà mô hình đưa ra quyết định.
Những nghiên cứu sau này, quá trình reasoning của mô hình LLM được cải thiện không chỉ bằng việc đưa CoT vào prompt, mà còn tác động nhiều hơn từ quá trình huấn luyện mô hình.

### 2.3. Retrieval-Augmented Generation (RAG)

GenAI nói chung và các LLM nói riêng hiện đang phải đối mặt với nhiều vấn đề và thách thức như đã được đề cập trong [bài viết này](/blog/tri-tue-nhan-tao-tao-sinh-generative-ai).
Một trong những vấn đề lớn nhất là **ảo giác (hallucination)**, tức là mô hình GenAI tạo ra thông tin rất khó đọc, khó hiểu hoặc đưa ra thông tin sai lệch, gây hiểu lầm.
Đây là một trong những thách thức lớn nhất của GenAI.

Mô hình có thể “ảo tưởng” (hallucinate) tạo ra thông tin vô căn cứ mà trông rất có vẻ thuyết phục, gây hiểu lầm cho người dùng.
Một kỹ thuật thường được dùng để giảm thiểu hiện tượng này là **Retrieval-Augmented Generation (RAG)**, trong đó mô hình sẽ truy vấn cơ sở dữ liệu bên ngoài để lấy thêm thông tin trước khi tạo ra câu trả lời.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/llm_rag.png" style="width: 800px;"/>

RAG (Retrieval-Augmented Generation – Tạo sinh có truy xuất thông tin) là một kỹ thuật kết hợp giữa mô hình generative và hệ thống truy xuất thông tin nhằm nâng cao độ chính xác và khả năng cập nhật kiến thức của mô hình AI.
Thay vì chỉ dựa vào dữ liệu huấn luyện tĩnh, một mô hình sử dụng RAG sẽ truy vấn một cơ sở tri thức bên ngoài (như bằng công cụ tìm kiếm, cơ sở dữ liệu, kho tài liệu ...) và kết hợp các thông tin đã tìm kiếm được vào prompt làm đầu vào cho mô hình LLM.

Nói cách khác, RAG bổ sung thêm bước retrieval (truy xuất) vào giữa quá trình để cung cấp cho LLM những kiến thức mới nhất hoặc chuyên biệt mà nó chưa biết, sau đó LLM mới generate (sinh câu trả lời) dựa trên cả kiến thức truy xuất được và kiến thức nội tại đã được huấn luyện từ trước đó.

Phương pháp này giúp giảm ảo giác (hallucination) của LLM (do câu trả lời được neo vào các tài liệu thực), đồng thời cho phép hệ thống cập nhật thông tin theo thời gian thực mà không cần huấn luyện lại mô hình nền tảng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/llm_rag.png" style="width: 800px;"/>

Hình ảnh trên được lấy từ bài báo [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) mô tả quá trình RAG đơn giản gồm các bước:
- **Bước 1: Indexing** Thu thập, xử lý và lưu trữ tài liệu vào vector database.
- **Bước 2: Retrieval** Khi có truy vấn từ người dùng, hệ thống sẽ truy xuất các tài liệu liên quan từ vector database.
- **Bước 3: Generation** Kết hợp các tài liệu truy xuất được với truy vấn để tạo thành prompt đầu vào cho mô hình LLM. Mô hình LLM sẽ sinh ra câu trả lời dựa vào prompt này.

## 3. Kiến trúc tổng quát của AI Agent

### 3.1. Reasoning - Act - ReAct

**Reasoning** là quá trình LLM tạo ra chuỗi suy luận trung gian để đi tới đáp án.
Quá trình này được thực hiện bằng việc tác động vào mô hình LLM thông qua quá trình prompting hoặc thông qua việc chuẩn bị các bộ dữ liệu tập trung vào quá trình reasoning và finetune mô hình LLM với các bộ dữ liệu này.

**Act** là quá trình LLM lựa chọn các hàm cụ thể và chuẩn bị các tham số để gọi hàm.
Các kết quả return của hàm sẽ được tiếp tục sử dụng làm giàu cho prompt của LLM và ta kỳ vọng rằng những kết quả này sẽ giúp LLM đưa ra các thông tin hay câu trả lời chính xác hơn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/react_example_1.png" style="width: 800px;"/>

**ReAct** là một phương pháp kết hợp giữa Reasoning và Action, được giới thiệu trong bài báo [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).
ReAct xen kẽ Reasoning và Action trong quá trình LLM xử lý một tác vụ, cho phép mô hình vừa suy luận vừa thực hiện hành động một cách linh hoạt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/react_example_2.png" style="width: 800px;"/>

Việc kết hợp xen kẽ giữa Reasoning và Action giúp mô hình LLM tận dụng tốt hơn các kết quả thu được từ các hàm đã gọi (Act), từ đó, đưa ra tiếp tục quyết định về việc có nên gọi thêm các hàm khác hay không, hoặc tiếp tục suy luận để đi tới đáp án cuối cùng.

### 3.2. Memory

Memory là một thành phần quan trọng trong hệ thống AI Agent, giúp Agent lưu trữ quan sát (episodic) và kiến thức rút gọn (semantic), sau đó sử dụng theo từng ngữ cảnh đầu vào khác nhau.

Hình ảnh dưới đây được lấy từ bài báo [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) mô tả vai trò của Memory trong hệ thống AI Agent.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/memory.png" style="width: 800px;"/>

Có hai loại memory chính trong AI Agent:
- **Short Term Memory:** Lưu trữ thông tin tạm thời trong một phiên làm việc, thường là các thông tin liên quan đến cuộc trò chuyện hiện tại hoặc các tác vụ đang thực hiện trong phiên làm việc đó.
    - Việc các hệ thống Agent sử dụng ReAct, liên tục Act để lấy kết quả và tiếp tục Reasoning, sẽ tạo ra một lượng thông tin rất lớn trong quá trình làm việc trước khi phản hồi lại cho người dùng.
    Các thông tin này sẽ được lưu trữ trong Short Term Memory.
    - Sau khi nhận phản hồi, người dùng có thể tiếp tục tương tác với Agent như đưa ra yêu cầu tương tự, đưa ra yêu cầu cụ thể hơn ...
    Các input này từ người dùng cũng sẽ được lưu trữ trong Short Term Memory.
    - Các dữ liệu trong Short Term Memory thường được đưa vào LLM trực tiếp thông qua prompt đầu vào.
- **Long Term Memory:** Lưu trữ thông tin lâu dài, bao gồm kiến thức cụ thể cho một lĩnh vực nào đó hoặc các thông tin liên quan cá nhân hoá liên quan đến người dùng mà hệ thống đang phục vụ.
Long Term Memory thường được sử dụng để cải thiện khả năng cá nhân hoá và hiểu biết của Agent về người dùng hoặc lĩnh vực cụ thể.
    - Các dữ liệu trong Long Term Memory thường được lưu trữ trong các vector database hoặc các hệ thống lưu trữ dữ liệu khác.
    - Các database này được coi như là một tool và LLM sẽ tự reasoning để quyết định có nên truy xuất dữ liệu từ Long Term Memory hay không.
    - Các kỹ thuật thiết kế Long Term Memory dựa trên các phương pháp như RAG (Retrieval-Augmented Generation).

### 3.3. Tools use

Ví dụ, với danh sách các tools như sau, hình ảnh này được lấy từ bài báo [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/tools.png" style="width: 800px;"/>

Các tools được sử dụng trong bài báo này đa dạng gồm:
- Sử dụng một mô hình ngôn ngữ khác (Question Answering, Machine Translation)
- Công cụ tìm kiếm (Wikipedia Search)
- Công cụ tính toán (Calculator)
- Lịch (Calendar)

Và để huấn luyện mô hình LLM sử dụng các tools này, tác giả đã xây dựng bộ prompt như sau:

- Đối với **Question Answering API**:
```
Your task is to add calls to a Question Answering API to a piece of text.
The questions should help you get information required to complete the text.
You can call the API by writing "[QA(question)]" where "question" is the question you want to ask.
Here are some examples of API calls:

Input: Joe Biden was born in Scranton, Pennsylvania.
Output: Joe Biden was born in [QA("Where was Joe Biden born?")] Scranton, [QA("In which state is Scranton?")] Pennsylvania.

Input: Coca-Cola, or Coke, is a carbonated soft drink manufactured by the Coca-Cola Company.
Output: Coca-Cola, or [QA("What other name is Coca-Cola known by?")] Coke, is a carbonated soft drink manufactured by [QA("Who manufactures Coca-Cola?")] the Coca-Cola Company.

Input: x
Output:
```

- Đối với **Calculator API**:
```
Your task is to add calls to a Calculator API to a piece of text.
The calls should help you get information required to complete the text.
You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed.
Here are some examples of API calls:

Input: The number in the next term is 18 + 12 x 3 = 54.
Output: The number in the next term is 18 + 12 x 3 = [Calculator(18 + 12 * 3)] 54.

Input: The population is 658,893 people. This is 11.4% of the national average of 5,763,868 people.
Output: The population is 658,893 people. This is 11.4% of the national average of [Calculator(658,893 / 11.4%)] 5,763,868 people.

Input: A total of 252 qualifying matches were played, and 723 goals were scored (an average of 2.87 per match). This is three times less than the 2169 goals last year.
Output: A total of 252 qualifying matches were played, and 723 goals were scored (an average of [Calculator(723 / 252)] 2.87 per match). This is twenty goals more than the [Calculator(723 - 20)] 703 goals last year.

Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years.
Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011 - 1994)] 17 years.

Input: From this, we have 4 * 30 minutes = 120 minutes.
Output: From this, we have 4 * 30 minutes = [Calculator(4 * 30)] 120 minutes.

Input: x
Output:
```

- Đối với **Wikipedia Search API**:
```
Your task is to complete a given piece of text.
You can use a Wikipedia Search API to look up information.
You can do so by writing "[WikiSearch(term)]" where "term" is the search term you want to look up.
Here are some examples of API calls:

Input: The colors on the flag of Ghana have the following meanings: red is for the blood of martyrs, green for forests, and gold for mineral wealth.
Output: The colors on the flag of Ghana have the following meanings: red is for [WikiSearch("Ghana flag red meaning")] the blood of martyrs, green for forests, and gold for mineral wealth.

Input: But what are the risks during production of nanomaterials? Some nanomaterials may give rise to various kinds of lung damage.
Output: But what are the risks during production of nanomaterials? [WikiSearch("nanomaterial production risks")] Some nanomaterials may give rise to various kinds of lung damage.

Input: Metformin is the first-line drug for patients with type 2 diabetes and obesity.
Output: Metformin is the first-line drug for [WikiSearch("Metformin first-line drug")] patients with type 2 diabetes and obesity.

Input: x
Output:
```

- Đối với **Machine Translation API**:
```
Your task is to complete a given piece of text by using a Machine Translation API.
You can do so by writing "[MT(text)]" where text is the text to be translated into English.
Here are some examples:

Input: He has published one book: Ohomem suprimido (“The Supressed Man”)
Output: He has published one book: OOhomem suprimido [MT(O homem suprimido)] (“The Supressed Man”)

Input: In Morris de Jonge’s Jeschuah, der klassische jüdische Mann, there is a description of a Jewish writer
Output: In Morris de Jonge’s Jeschuah, der klassische jüdische Mann [MT(derklassische jüdische Mann)], there is a description of a Jewish writer

Input: 南 京 高 淳 县 住 房 和 城 乡 建 设 局 城 市 新 区 设 计 a plane of reference Gaochun is one of seven districts of the provincial capital Nanjing
Output: [MT(南京高淳县住房和城乡建设局 城市新区 设计)] a plane of reference Gaochun is one of seven districts of the provincial capital Nanjing

Input: x
Output:
```

- Đối với **Calendar API**:
```
Your task is to add calls to a Calendar API to a piece of text.
The API calls should help you get information required to complete the text.
You can call the API by writing "[Calendar()]"
Here are some examples of API calls:

Input: Today is the first Friday of the year.
Output: Today is the first [Calendar()] Friday of the year.

Input: The president of the United States is Joe Biden.
Output: The president of the United States is [Calendar()] Joe Biden.

Input: The current day of the week is Wednesday.
Output: The current day of the week is [Calendar()] Wednesday.

Input: The number of days from now until Christmas is 30.
Output: The number of days from now until Christmas is [Calendar()] 30.

Input: The store is never open on the weekend, so today it is closed.
Output: The store is never open on the weekend, so today [Calendar()] it is closed.

Input: x
Output:
```

Bằng cách này, LLM có thể học được cách sử dụng các tools một cách hợp lý, từ đó, tạo ra các output chính xác và có ý nghĩa hơn.

### 3.4. Kiến trúc tổng quát

## 4. Công cụ xây dựng AI Agent

| Lớp trong “stack” Agentic AI *điển hình*     | Ví dụ sản phẩm                                               |
| -------------------------------------------- | ------------------------------------------------------------ |
| **1. Model / Foundation**                    | ChatGPT, Claude, Gemini, ...                                 |
| **2. Agent Framework & Reasoning**           | LangChain, LlamaIndex, CrewAI, AutoGPT, ...                  |
| **3. Low‑/No‑code Automation & Integration** | n8n, Zapier Agents, Make AI Agents, Microsoft Power Automate |
| **4. Ứng dụng / Domain Layer**               | Sản phẩm doanh nghiệp, giải pháp vertical                    |

### 4.1. Foundation models

Foundation models, giống như cái tên của nó, là các mô hình nền tảng được huấn luyện trên một lượng dữ liệu khổng lồ và có khả năng thực hiện nhiều tác vụ khác nhau trong lĩnh vực AI.
Các mô hình này thường là các mô hình ngôn ngữ lớn (LLM) như ChatGPT, Claude, Gemini, LLaMA, v.v.
Chúng được coi là "bộ não" của các AI Agent, cung cấp khả năng hiểu ngôn ngữ tự nhiên, sinh văn bản, và thực hiện các tác vụ phức tạp khác.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/foundation_models.png" style="width: 1000px;"/>

### 4.2. Agent Framework và Reasoning

Hệ sinh thái AI agent đang phát triển rất sôi động, với nhiều framework và công cụ nguồn mở ra đời để giúp các nhà phát triển dễ dàng tạo ra ứng dụng AI tác tử.
Mỗi công cụ có mục tiêu và cách tiếp cận hơi khác nhau, nhưng đều phục vụ chung việc xây dựng các hệ thống AI linh hoạt dựa trên LLM.

Các framework này giúp các nhà phát triển dễ dàng hơn trong việc triển khai loop ReAct, quản lý memory, chọn công cụ ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/agent_framework.png" style="width: 1000px;"/>

### 4.3. Low‑/No‑code Automation & Integration

Các công cụ tự động hoá và tích hợp không cần mã (low-code/no-code) đang trở thành xu hướng trong việc xây dựng AI Agent.
Chúng cho phép người dùng không chuyên về lập trình có thể tạo ra các quy trình tự động hoá phức tạp bằng cách kéo thả các thành phần, kết nối các API và dịch vụ khác nhau mà không cần viết mã hoặc chỉ cần viết rất ít mã.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/4_deep_learning/images/4-ai-agent/automation_tools.png" style="width: 1000px;"/>
