---
time: 06/24/2022
title: Bài toán image captioning
description: Image captioning là bài toán xây dựng các mô hình có khả năng tự động tạo ra câu mô tả bằng ngôn ngữ tự nhiên cho một hình ảnh bất kỳ. Đây là bài toán giao thoa giữa Computer Vision và Natural Language Processing, đòi hỏi mô hình không chỉ "nhìn" và "hiểu" nội dung hình ảnh mà còn có khả năng diễn đạt thông tin đó thành ngôn ngữ tự nhiên mạch lạc và chính xác.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: false
---

## 1. Giới thiệu chung về image captioning

Bài toán image captioning là nhiệm vụ xây dựng các mô hình có khả năng tự động tạo ra câu mô tả bằng ngôn ngữ tự nhiên khi nhận đầu vào là một hình ảnh bất kỳ.
Đây là một bài toán phức tạp vì nó đòi hỏi mô hình phải đồng thời xử lý hai loại dữ liệu hoàn toàn khác nhau: thông tin thị giác (hình ảnh) và thông tin ngôn ngữ (văn bản), tạo nên sự giao thoa giữa Computer Vision và Natural Language Processing.

Image captioning có ứng dụng thực tiễn rộng khắp trong nhiều lĩnh vực:
- **Hỗ trợ người khiếm thị:** Các công nghệ screen reader tích hợp AI có thể tự động đọc mô tả nội dung ảnh cho người dùng khiếm thị, giúp họ tiếp cận thông tin trực quan trên internet.
- **Tìm kiếm ảnh bằng ngôn ngữ tự nhiên:** Thay vì tìm kiếm bằng từ khóa thủ công, hệ thống có thể tự động lập chỉ mục ảnh dựa trên nội dung, cho phép tìm kiếm bằng mô tả ngôn ngữ tự nhiên.
- **Tự động sinh alt text cho web:** Giúp cải thiện SEO và khả năng tiếp cận của website bằng cách tự động thêm mô tả văn bản cho mọi hình ảnh.
- **Hệ thống giám sát an ninh:** AI có thể tự động tạo báo cáo mô tả các sự kiện được ghi lại trong camera an ninh mà không cần người theo dõi liên tục.
- **Hỗ trợ y tế:** Tự động mô tả kết quả ảnh chụp X-quang, MRI, hỗ trợ bác sĩ trong quá trình chẩn đoán.

<!-- PLACEHOLDER IMAGE: applications.jpeg
Prompt: "Image captioning applications overview — 4-panel infographic: (1) blind person with screen reader describing a photo, (2) image search engine with text query returning images, (3) webpage with auto-generated alt text tags, (4) security camera with AI caption overlay. Clean infographic style, white background."
Sau khi sinh ảnh, thêm vào đây:
<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/applications.jpeg" style="width: 1000px;"/>
-->

Bài toán image captioning có thể được phân loại thành nhiều dạng bài toán con:

**Standard Image Captioning** là bài toán tạo ra một câu mô tả ngắn gọn, súc tích cho toàn bộ nội dung của hình ảnh.
Đây là dạng bài toán phổ biến nhất và là nền tảng cho các biến thể khác.

**Dense Captioning** là bài toán phát hiện và mô tả đồng thời nhiều vùng (region) khác nhau trong một hình ảnh.
Mô hình cần xác định tọa độ bounding box của từng vùng và tạo ra câu mô tả riêng cho mỗi vùng đó.

**Visual Question Answering (VQA)** là bài toán trả lời câu hỏi bằng ngôn ngữ tự nhiên dựa trên nội dung hình ảnh.
Đầu vào là cặp (hình ảnh, câu hỏi), đầu ra là câu trả lời tương ứng.

**Image-Text Matching** là bài toán đánh giá mức độ phù hợp giữa một hình ảnh và một đoạn văn bản mô tả, không sinh ra văn bản mới mà xếp hạng độ khớp giữa các cặp (ảnh, mô tả) cho trước.

## 2. Nhóm các phương pháp giải bài toán image captioning

### 2.1. Mô hình CNN + RNN Encoder-Decoder

Đây là kiến trúc đặt nền móng cho bài toán image captioning hiện đại, ra đời từ ý tưởng kết hợp sức mạnh của CNN trong xử lý ảnh với khả năng sinh chuỗi của RNN trong xử lý ngôn ngữ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/encoder_decoder.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

Hãy tưởng tượng một người mù được trao cho một bức ảnh thông qua người trợ lý:
- **Người trợ lý (CNN Encoder):** Quan sát toàn bộ bức ảnh và tóm tắt lại thành một đoạn ghi chú ngắn gọn — một vector số đặc trưng cho nội dung ảnh.
- **Người mù (RNN Decoder):** Đọc đoạn ghi chú đó và đặt câu bằng ngôn ngữ tự nhiên, từng từ một, dựa trên ghi chú và những từ đã nói trước đó.

Về mặt kỹ thuật, kiến trúc CNN+RNN bao gồm:
- **CNN Encoder:** Một mạng CNN (VGG, ResNet, GoogLeNet...) đã được pre-train trên ImageNet. Ảnh đầu vào $I$ được đưa qua CNN để trích xuất một **feature vector** cố định $v \in \mathbb{R}^d$ — thường là output của layer trước softmax.
- **RNN Decoder:** Một mạng LSTM (Long Short-Term Memory) nhận feature vector $v$ làm trạng thái khởi tạo, sau đó sinh ra câu mô tả từng từ một theo mô hình ngôn ngữ autoregressive:

$$p(y_1, y_2, \ldots, y_T | I) = \prod_{t=1}^{T} p(y_t | y_1, \ldots, y_{t-1}, v)$$

Tại mỗi timestep $t$, LSTM nhận đầu vào là embedding của từ $y_{t-1}$ đã sinh và hidden state $h_{t-1}$, cập nhật hidden state, rồi dự đoán phân phối xác suất trên toàn bộ vocabulary để chọn từ tiếp theo $y_t$.

**Hàm mất mát** là cross-entropy loss trên chuỗi từ dự đoán:
$$\mathcal{L} = -\sum_{t=1}^{T} \log p(y_t | y_1, \ldots, y_{t-1}, v)$$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/show_and_tell_arch.jpeg" style="width: 800px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **Đơn giản và end-to-end:** Kiến trúc gọn nhẹ, dễ hiểu và dễ huấn luyện từ đầu đến cuối.
- **Tận dụng pre-trained CNN:** Tận dụng được kiến thức thị giác phong phú từ CNN đã được huấn luyện trên ImageNet với hàng triệu ảnh.
- **Là nền tảng cho các phương pháp sau:** Đặt ra framework encoder-decoder mà hầu hết các phương pháp hiện đại đều kế thừa và cải tiến.

**Nhược điểm:**
- **Feature vector cố định:** CNN chỉ tạo ra một vector duy nhất đại diện cho toàn bộ ảnh, không thể "nhìn lại" những vùng cụ thể khi sinh từng từ. Ví dụ, khi sinh từ "chó", mô hình không thể tập trung vào vùng ảnh có con chó.
- **Thông tin chi tiết bị mất:** Quá trình pooling trong CNN làm mất thông tin không gian (spatial information), khiến mô hình khó mô tả chính xác vị trí và quan hệ không gian của các đối tượng.
- **Khó xử lý ảnh phức tạp:** Với ảnh có nhiều đối tượng, mô hình thường sinh ra câu mô tả chung chung, thiếu chi tiết.

#### Một số mô hình tiêu biểu trong nhóm

- **Show and Tell - NIC (Vinyals et al., 2015)** — [paper](https://arxiv.org/abs/1411.4555) — GoogLeNet + LSTM, là mô hình CNN+RNN đầu tiên đạt kết quả ấn tượng, baseline kinh điển của bài toán image captioning.
- **Show and Tell v2 (Google Brain, 2016)** — cải tiến với Inception-v3 encoder, đạt BLEU-4 = 27.7 và CIDEr = 86.5 trên MSCOCO.
- **MSR-VTT (Xu et al., 2016)** — mở rộng sang video captioning với LSTM encoder-decoder cho chuỗi frame.

### 2.2. Nhóm mô hình Attention-based

Cơ chế Attention (chú ý) ra đời để giải quyết điểm yếu cốt lõi của CNN+RNN: thay vì nén toàn bộ thông tin ảnh vào một vector cố định, mô hình học cách "nhìn vào đâu" trong ảnh khi sinh ra từng từ.

#### Mô tả ý tưởng và cơ chế hoạt động

Hãy tưởng tượng bạn đang mô tả một bức ảnh đông người:
- Khi bạn nói từ "người đàn ông", mắt bạn tập trung vào vùng ảnh có người đàn ông.
- Khi bạn nói "đang cầm", mắt bạn nhìn vào tay của người đó.
- Khi bạn nói "ô dù", mắt bạn chuyển sang vùng có cái ô dù.

Đó chính xác là điều cơ chế attention mô phỏng.

Về mặt kỹ thuật, thay vì dùng một feature vector duy nhất, CNN nay sinh ra một **feature map** 2D: $A \in \mathbb{R}^{L \times D}$ với $L$ là số vùng không gian (spatial locations) và $D$ là số chiều feature mỗi vùng.

Tại timestep $t$ khi sinh từ $y_t$, cơ chế attention tính **attention weight** $\alpha_{t,i}$ cho từng vùng $i$:

$$e_{t,i} = f_{att}(A_i, h_{t-1}), \quad \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{L} \exp(e_{t,j})}$$

Với $f_{att}$ là một MLP nhỏ và $h_{t-1}$ là hidden state của LSTM bước trước.

**Context vector** là tổng trọng số của các feature vectors:

$$\hat{z}_t = \sum_{i=1}^{L} \alpha_{t,i} A_i$$

LSTM nhận context vector $\hat{z}_t$ cùng với embedding từ $y_{t-1}$ để sinh từ tiếp theo $y_t$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/show_attend_tell_arch.jpeg" style="width: 800px;"/>

Có hai loại attention chính:
- **Soft attention (deterministic):** $\hat{z}_t$ là trung bình trọng số của tất cả các vùng — differentiable, có thể train bằng backpropagation thông thường.
- **Hard attention (stochastic):** Chọn đúng một vùng tại mỗi timestep theo phân phối $\alpha_t$ — yêu cầu REINFORCE để train.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/bottom_up_top_down_arch.jpeg" style="width: 800px;"/>

Mô hình **Bottom-Up and Top-Down Attention** cải tiến tiếp bằng cách dùng Faster R-CNN để trích xuất **region features** (Bottom-Up): thay vì dùng grid đồng đều của CNN, mô hình xác định các vùng có ý nghĩa ngữ nghĩa (objects, attributes) trong ảnh và chỉ attend vào những vùng đó.
Phần Top-Down là một LSTM thứ hai điều phối attention dựa trên ngữ cảnh ngôn ngữ hiện tại.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Khả năng diễn giải (interpretability):** Attention weights cho thấy rõ mô hình đang "nhìn vào đâu" khi sinh mỗi từ, giúp debug và giải thích kết quả.
- **Mô tả chi tiết hơn:** Bằng cách tập trung vào từng vùng cụ thể, mô hình có thể mô tả chính xác hơn các chi tiết nhỏ và quan hệ không gian giữa các đối tượng.
- **Kết quả SOTA nhiều năm:** Bottom-Up Top-Down giữ SOTA trên MSCOCO suốt nhiều năm trước khi Transformer ra đời.

**Nhược điểm:**
- **CNN vẫn là bottleneck:** Chất lượng caption vẫn phụ thuộc nhiều vào chất lượng feature map từ CNN.
- **LSTM khó học long-range dependency:** Với câu dài và phức tạp, LSTM vẫn gặp khó khăn trong việc duy trì ngữ cảnh xuyên suốt.
- **Slower inference:** Tính toán attention tại mỗi timestep tăng chi phí so với CNN+RNN đơn giản.

#### Một số mô hình tiêu biểu trong nhóm

- **Show, Attend and Tell (Xu et al., 2015)** — [paper](https://arxiv.org/abs/1502.03044) — soft và hard attention trên feature map CNN, bài báo đặt nền móng cho attention trong image captioning.
- **Bottom-Up and Top-Down Attention (Anderson et al., 2018)** — [paper](https://arxiv.org/abs/1707.07998) — Faster R-CNN region features + two-level LSTM, đạt SOTA trên MSCOCO nhiều năm với CIDEr = 120.1.
- **Adaptive Attention (Lu et al., 2017)** — [paper](https://arxiv.org/abs/1612.01887) — thêm "visual sentinel" để mô hình học khi nào nên attend vào ảnh và khi nào nên dựa vào ngữ cảnh ngôn ngữ thuần túy.
- **GCN-LSTM (Yao et al., 2018)** — [paper](https://arxiv.org/abs/1809.07041) — dùng Graph Convolutional Network để mô hình hóa quan hệ giữa các regions trước khi đưa vào LSTM.

### 2.3. Nhóm mô hình dựa trên Transformer

Sự ra đời của Transformer với cơ chế self-attention trong bài báo [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) đã mở ra một hướng hoàn toàn mới: thay thế CNN+LSTM bằng kiến trúc Transformer thuần túy, tận dụng khả năng học biểu diễn joint (chung) giữa hình ảnh và ngôn ngữ từ lượng dữ liệu khổng lồ thu thập trên internet.

Chi tiết hơn về kiến trúc Transformer đã được mình viết trong [bài viết này](/blog/mo-hinh-transformer).

#### Mô tả ý tưởng và cơ chế hoạt động

Quy trình của các mô hình Transformer-based captioning thường gồm ba bước:

1. **Visual Encoding:** Dùng ViT (Vision Transformer) hoặc CNN để biến ảnh thành chuỗi các **patch token** hoặc region token. Mỗi token đại diện cho một vùng ảnh và được biểu diễn bằng một vector.

2. **Pre-training với dữ liệu lớn:** Mô hình được pre-train trên hàng trăm triệu cặp (ảnh, văn bản) thu thập từ internet với nhiều objective:
    - **Image-Text Contrastive (ITC):** Học biểu diễn chung cho ảnh và văn bản (tương tự CLIP).
    - **Image-Text Matching (ITM):** Phân loại xem cặp (ảnh, văn bản) có khớp nhau không.
    - **Language Modeling (LM):** Sinh văn bản mô tả ảnh theo kiểu autoregressive.

3. **Fine-tuning:** Sau pre-training, mô hình được fine-tune trên dataset captioning cụ thể (MSCOCO, Flickr30k...).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/blip_arch.jpeg" style="width: 800px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **Scale tốt với dữ liệu:** Transformer tận dụng triệt để lượng dữ liệu lớn — càng nhiều dữ liệu pre-training, chất lượng caption càng cao.
- **Hiểu ngữ cảnh phong phú:** Self-attention cho phép mô hình nắm bắt quan hệ giữa các đối tượng và hoàn cảnh toàn cục trong ảnh.
- **Biểu diễn joint vision-language:** Học được không gian biểu diễn chung cho ảnh và văn bản, cho phép zero-shot transfer sang nhiều tác vụ khác.

**Nhược điểm:**
- **Yêu cầu lượng dữ liệu lớn:** Pre-training đòi hỏi hàng trăm triệu cặp ảnh-văn bản, chi phí thu thập và huấn luyện rất cao.
- **Phức tạp và nặng hơn CNN+LSTM:** Kiến trúc phức tạp hơn, khó customize và tune hyperparameter hơn các phương pháp cũ.
- **Noise trong web data:** Dữ liệu crawl từ web thường chứa nhiều cặp ảnh-văn bản không khớp, làm nhiễu quá trình pre-training.

#### Một số mô hình tiêu biểu trong nhóm

- **OSCAR (Li et al., 2020)** — [paper](https://arxiv.org/abs/2004.06871) — dùng object tags (nhãn đối tượng) làm anchor point để align visual tokens và text tokens, đạt SOTA trên nhiều benchmark VL.
- **VLP (Zhou et al., 2020)** — [paper](https://arxiv.org/abs/1909.11059) — unified pre-training cho cả understanding và generation.
- **BLIP (Li et al., 2022)** — [paper](https://arxiv.org/abs/2201.12086) — bootstrapped language-image pre-training với CapFilt module: lọc nhiễu từ web data bằng cách dùng chính mô hình để tạo caption mới và lọc caption xấu.
- **OFA (Wang et al., 2022)** — [paper](https://arxiv.org/abs/2202.03052) — sequence-to-sequence unified framework, một mô hình đơn xử lý hàng chục tác vụ multimodal và NLP khác nhau.
- **CoCa (Yu et al., 2022)** — [paper](https://arxiv.org/abs/2205.01917) — Contrastive Captioners, kết hợp contrastive loss và captioning loss trong một framework thống nhất.

### 2.4. Nhóm mô hình Large Vision-Language Models (VLMs)

Sự bùng nổ của Large Language Models (LLMs) như GPT-3/4, LLaMA đã tạo ra một xu hướng mới: kết nối trực tiếp một visual encoder với một LLM đã được pre-train với lượng tham số khổng lồ.
Thay vì huấn luyện toàn bộ mô hình từ đầu, cách tiếp cận này tận dụng năng lực sinh ngôn ngữ vượt trội của LLM.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/vlm.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

Ý tưởng cốt lõi là xây dựng một "cầu nối" (bridge) giữa không gian thị giác và không gian ngôn ngữ:

1. **Visual Encoder (thường bị đóng băng):** Một mô hình ViT mạnh như CLIP ViT-L/14 trích xuất visual tokens từ ảnh đầu vào. Các tham số của phần này thường được **frozen** (không cập nhật) trong quá trình training để bảo toàn kiến thức thị giác đã học.

2. **Projection / Bridge Layer:** Một module nhẹ học cách ánh xạ visual tokens sang không gian embedding của LLM:
    - **Linear projection (LLaVA):** Đơn giản nhất — một lớp linear chiếu visual tokens sang LLM embedding space.
    - **Q-Former (BLIP-2):** Phức tạp hơn — một transformer nhỏ với một tập **learnable query tokens** trích xuất thông tin liên quan từ visual tokens thông qua cross-attention.

3. **LLM Decoder (thường bị đóng băng một phần):** Mô hình ngôn ngữ lớn (LLaMA, Vicuna, Flan-T5...) nhận visual tokens đã được chiếu cùng với text prompt và sinh ra caption.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/blip2_arch.jpeg" style="width: 800px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/llava_arch.jpeg" style="width: 800px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **Khả năng sinh ngôn ngữ vượt trội:** Tận dụng năng lực ngôn ngữ đã được tôi luyện trên hàng nghìn tỷ token của LLM, tạo ra caption tự nhiên, mạch lạc và chi tiết.
- **Instruction following và zero-shot:** Mô hình có thể theo dõi hướng dẫn phức tạp ("mô tả chi tiết màu sắc", "viết caption theo phong cách thơ") mà không cần fine-tune.
- **Reasoning về ảnh:** VLMs có khả năng suy luận, giải thích, và trả lời câu hỏi về ảnh, vượt xa khả năng mô tả đơn thuần.

**Nhược điểm:**
- **Hallucination:** Mô hình có xu hướng "bịa" ra các đối tượng hoặc chi tiết không có trong ảnh — đây là hệ quả kế thừa từ LLM và là thách thức lớn nhất hiện nay.
- **Chi phí inference cao:** Với hàng tỷ tham số, inference trên thiết bị edge hoặc real-time khó khăn.
- **Alignment không hoàn hảo:** Khoảng cách giữa visual representation và text representation vẫn là thách thức kỹ thuật, đặc biệt với các ảnh có nhiều chi tiết tinh tế.

#### Một số mô hình tiêu biểu trong nhóm

- **Flamingo (Alayrac et al., 2022)** — [paper](https://arxiv.org/abs/2204.14198) — perceiver resampler + cross-attention vào frozen Chinchilla LLM, few-shot learning mạnh với in-context examples.
- **BLIP-2 (Li et al., 2023)** — [paper](https://arxiv.org/abs/2301.12597) — Q-Former bridge giữa frozen ViT và frozen LLM, hiệu quả vì chỉ train Q-Former nhỏ.
- **LLaVA (Liu et al., 2023)** — [paper](https://arxiv.org/abs/2304.08485) — CLIP ViT-L/14 + linear projection + LLaMA/Vicuna, visual instruction tuning với dữ liệu GPT-4 tổng hợp.
- **InstructBLIP (Dai et al., 2023)** — [paper](https://arxiv.org/abs/2305.06500) — instruction-tuned BLIP-2 với diverse instruction templates.
- **LLaVA-1.5 (Liu et al., 2023)** — [paper](https://arxiv.org/abs/2310.03744) — thay linear projection bằng MLP 2-layer, thêm academic VQA data, đạt SOTA trên nhiều benchmark với chi phí thấp.
- **GPT-4V (OpenAI, 2023)** — multimodal GPT-4, khả năng reasoning và mô tả ảnh mạnh nhất hiện nay trên các tác vụ open-ended.
- **Gemini (Google DeepMind, 2023)** — kiến trúc natively multimodal từ đầu, xử lý text, ảnh, audio, video trong cùng một mô hình.

## 3. Các metrics trong image captioning

Đánh giá chất lượng của một mô hình image captioning là bài toán không hề đơn giản.
Một câu caption tốt cần vừa chính xác về nội dung (đúng đối tượng, thuộc tính, quan hệ), vừa trôi chảy về ngôn ngữ (đúng ngữ pháp, tự nhiên), và vừa phù hợp với cách con người thường diễn đạt.
Không có một metric nào đơn lẻ có thể đo lường tất cả các khía cạnh này.

<!-- PLACEHOLDER IMAGE: metrics_overview.jpeg
Prompt: "Image captioning evaluation metrics taxonomy — a clear diagram organizing metrics into: (1) N-gram based: BLEU, METEOR, ROUGE-L, CIDEr; (2) Semantic/graph based: SPICE, BERTScore; (3) Reference-free: CLIP Score. Show the key difference between reference-based vs reference-free. Clean academic infographic style, white background."
Sau khi sinh ảnh, thêm vào đây:
<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/metrics_overview.jpeg" style="width: 800px;"/>
-->

### 3.1. BLEU (Bilingual Evaluation Understudy)

> 📌 **Metric kế thừa từ bài toán dịch máy.** Xem chi tiết công thức, ví dụ và ưu nhược điểm tại [bài giảng Sequence-to-Sequence](/blog/sequence-to-sequence).

BLEU đo lường độ chính xác (precision) của các n-gram trong câu dự đoán so với các câu tham chiếu của con người.
Đây là metric phổ biến nhất và là chuẩn so sánh cơ bản trong tất cả các paper image captioning.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/7-seq-to-seq/bleu.jpeg" style="width: 800px;"/>

### 3.2. METEOR (Metric for Evaluation of Translation with Explicit ORdering)

> 📌 **Metric kế thừa từ bài toán dịch máy.** Xem chi tiết công thức, ví dụ và ưu nhược điểm tại [bài giảng Sequence-to-Sequence](/blog/sequence-to-sequence).

METEOR cải tiến BLEU với so khớp linh hoạt hơn: exact match, stemming ("running" ↔ "runs") và WordNet synonyms ("dog" ↔ "canine"), kết hợp với chunk penalty cho các từ match không đúng thứ tự.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/7-seq-to-seq/meteor.jpeg" style="width: 800px;"/>

### 3.3. ROUGE-L

> 📌 **Metric kế thừa từ bài toán tóm tắt văn bản.** Xem chi tiết công thức, ví dụ và ưu nhược điểm tại [bài giảng Sequence-to-Sequence](/blog/sequence-to-sequence).

ROUGE-L dựa trên Longest Common Subsequence (LCS) — chuỗi con chung dài nhất giữa câu dự đoán và tham chiếu — cho phép các từ khớp không liền kề nhưng phải theo đúng thứ tự.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/7-seq-to-seq/rouge.jpeg" style="width: 800px;"/>

### 3.4. CIDEr (Consensus-based Image Description Evaluation)

CIDEr được giới thiệu trong bài báo [CIDEr: Consensus-based Image Description Evaluation](https://arxiv.org/abs/1411.5726) (Vedantam et al., 2015), được thiết kế đặc biệt cho bài toán image captioning và hiện là metric quan trọng nhất trong lĩnh vực này.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/cider.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

Ý tưởng chính: Một câu chú thích tốt là câu nắm bắt được những gì **hầu hết mọi người đồng thuận** (consensus) khi mô tả bức ảnh đó, đồng thời phải chứa các chi tiết **đặc trưng và độc đáo** của bức ảnh đó (không phải những từ chung chung xuất hiện trong mọi caption).

CIDEr đạt được điều này thông qua **trọng số TF-IDF** trên n-gram:
- **TF (Term Frequency):** N-gram xuất hiện nhiều trong các câu tham chiếu của ảnh đó → quan trọng với ảnh đó.
- **IDF (Inverse Document Frequency):** N-gram hiếm khi xuất hiện trong toàn bộ dataset → đặc trưng và độc đáo.

**Công thức đầy đủ:**

**Bước 1 — TF-IDF weight** cho n-gram $\omega$ trong câu $s_{ij}$ (câu tham chiếu thứ $j$ của ảnh $I_i$):

$$w_{ij}(\omega) = \underbrace{\frac{h_k(s_{ij})}{\sum_{\omega' \in \Omega} h_k(s_{ij})}}_{\text{TF}} \cdot \underbrace{\log \frac{|I|}{\sum_{I_p \in I} \min\!\left(1,\, \sum_{q} h_k(s_{pq})\right)}}_{\text{IDF}}$$

Với $h_k(s_{ij})$ là số lần n-gram $\omega$ xuất hiện trong câu $s_{ij}$, $|I|$ là tổng số ảnh trong dataset.

**Bước 2 — CIDEr-n** (cho order $n$ cụ thể):

$$CIDEr_n(c_i, S_i) = \frac{1}{m} \sum_{j=1}^{m} \frac{\mathbf{g}^n(c_i) \cdot \mathbf{g}^n(s_{ij})}{\|\mathbf{g}^n(c_i)\| \cdot \|\mathbf{g}^n(s_{ij})\|}$$

Với $\mathbf{g}^n(c_i)$ là vector TF-IDF của tất cả n-gram order $n$ trong câu dự đoán $c_i$, và $m$ là số câu tham chiếu.

**CIDEr cuối cùng** (trung bình qua $n$ từ 1 đến 4):

$$CIDEr(c_i, S_i) = \frac{1}{4} \sum_{n=1}^{4} CIDEr_n(c_i, S_i)$$

#### Ví dụ

Xét hai câu dự đoán cho một ảnh chụp "chú chó nâu đang chạy trên bãi cỏ":
- **Câu A:** "a brown dog is running on the grass" → CIDEr cao vì từ "brown", "running", "grass" đặc trưng cho ảnh này và trùng với nhiều câu tham chiếu.
- **Câu B:** "there is an animal in a field" → CIDEr thấp vì "animal", "field" là những từ chung chung, IDF thấp.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Tương quan cao nhất với đánh giá con người** trong số các metric n-gram cho captioning.
- **Thưởng cho chi tiết đặc trưng:** Khuyến khích mô hình sinh ra những mô tả độc đáo và cụ thể, không chỉ là những câu "an toàn" và chung chung.
- **Xét toàn bộ consensus:** So sánh với nhiều câu tham chiếu từ nhiều người khác nhau.

**Nhược điểm:**
- **Không ổn định trên tập dữ liệu nhỏ:** IDF cần đủ lượng dữ liệu để ước lượng tần suất n-gram chính xác.
- **Vẫn dựa trên n-gram:** Không xử lý được từ đồng nghĩa và vẫn yêu cầu có câu tham chiếu của con người.

### 3.5. SPICE (Semantic Propositional Image Caption Evaluation)

SPICE được giới thiệu trong bài báo [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/abs/1607.08822) (Anderson et al., 2016), tập trung hoàn toàn vào ý nghĩa ngữ nghĩa (semantics) thay vì sự trùng khớp từ vựng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/spice.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

Ý tưởng chính: Đánh giá xem câu chú thích có mô tả đúng các **đối tượng, thuộc tính và mối quan hệ** giữa chúng trong ảnh hay không.

SPICE phân tích câu thành một **đồ thị cảnh** (scene graph) $G = (O, E, K)$:
- $O$: tập hợp **objects** ("người đàn ông", "con chó")
- $E$: tập hợp **relations** ("đang dắt", "đứng gần")
- $K$: tập hợp **attributes** ("màu nâu", "cao")

**Công thức đầy đủ:**

**Bước 1 — Tập hợp tuples** từ scene graph:

$$T(G) = \underbrace{\{(o) : o \in O\}}_{\text{object tuples}} \cup \underbrace{\{(o_1, r, o_2) : (o_1, r, o_2) \in E\}}_{\text{relation triples}} \cup \underbrace{\{(o, a) : (o, a) \in K\}}_{\text{attribute tuples}}$$

**Bước 2 — Precision và Recall** trên tuple matching:

$$P(c, S) = \frac{|T(G(c)) \otimes T(G(S))|}{|T(G(c))|}, \quad R(c, S) = \frac{|T(G(c)) \otimes T(G(S))|}{|T(G(S))|}$$

Với $\otimes$ là soft matching — hai tuples khớp nếu chúng semantically equivalent (không yêu cầu exact string match).

**SPICE cuối cùng:**

$$SPICE(c, S) = F_1(P, R) = \frac{2 \cdot P(c,S) \cdot R(c,S)}{P(c,S) + R(c,S)}$$

#### Ví dụ

Xét ảnh "một người đàn ông cao mặc áo đỏ đang dắt con chó nâu trên vỉa hè":
- **Câu tốt:** "a tall man in a red shirt walks a brown dog" → SPICE cao vì đúng objects (man, dog), attributes (tall, red, brown), relations (walks).
- **Câu thiếu ngữ nghĩa:** "there is a person and an animal" → SPICE thấp vì thiếu attributes và relations.
- **Câu đúng nhưng sai ngữ pháp:** "man tall shirt red dog brown walk" → SPICE vẫn cao (đúng tuples) nhưng BLEU thấp (sai ngữ pháp).

#### Ưu và nhược điểm

**Ưu điểm:**
- **Metric tốt nhất để đánh giá ngữ nghĩa** — thực sự "hiểu" nội dung câu mô tả, không chỉ đếm n-gram.
- **Tương quan tốt với đánh giá của con người** về mặt nội dung.

**Nhược điểm:**
- **Phức tạp và chậm:** Yêu cầu scene graph parser mạnh, tốn thời gian tính toán hơn các metrics khác.
- **Không đánh giá ngữ pháp và tính trôi chảy:** Câu sai ngữ pháp hoàn toàn vẫn có thể đạt điểm SPICE cao nếu chứa đúng các tuples ngữ nghĩa.

### 3.6. Một số chỉ số đánh giá phụ khác

#### 3.6.1. BERTScore

> 📌 **Metric kế thừa từ bài toán NLP.** Xem chi tiết công thức, ví dụ và ưu nhược điểm tại [bài giảng Sequence-to-Sequence](/blog/sequence-to-sequence).

BERTScore dùng contextual embeddings của BERT để so sánh từng token dự đoán với token tham chiếu gần nhất về mặt ngữ nghĩa (greedy matching theo cosine similarity), nhờ đó xử lý tốt đồng nghĩa và paraphrase mà BLEU/ROUGE bỏ qua.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/7-seq-to-seq/bert_score.jpeg" style="width: 800px;"/>

#### 3.6.2. Human Evaluation

Đánh giá bởi con người là tiêu chuẩn vàng cuối cùng cho image captioning.
Mọi metric tự động đều được phát triển và kiểm chứng bằng cách so sánh xem chúng có tương quan tốt với nhận xét của con người hay không.

Có hai dạng đánh giá phổ biến:
- **Đánh giá so sánh (Pairwise / 2AFC):** Người đánh giá xem hai caption A/B và chọn cái tốt hơn theo tiêu chí cụ thể (độ chính xác, tính tự nhiên, độ chi tiết).
- **Đánh giá theo thang đo (Likert):** Chấm điểm từng caption theo thang 1–5 hoặc 1–7 cho các tiêu chí riêng biệt.

Để đảm bảo độ tin cậy, thường thuê 3–5 annotators cho mỗi ảnh, đo **inter-rater agreement** bằng Cohen's kappa hoặc Krippendorff's alpha, và loại bỏ annotators không đạt ngưỡng agreement.

**Bảng benchmark tổng hợp (MSCOCO Karpathy test split):**

| Mô hình | BLEU-4 | METEOR | CIDEr | SPICE |
|---------|--------|--------|-------|-------|
| Show and Tell | 27.7 | 23.7 | 86.5 | — |
| Bottom-Up Top-Down | 36.2 | 27.0 | 113.5 | 20.3 |
| BLIP | 39.7 | 29.3 | 133.3 | 23.3 |
| BLIP-2 | 43.7 | 31.1 | 145.8 | 25.0 |

## 4. Các thách thức của bài toán image captioning

Dù đã đạt được những bước tiến vượt bậc, bài toán image captioning vẫn còn nhiều thách thức chưa được giải quyết triệt để.

<!-- PLACEHOLDER IMAGE: challenges.jpeg
Prompt: "Image captioning failure modes illustrated — 4 panels: (1) hallucination: empty dinner table photo but model caption says 'a person eating food at a table', (2) generic captions: 5 different outdoor scenes all getting identical caption 'a person standing outside', (3) wrong count: photo with 3 cats but caption says '2 cats playing', (4) cultural misidentification: Vietnamese traditional festival scene misidentified as generic party. Red X marks on wrong predictions. Clean infographic style."
Sau khi sinh ảnh, thêm vào đây:
<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/14-image-captioning/challenges.jpeg" style="width: 800px;"/>
-->

- **Hallucination (Ảo giác):** Đây là thách thức nghiêm trọng nhất, đặc biệt với các mô hình VLM.
Mô hình có xu hướng "bịa" ra các đối tượng hoặc chi tiết không có trong ảnh — ví dụ mô tả một bàn ăn trống rỗng là "a family having dinner together".
Nguyên nhân chính là LLM học được các bias thống kê mạnh từ dữ liệu text (bàn ăn thường đi với người ăn), và áp đặt chúng lên ảnh.

- **Câu mô tả chung chung (Generic Captions):** Mô hình thường ưu tiên sinh ra những câu "an toàn" và phổ biến trong dữ liệu huấn luyện ("a person standing outside", "a dog in the park") thay vì mô tả cụ thể, chi tiết những đặc điểm độc đáo của bức ảnh.

- **Đa ngôn ngữ và văn hóa:** Hầu hết các dataset lớn (MSCOCO, Flickr30k) và các metrics tự động đều tập trung vào tiếng Anh và văn hóa phương Tây.
Mô hình thường gặp khó khăn khi mô tả các yếu tố văn hóa đặc thù của các nền văn hóa khác.

- **Fine-grained Details (Chi tiết tinh tế):** Mô hình thường bỏ qua hoặc mô tả sai các chi tiết quan trọng như số lượng chính xác ("3 cats" vs "some cats"), màu sắc cụ thể, vị trí tương đối ("on the left" vs "behind"), và các biểu cảm/cảm xúc tinh tế.

- **Mối quan hệ không gian phức tạp:** Hiểu và diễn đạt chính xác các quan hệ không gian ("người đứng phía sau cây") và quan hệ nhân quả ("em bé khóc vì bị ngã") vẫn là thách thức lớn.

- **Đánh giá chất lượng tự động:** Khoảng cách giữa các metric tự động (BLEU, CIDEr, SPICE) và đánh giá thực sự của con người vẫn còn lớn.
Một caption có thể đạt điểm CIDEr cao nhưng lại được con người đánh giá là sai hoặc thiếu tự nhiên, và ngược lại.
Cần có các metric tốt hơn để thoát khỏi sự phụ thuộc vào human evaluation tốn kém.
