---
time: 06/08/2022
title: Bài toán image generation
description: Các mô hình tạo sinh hình ảnh là một lĩnh vực đột phá, tập trung vào việc huấn luyện máy tính để tự tạo ra những hình ảnh mới, độc đáo và chân thực từ các dữ liệu đầu vào. Các mô hình này không chỉ đơn thuần "sao chép" hay "ghép" các phần của những bức ảnh có sẵn. Thay vào đó, chúng học các khái niệm, thuộc tính, phong cách và mối quan hệ giữa các đối tượng từ một tập dữ liệu khổng lồ. Từ đó, chúng có khả năng "tưởng tượng" và tổng hợp nên một hình ảnh hoàn toàn mới dựa trên yêu cầu của người dùng.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về image generation

> 🖼️ **[Ảnh placeholder #1 — Hero image generation]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. A cheerful cat artist standing in front of a large canvas, holding a magic paintbrush. Three glowing input streams flow into the canvas from the left side: a text bubble saying 'a dog on the beach', a tiny vector arrow, and a small photo thumbnail. On the canvas appears a freshly generated image of a kawaii puppy on a sunset beach with sparkles around it. Soft pastel pink, purple, and yellow colors, minimal clean background, friendly creative atmosphere."*

Bài toán image generation là nhiệm vụ xây dựng các mô hình có khả năng tạo ra hình ảnh mới sao cho chúng trông chân thực hoặc phù hợp với một mô tả đầu vào.
Đầu vào có thể là một vector bất kỳ, một hình ảnh khác, một đoạn văn bản ...
Đây là một hướng nghiên cứu quan trọng của trí tuệ nhân tạo, nơi máy học cách nắm bắt đặc trưng của dữ liệu ảnh để sinh ra ảnh chưa từng tồn tại trước đó.

Một thách thức lớn là ảnh sinh ra cần vừa đẹp về thị giác, vừa hợp ngữ nghĩa, đồng thời tránh lỗi như méo hình, thiếu chi tiết hoặc không đúng với yêu cầu đầu vào.
Tóm lại, sinh ảnh là bài toán giúp máy tính không chỉ "nhìn" và "hiểu" ảnh, mà còn có thể sáng tạo ra ảnh mới, mở ra rất nhiều ứng dụng thực tiễn trong đời sống và công nghệ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/applications.jpeg" style="width: 1000px;"/>

Image generation có ứng dụng rộng khắp, từ giải trí đến sáng tạo nghệ thuật số, hỗ trợ y tế, quảng cáo, giáo dục, đến giám sát an ninh.
Ví dụ:
- **Sáng tạo nhân vật game, điện ảnh:** Hệ thống AI tự động hóa design, tạo phong cách đa dạng, phục vụ sản xuất nhanh chóng, hiệu quả.
- **Phục chế và nâng cấp ảnh cũ:** AI hồi phục ảnh cổ, tăng độ phân giải, đem lại giá trị lịch sử, nghệ thuật.
- **Media, quảng cáo:** Tạo ảnh/clip cá nhân hóa, truyền thông sáng tạo.
- **Giáo dục:** Hỗ trợ trực quan hóa, sinh dữ liệu giả phục vụ nghiên cứu, luyện tập.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/image_synthesis_translation.jpeg" style="width: 1000px;"/>

**Image synthesis** là bài toán tạo ra ảnh mới từ nhiễu ngẫu nhiên.
Mục tiêu của bài toán này là sinh ra những hình ảnh chân thực, đa dạng và phù hợp với nội dung mong muốn, chẳng hạn như tạo khuôn mặt người không có thật.

**Image-to-image translation** là bài toán biến đổi một ảnh đầu vào thành một ảnh đầu ra nhưng vẫn giữ lại một phần cấu trúc hoặc nội dung gốc.
Ví dụ:
- Chuyển ảnh phác thảo thành ảnh thật
- Chuyển ảnh đen trắng thành ảnh màu
- Chuyển ảnh chụp ban ngày thành ảnh ban đêm

Khác với image synthesis, bài toán này không tạo ảnh hoàn toàn từ đầu mà tập trung vào ánh xạ từ miền ảnh này sang miền ảnh khác.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/text_to_image.jpeg" style="width: 500px;"/>

**Text-to-image** nhằm tạo ra hình ảnh từ mô tả bằng văn bản.
Cụ thể, mô hình nhận đầu vào là một câu hoặc đoạn mô tả như "một chú chó đang chạy trên bãi biển lúc hoàng hôn", sau đó sinh ra một bức ảnh phù hợp với nội dung ngữ nghĩa, bối cảnh và chi tiết thị giác được nhắc đến trong văn bản.
Mục tiêu của bài toán này là tạo ảnh vừa chân thực, vừa đúng với mô tả, đồng thời vẫn đảm bảo tính đa dạng và thẩm mỹ trong kết quả sinh ra.

> 🖼️ **[Ảnh placeholder #2 — Text-to-image flow]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. On the left, a cat wearing reading glasses holds a paper scroll with the Vietnamese prompt 'một chú chó chạy trên bãi biển lúc hoàng hôn'. An arrow labeled 'Text-to-Image AI' points to the right, where a glowing canvas frame appears showing a kawaii puppy running on a sunset beach with pink and orange clouds, palm tree silhouettes, and soft waves. Tiny sparkles around the canvas. Soft pastel colors, minimal clean background."*

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/inpainting_outpainting.jpeg" style="width: 500px;"/>

**Image inpainting** là bài toán khôi phục hoặc điền vào những vùng bị thiếu, bị che khuất, hoặc bị hư hỏng trong ảnh sao cho phần được sinh ra tự nhiên và phù hợp với ngữ cảnh xung quanh.
Ví dụ, mô hình có thể xóa một vật thể không mong muốn trong ảnh rồi tự động điền lại nền phía sau một cách hợp lý.

**Image outpainting** là bài toán mở rộng ảnh ra ngoài phạm vi ban đầu bằng cách sinh thêm nội dung mới ở các vùng biên, nhưng vẫn phải giữ được sự nhất quán về bố cục, màu sắc và ngữ nghĩa với ảnh gốc.
Ví dụ, từ một bức ảnh phong cảnh hẹp, mô hình có thể mở rộng thêm bầu trời, núi hoặc mặt đất để tạo thành một khung hình lớn hơn.

## 2. Nhóm các phương pháp giải bài toán image generation

### 2.1. Mô hình Variational Autoencoders

**VAE - Variational Autoencoder** được giới thiệu vào năm 2013 trong bài báo [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114) (Kingma & Welling) là một loại mô hình sinh (generative model) với mục tiêu không chỉ là nén và tái tạo lại dữ liệu (như autoencoder tiêu chuẩn) mà học một **không gian ẩn (latent space)** có cấu trúc và liên tục.
Từ latent space, mô hình có thể "sinh" ra những dữ liệu mới chưa từng tồn tại nhưng vẫn hợp lý và giống với dữ liệu gốc.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/10-vae/banner.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

Ví dụ, hãy tưởng tượng bạn là một họa sĩ:
- **Học (Training):** Bạn xem hàng ngàn bức ảnh về khuôn mặt người.
Thay vì nhớ thuộc lòng từng chi tiết của mỗi bức ảnh, bạn học được những "đặc trưng cốt lõi" của một khuôn mặt: mắt, mũi, miệng trông như thế nào, tỷ lệ ra sao...
Toàn bộ kiến thức này được bạn nén vào trong não bộ.
- **Sáng tạo (Generation):** Khi có người yêu cầu bạn vẽ một khuôn mặt hoàn toàn mới, bạn không chép lại một bức ảnh nào đã xem.
Thay vào đó, bạn kết hợp các đặc trưng cốt lõi trong đầu mình để vẽ ra một người không có thật.

Trong ví dụ này:
- Bộ não của bạn chính là **Không gian ẩn (Latent Space)**.
- Quá trình bạn học từ ảnh thật chính là **Encoder (Bộ mã hóa)**.
- Quá trình bạn vẽ ra ảnh mới chính là **Decoder (Bộ giải mã)**.

> 🖼️ **[Ảnh placeholder #3 — VAE Encoder–Decoder pipeline]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. A pipeline from left to right: an input photo of a cute bear face → enters a pink funnel-shaped 'Encoder' that compresses it into a small floating sketchbook labeled 'Latent Space' showing two tiny Gaussian curves (μ, σ); → from the sketchbook, a sampled blue dot exits → enters a mirror-image funnel 'Decoder' on the right → outputs a freshly drawn bear face that looks similar but slightly different. A bear-artist mascot oversees the whole pipeline. Soft pastel pink and lavender colors, minimal clean background."*

Về mặt kỹ thuật, kiến trúc VAE bao gồm:
- **Encoder $q_\phi(z|x)$:** nhận ảnh đầu vào $x$ và ánh xạ vào hai vector tham số của phân phối Gaussian — vector trung bình $\mu$ và vector độ lệch chuẩn $\sigma$.
- **Sampling với Reparameterization Trick:** lấy mẫu $z = \mu + \sigma \odot \epsilon$ với $\epsilon \sim \mathcal{N}(0, I)$ để cho phép gradient lan ngược qua bước sampling.
- **Decoder $p_\theta(x|z)$:** nhận vector latent $z$ và sinh ra ảnh tái tạo $\hat{x}$.

Hàm mất mát của VAE là **Evidence Lower Bound (ELBO)** gồm hai thành phần:

$$\mathcal{L}_{VAE} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Loss}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL Divergence}}$$

- **Reconstruction Loss:** đảm bảo ảnh tái tạo $\hat{x}$ giống ảnh gốc $x$.
- **KL Divergence:** ràng buộc phân phối latent $q_\phi(z|x)$ gần với prior $p(z) = \mathcal{N}(0, I)$, tạo nên một không gian latent mượt mà, liên tục.

> 🖼️ **[Ảnh placeholder #4 — Latent space interpolation]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. A horizontal row of 7 cute cat faces showing a smooth morphing animation from left (happy smiling cat) → middle (neutral cat) → right (sad teary cat). Below the row, a soft pastel rainbow gradient line labeled 'Latent Space' with 7 tiny dots aligned under each cat face. Tiny floating sparkles between faces showing 'continuous interpolation'. Soft pastel colors, minimal clean background."*

#### Ưu và nhược điểm

**Ưu điểm:**
- **Quá trình huấn luyện ổn định:** VAEs có một hàm mục tiêu rõ ràng (Evidence Lower Bound - ELBO) để tối ưu hóa nên ổn định hơn nhiều so với GANs và chắc chắn sẽ hội tụ.
- **Latent Space có ý nghĩa:** Do được ràng buộc phải tuân theo một phân phối xác suất (VD: phân phối chuẩn), latent space của VAEs rất mượt mà và liên tục.
Điều này giúp các tác vụ như nội suy (VD: biến đổi từ từ một khuôn mặt này sang một khuôn mặt khác).
- **Có thể ước lượng xác suất:** VAEs cung cấp một cách để ước lượng xác suất của một điểm dữ liệu, điều mà GANs không làm được.

**Nhược điểm:**
- **Chất lượng ảnh sinh ra bị mờ:** Đây là nhược điểm cố hữu của VAEs.
Do hàm mất mát có thành phần tái tạo (reconstruction loss) thường dùng sai số bình phương trung bình (MSE), mô hình có xu hướng tạo ra các ảnh "trung bình", an toàn, dẫn đến kết quả bị mờ và thiếu các chi tiết sắc nét so với GANs.
- **Vấn đề "Prior Hole":** Đôi khi, Decoder có thể học cách "lờ đi" thông tin từ không gian ẩn và chỉ tạo ra một kết quả trung bình cho mọi đầu vào, làm cho mô hình trở nên vô dụng.

> 🖼️ **[Ảnh placeholder #5 — VAE blurry vs GAN sharp]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style, split into two side-by-side panels with a soft divider. Left panel labeled 'VAE': a bear-artist holding a paintbrush, showing a blurry, fuzzy painting of a cat face — the cat looks dreamy and averaged-out. Right panel labeled 'GAN': a fox-forger artist showing a crisp, sharp painting of a cat face with clear whiskers, bright eyes, and fine details. A small comparison emoji '🌫️ vs ✨' between them. Soft pastel colors, minimal clean background."*

#### Một số mô hình tiêu biểu trong nhóm

- **β-VAE (Higgins et al., 2017)** — [paper](https://openreview.net/forum?id=Sy2fzU9gl) — thêm hệ số $\beta$ vào phần KL Divergence để học latent space disentangled (mỗi chiều ứng với một thuộc tính ngữ nghĩa).
- **Conditional VAE - CVAE (Sohn et al., 2015)** — [paper](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html) — sinh ảnh có điều kiện theo nhãn lớp hoặc thuộc tính cho trước.
- **VQ-VAE (van den Oord et al., 2017)** — [paper](https://arxiv.org/abs/1711.00937) — sử dụng latent space **rời rạc** (codebook) thay vì liên tục, mở đường cho kết hợp với Transformer.
- **VQ-VAE-2 (Razavi et al., 2019)** — [paper](https://arxiv.org/abs/1906.00446) — kiến trúc hierarchical (nhiều tầng codebook) cho chất lượng cao trên ImageNet.
- **NVAE (Vahdat & Kautz, 2020)** — [paper](https://arxiv.org/abs/2007.03898) — kiến trúc deep hierarchical với residual cells, đạt SOTA về log-likelihood trên CelebA-HQ và FFHQ.

### 2.2. Nhóm mô hình Generative Adversarial Networks - GANs

**GANs - Generative Adversarial Networks** được giới thiệu bởi Ian Goodfellow và các đồng nghiệp vào năm 2014 trong bài báo [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) với ý tưởng cốt lõi là xây dựng và huấn luyện một mô hình như một trò chơi đối kháng (adversarial game) giữa hai mạng nơ-ron.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/9-gans/banner.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

GAN là trò chơi giữa hai người chơi:
- **Generator - Kẻ làm giả:** Một họa sĩ chuyên làm giả các tác phẩm nghệ thuật. Mục tiêu của anh ta là tạo ra những bức tranh giả tinh vi đến mức không thể phân biệt được với tranh thật.
- **Discriminator - Chuyên gia nghệ thuật:** Một chuyên gia có con mắt tinh tường. Nhiệm vụ của ông là xem một bức tranh và xác định xem đó là tranh thật (từ một bộ sưu tập gốc) hay tranh giả (do kẻ làm giả tạo ra).

Trò chơi này diễn ra như sau:
- Ban đầu, kẻ làm giả còn non tay, các bức tranh giả rất dễ bị phát hiện. Chuyên gia dễ dàng chỉ ra đâu là giả.
- Kẻ làm giả nhận được phản hồi (biết mình đã bị phát hiện) và học hỏi từ những sai lầm để vẽ ra những bức tranh giả ngày càng tốt hơn.
Khi kẻ làm giả giỏi lên, chuyên gia cũng phải rèn luyện con mắt của mình để trở nên tinh tường hơn, tìm ra những chi tiết nhỏ nhất để phân biệt thật-giả.
Quá trình "đối đầu" này tiếp tục, cả hai cùng tiến bộ.
- Cuối cùng, khi trò chơi đạt đến trạng thái cân bằng, kẻ làm giả sẽ tạo ra những tác phẩm giả hoàn hảo đến mức chuyên gia chỉ có thể đoán bừa (với xác suất 50/50) xem nó là thật hay giả.
Khi đó, chúng ta đã có một Generator cực kỳ tài năng, có khả năng tạo ra những hình ảnh siêu thực.

> 🖼️ **[Ảnh placeholder #6 — Generator vs Discriminator adversarial game]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. On the left, a sneaky fox artist labeled 'Generator (Kẻ làm giả)' painting fake cat portraits at an easel with sparkly noise vectors flowing into their brush. On the right, a wise owl detective labeled 'Discriminator (Chuyên gia)' wearing a monocle, examining paintings on a table — half real cat photos, half fox-painted ones — holding a sign showing 'Real ✓ / Fake ✗'. A glowing feedback arrow loops between them showing they learn from each other. Soft pastel orange and teal colors, minimal clean background, friendly competitive vibe."*

Về mặt toán học, GAN tối ưu hoá hàm minimax:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

- $D(x)$: xác suất Discriminator dự đoán $x$ là ảnh thật.
- $G(z)$: ảnh được Generator sinh ra từ noise vector $z$.
- **Discriminator** muốn maximize $V$ — đoán đúng thật/giả nhiều nhất có thể.
- **Generator** muốn minimize $V$ — lừa được Discriminator nhiều nhất.

Training loop alternate giữa hai bước: cập nhật $D$ vài bước → cập nhật $G$ một bước, lặp lại cho đến khi cân bằng.

> 🖼️ **[Ảnh placeholder #7 — Mode collapse]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. Left side labeled 'Training data': a colorful grid of 12 diverse cat photos — different breeds, colors, poses, expressions. An arrow labeled 'GAN với mode collapse' points to the right. Right side labeled 'Generator output': 12 nearly identical orange tabby kittens, all the same pose, looking slightly confused. A small sad emoji '😅' and a 'Mode Collapse!' label in red. Soft pastel colors, minimal clean background."*

#### Ưu và nhược điểm

**Ưu điểm:**
- **Chất lượng ảnh sinh ra tốt:** Do tính chất đối nghịch, Generator bị "ép" phải tạo ra các ảnh cực kỳ sắc nét và chi tiết để có thể qua mặt được Discriminator.
- **Tốc độ sinh ảnh cực nhanh:** Sau khi đã được huấn luyện, Generator là một feed-forward model.
Do đó, chỉ cần đưa một vector nhiễu (noise vector) vào và nhận ngay kết quả đầu ra.

**Nhược điểm:**
- **Quá trình huấn luyện không ổn định:** Việc cân bằng giữa Generator và Discriminator rất khó. Nếu một trong hai thành phần mạnh hơn quá nhiều, quá trình học sẽ sụp đổ và không hội tụ.
- **Mode Collapse:** Generator có thể overfit bằng cách chỉ học và tạo ra **một vài** ảnh trông rất thật mà lừa được Discriminator. Kết quả là mô hình chỉ sinh ra được một vài loại ảnh giống nhau, thiếu sự đa dạng, dù tập dữ liệu huấn luyện rất phong phú.

> 🖼️ **[Ảnh placeholder #8 — StyleGAN style mixing]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. Three cat portraits arranged in a row: left labeled 'Source A (coarse style)' showing a fluffy gray cat with a round face shape and big ears; middle labeled '= Style Mixing Result' showing a hybrid cat with A's face shape and B's color/texture; right labeled 'Source B (fine style)' showing a sleek black-and-white tuxedo cat with sharp whiskers. Small arrows showing 'coarse: shape, pose' flowing from A and 'fine: color, texture' flowing from B into the middle. Soft pastel colors, minimal clean background."*

#### Một số mô hình tiêu biểu trong nhóm

- **DCGAN (Radford et al., 2015)** — [paper](https://arxiv.org/abs/1511.06434) — sử dụng backbone CNN (deep convolutional) cho cả Generator và Discriminator, là baseline ổn định đầu tiên.
- **Conditional GAN - cGAN (Mirza & Osindero, 2014)** — [paper](https://arxiv.org/abs/1411.1784) — bổ sung điều kiện (class label, text) vào cả G và D để sinh ảnh có điều khiển.
- **Pix2Pix (Isola et al., 2017)** — [paper](https://arxiv.org/abs/1611.07004) — image-to-image translation với cặp ảnh có nhãn (paired): sketch → photo, ngày → đêm, semantic map → ảnh thật.
- **CycleGAN (Zhu et al., 2017)** — [paper](https://arxiv.org/abs/1703.10593) — image-to-image translation không cần cặp ảnh (unpaired), dùng **cycle-consistency loss** (ảnh nguồn → đích → ngược lại phải khôi phục ảnh nguồn). Ví dụ: ngựa ↔ ngựa vằn, ảnh thật ↔ tranh Monet.
- **Progressive Growing of GANs (Karras et al., 2017)** — [paper](https://arxiv.org/abs/1710.10196) — huấn luyện từ độ phân giải thấp (4×4) tăng dần lên cao (1024×1024), giúp ổn định và sinh ảnh high-res.
- **BigGAN (Brock et al., 2018)** — [paper](https://arxiv.org/abs/1809.11096) — scale lớn với class-conditional generation, đạt SOTA trên ImageNet.
- **StyleGAN / StyleGAN2 / StyleGAN3 (Karras et al., 2019–2021)** — [StyleGAN](https://arxiv.org/abs/1812.04948), [StyleGAN2](https://arxiv.org/abs/1912.04958), [StyleGAN3](https://arxiv.org/abs/2106.12423) — kiến trúc style-based generator với latent space disentangled, cho phép điều khiển style ở nhiều mức (coarse/fine) và mixing giữa các ảnh.
- **SAGAN - Self-Attention GAN (Zhang et al., 2018)** — [paper](https://arxiv.org/abs/1805.08318) — đưa self-attention vào GAN để mô hình hoá long-range dependency, cải thiện chất lượng đáng kể.

### 2.3. Nhóm mô hình dựa trên Transformer

Sự ra đời của Transformer cùng với cơ chế tự chú ý (self-attention) trong bài báo [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) đã mở ra một hướng tiếp cận mới cho image generation: coi ảnh như một **chuỗi token rời rạc** và sinh ảnh giống như cách Transformer sinh văn bản — token này sau token kia, hoặc đồng thời theo kiểu masked language model.

#### Mô tả ý tưởng và cơ chế hoạt động

Quy trình sinh ảnh bằng Transformer thường gồm ba bước:

1. **Bước 1 — Tokenize ảnh:** Dùng một mô hình autoencoder rời rạc (như **VQ-VAE** hoặc **VQ-GAN**) để biến ảnh $256 \times 256$ pixel thành một lưới $16 \times 16$ token rời rạc. Mỗi token là một chỉ số trong **codebook** (vocabulary cố định, ví dụ 8192 token). Bước này tương tự như tokenize văn bản trong NLP.

2. **Bước 2 — Sinh chuỗi token bằng Transformer:** Có hai cách tiếp cận chính:
    - **Autoregressive (kiểu GPT):** sinh từng token một theo thứ tự raster (trái-phải, trên-xuống), điều kiện hoá trên các token đã sinh trước đó và (tuỳ chọn) trên text prompt.
    - **Masked / Bidirectional (kiểu BERT):** ban đầu toàn bộ token bị mask, mô hình predict song song nhiều token, lặp lại trong vài bước để hoàn thiện. Cách này nhanh hơn autoregressive đáng kể.

3. **Bước 3 — Decode trở lại ảnh:** đưa chuỗi token đã sinh qua decoder của VQ-VAE để khôi phục ảnh ở không gian pixel.

> 🖼️ **[Ảnh placeholder #9 — Image tokenization với VQ-VAE]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. Left: a cute cat photo. Middle: an arrow labeled 'VQ-VAE Encoder' pointing to a 16×16 grid where each cell is a small colored tile representing a token (some red, some blue, some yellow). Right: a 'Codebook' shown as a small box of 8 colored swatches with index numbers, with arrows mapping each grid cell back to a swatch. Tiny robot mascot pointing at the codebook with a magnifying glass. Soft pastel colors, minimal clean background."*

> 🖼️ **[Ảnh placeholder #10 — Autoregressive token generation]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. A friendly pastel robot mascot sitting at a desk, writing tokens one by one onto a 4×4 grid from top-left to bottom-right in raster order. The first 6 tiles are already filled with colorful glowing tokens, the 7th tile has a glowing pen drawing it now, the remaining tiles are empty dotted squares. A speech bubble from the robot says 'Token 7 / 16…'. Above the desk, a thought bubble shows the Transformer architecture (stacked attention blocks). Soft pastel blue and pink colors, minimal clean background."*

> 🖼️ **[Ảnh placeholder #11 — MaskGIT vs Autoregressive]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style, split into two side-by-side panels. Left panel labeled 'Autoregressive (chậm)': a 4×4 grid being filled token by token from top-left, with a small clock showing many ticks. Right panel labeled 'MaskGIT (nhanh)': the same 4×4 grid starting all masked (gray ?), then in just 3 parallel steps the high-confidence tokens are unmasked together, the clock shows fewer ticks. A speed comparison arrow '⚡ 8× faster' between them. Soft pastel colors, minimal clean background."*

#### Ưu và nhược điểm

**Ưu điểm:**
- **Khả năng scale tốt:** Transformer scale tuyến tính theo số token và mạnh dần khi tăng tham số/dữ liệu — giống quy luật scaling của LLM.
- **Hiểu prompt phức tạp:** Tận dụng được pre-training NLP và cross-attention với text, hiểu được các prompt dài, nhiều thực thể, quan hệ không gian phức tạp.
- **Kiến trúc thống nhất với LLM:** dễ tích hợp text-image, image-text, multimodal vào cùng một backbone Transformer.

**Nhược điểm:**
- **Mất chi tiết do tokenize:** Ảnh bị nén về codebook hữu hạn nên chi tiết fine-grained (lỗ chân lông, sợi tóc) có thể bị mất.
- **Autoregressive chậm:** Sinh ảnh $1024 \times 1024$ với $64 \times 64 = 4096$ token cần 4096 lần forward — chậm hơn GAN nhiều.
- **Codebook collapse:** Quá trình train VQ-VAE có thể khiến phần lớn token trong codebook không được dùng, làm giảm chất lượng.

#### Một số mô hình tiêu biểu trong nhóm

- **VQ-VAE + Autoregressive Transformer (Razavi et al., 2019)** — [paper](https://arxiv.org/abs/1906.00446) — nền tảng cho hướng tiếp cận discrete latent + transformer.
- **DALL-E (Ramesh et al., 2021)** — [paper](https://arxiv.org/abs/2102.12092) — dVAE + GPT-style 12B params, mô hình text-to-image quy mô lớn đầu tiên dùng transformer.
- **VQ-GAN (Esser et al., 2020)** — [paper](https://arxiv.org/abs/2012.09841) — kết hợp VQ-VAE với GAN loss và perceptual loss để codebook học được token chất lượng cao hơn.
- **Parti - Pathways Autoregressive Text-to-Image (Yu et al., 2022)** — [paper](https://arxiv.org/abs/2206.10789) — scale autoregressive transformer lên 20B params, chất lượng cạnh tranh diffusion.
- **MaskGIT (Chang et al., 2022)** — [paper](https://arxiv.org/abs/2202.04200) — bidirectional masked image transformer, sinh ảnh nhanh hơn autoregressive 8× với chất lượng tương đương.
- **Muse (Chang et al., 2023)** — [paper](https://arxiv.org/abs/2301.00704) — masked transformer text-to-image dùng T5 text encoder, nhanh và hiệu quả hơn diffusion.

### 2.4. Nhóm mô hình Diffusion Models

**Diffusion Models** là một trong những kiến trúc mô hình đột phá và mạnh mẽ nhất trong lĩnh vực Generative AI nói chung và Image Generation nói riêng hiện nay.
Nền tảng lý thuyết của Diffusion Models được giới thiệu vào năm 2015 trong bài báo [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585) (Sohl-Dickstein et al.) trong khi ứng dụng cụ thể được áp dụng và mang lại bước đột phá được giới thiệu vào năm 2020 trong bài báo [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) (Ho et al.).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/blog-sharing/refs/heads/master/1_pixta_seminar/12_from_diffusion_to_flow/diffusion_model_img/banner.jpg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

Ý tưởng cốt lõi của Diffusion Model là điêu khắc từ một khối nhiễu không có nội dung cụ thể.
- Bắt đầu với một bức tượng hoàn hảo (một bức ảnh sạch, rõ nét).
- Từ từ phá hủy nó bằng cách thêm vào từng lớp "bụi" hoặc "nhiễu" (noise) cho đến khi nó trở thành một khối bụi/nhiễu không còn hình thù gì.
- Học cách đảo ngược quá trình này: Tức là học cách "thổi bụi" ra khỏi khối nhiễu để khôi phục lại bức tượng ban đầu.
- Sau khi đã học được kỹ năng "thổi bụi" này một cách thành thạo, mô hình có thể bắt đầu với một khối nhiễu hoàn toàn ngẫu nhiên và "điêu khắc" nó thành một bức ảnh mới toanh, chân thực và độc đáo.

Đó chính là triết lý của Diffusion Model, được chia thành hai quá trình chính:
- **Quá trình Thuận (Forward Process / Diffusion Process):** Thêm nhiễu vào ảnh.
- **Quá trình Nghịch (Reverse Process / Denoising Process):** Loại bỏ nhiễu để tạo ra ảnh.

> 🖼️ **[Ảnh placeholder #12 — Forward diffusion process]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. A sequence of 6 frames from left to right showing a clear cute bear face gradually being covered in more and more pastel dust/noise particles. Frame 1: crisp bear. Frame 2: slightly grainy. Frame 3: visibly noisy. Frame 4: very noisy with bear barely visible. Frame 5: mostly noise. Frame 6: pure pastel TV-static. Each frame labeled 't=0, t=200, t=400, t=600, t=800, t=1000'. A bear sculptor mascot at the left watches sadly. Soft pastel pink/purple/yellow noise particles, minimal clean background."*

> 🖼️ **[Ảnh placeholder #13 — Reverse denoising process]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. A sequence of 6 frames from left to right showing pure pastel noise gradually being 'sculpted' into a clear cute bear face. Frame 1: pure pastel static. Frame 2-5: bear emerging step by step from the noise. Frame 6: crystal clear bear. A bear-sculptor mascot blowing 'magic dust away' with a glowing brush, with a 'U-Net ε_θ' label floating above. Tiny sparkles around the emerging bear. Soft pastel colors, minimal clean background, magical mood."*

Về mặt toán học:

**Quá trình thuận** là một chuỗi Markov thêm Gaussian noise theo lịch trình $\{\beta_t\}_{t=1}^T$:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)$$

Sau $T$ bước (thường $T = 1000$), $x_T$ gần như là pure Gaussian noise.

**Quá trình nghịch** học cách đảo ngược: mô hình $p_\theta(x_{t-1} | x_t)$ được parameterize bằng **U-Net** (kiến trúc encoder-decoder với skip connection) dự đoán noise $\epsilon_\theta(x_t, t)$ đã được thêm vào ở bước $t$.

**Hàm mất mát** rút gọn của DDPM:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

Lúc inference, bắt đầu từ $x_T \sim \mathcal{N}(0, I)$ và áp dụng quá trình nghịch lặp $T$ bước để ra $x_0$.

> 🖼️ **[Ảnh placeholder #14 — U-Net architecture]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. A U-shaped neural network architecture drawn as stacked rounded blocks: on the left side going down (encoder) the blocks get smaller and labeled 'Conv 256→128→64→32'; at the bottom a small bottleneck block; on the right side going up (decoder) the blocks get bigger 'Conv 32→64→128→256'. Dashed glowing arrows connect corresponding encoder-decoder blocks ('skip connections'). Input on top-left: noisy bear at t=500. Output on top-right: predicted noise ε. A small clock icon showing 't=500' fed into the bottleneck. Soft pastel colors, minimal clean background."*

> 🖼️ **[Ảnh placeholder #15 — Latent vs Pixel Diffusion]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style, split into two panels. Top panel labeled 'Pixel Diffusion (chậm, tốn RAM)': denoising happens on a big 512×512 grid of bear pixels with many CPU/GPU sweat-drops emojis. Bottom panel labeled 'Latent Diffusion (nhanh, ít RAM)': first an encoder shrinks bear to a tiny 64×64 latent grid, denoising happens here much faster, then a decoder expands back to full 512×512 bear. A tiny 'VAE encoder/decoder' tag on the bottom panel. A speed comparison '8× faster, 1/8 RAM'. Soft pastel colors, minimal clean background."*

#### Ưu và nhược điểm

**Ưu điểm:**
- **Chất lượng và độ đa dạng ảnh vượt trội:** Diffusion Models có khả năng tạo ra những hình ảnh cực kỳ chi tiết, chân thực và đa dạng, vượt qua GANs.
- **Quá trình huấn luyện ổn định:** Tương tự VAEs, Diffusion Models có một hàm mục tiêu được định nghĩa rõ ràng, giúp quá trình huấn luyện rất ổn định và dễ hội tụ.
- **Hạn chế Mode Collapse:** Chúng rất ít khi gặp phải vấn đề sụp đổ chế độ như GANs, có khả năng học và tái tạo lại toàn bộ sự đa dạng của dữ liệu huấn luyện.

**Nhược điểm:**
- **Tốc độ sinh ảnh rất chậm:** Để tạo ra một ảnh, mô hình phải thực hiện quá trình khử nhiễu lặp đi lặp lại qua hàng chục, hàng trăm, thậm chí hàng nghìn bước.
Quá trình này chậm hơn rất nhiều so với một lần truyền thẳng của GAN.
Tuy nhiên, các nghiên cứu gần đây đang cải thiện tốc độ này một cách đáng kể.
- **Yêu cầu tính toán cao:** Cả quá trình huấn luyện và sinh ảnh đều đòi hỏi tài nguyên tính toán (GPU/TPU) rất lớn do bản chất lặp đi lặp lại của chúng.

#### Một số mô hình tiêu biểu trong nhóm

- **DDPM - Denoising Diffusion Probabilistic Models (Ho et al., 2020)** — [paper](https://arxiv.org/abs/2006.11239) — baseline gốc, đặt nền móng cho mọi diffusion model hiện đại.
- **DDIM (Song et al., 2020)** — [paper](https://arxiv.org/abs/2010.02502) — sampling **deterministic** rút gọn từ 1000 bước xuống 50–100 bước với chất lượng tương đương.
- **Classifier-Free Guidance (Ho & Salimans, 2022)** — [paper](https://arxiv.org/abs/2207.12598) — kỹ thuật kết hợp output có/không có condition để tăng độ "tuân thủ" prompt, là chuẩn cho mọi text-to-image hiện nay.
- **GLIDE (Nichol et al., 2021)** — [paper](https://arxiv.org/abs/2112.10741) — text-guided diffusion với CLIP guidance, tiền thân của DALL-E 2.
- **Latent Diffusion / Stable Diffusion (Rombach et al., 2022)** — [paper](https://arxiv.org/abs/2112.10752) — thực hiện diffusion trong latent space của một VAE pre-trained, **mở mã nguồn**, giảm chi phí tính toán đáng kể và mở ra hệ sinh thái ecosystem khổng lồ (LoRA, ControlNet, DreamBooth).
- **DALL-E 2 (Ramesh et al., 2022)** — [paper](https://arxiv.org/abs/2204.06125) — CLIP text embedding + diffusion prior + diffusion decoder.
- **Imagen (Saharia et al., 2022)** — [paper](https://arxiv.org/abs/2205.11487) — T5-XXL text encoder + cascaded diffusion (super-resolution nhiều tầng), chứng minh text encoder lớn quan trọng hơn diffusion model.
- **ControlNet (Zhang et al., 2023)** — [paper](https://arxiv.org/abs/2302.05543) — bổ sung control bằng edge map, depth, pose, segmentation map vào Stable Diffusion đã pre-trained mà không phá vỡ trọng số gốc.
- **SDXL (Podell et al., 2023)** — [paper](https://arxiv.org/abs/2307.01952) — Stable Diffusion XL với base + refiner, chất lượng cải thiện đáng kể.
- **Flux.1 (Black Forest Labs, 2024)** — rectified flow transformer, được xem là SOTA open-weight hiện tại cho text-to-image.

## 3. Các metrics trong image generation

Một mô hình sinh ảnh tốt cần đáp ứng hai tiêu chí chính:
- **Chất lượng (Quality/Fidelity):** Các ảnh được sinh ra phải rõ nét, chân thực và có thể nhận biết được đối tượng trong đó.
Ví dụ, nếu mô hình sinh ra ảnh một con chó, nó phải trông giống một con chó thật, chứ không phải một vệt mờ.
- **Đa dạng (Diversity/Variety):** Mô hình phải có khả năng sinh ra nhiều loại ảnh khác nhau, bao phủ được sự đa dạng của dữ liệu thật. Nó không nên chỉ sinh đi sinh lại một vài kiểu ảnh đẹp (hiện tượng này gọi là "mode collapse").

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/metrics.jpeg" style="width: 800px;"/>

### 3.1. Inception Score (IS)

Inception Score được giới thiệu vào năm 2016 trong bài báo [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) (Salimans et al.), sử dụng một mô hình Inception-v3 pretrained trên bộ dữ liệu ImageNet.
Mục tiêu của IS là đo lường đồng thời cả chất lượng và sự đa dạng của các ảnh được sinh ra mà **không cần đến dữ liệu ảnh thật**.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/is.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

Trực giác của IS dựa trên hai điều kiện cần đối với một mô hình sinh ảnh tốt:

- **Để đo chất lượng:** IS đưa một ảnh qua mô hình Inception-v3 và lấy prediction của mô hình:
    - Nếu phân phối xác suất đầu ra $p(y|x)$ có entropy thấp (tức là có một đỉnh nhọn ở một lớp), nghĩa là mô hình tự tin dự đoán ảnh đó thuộc về một lớp cụ thể nào đó.
    Suy ra, **Đây là một ảnh đẹp và rõ ràng nội dung.**
    - Nếu phân phối xác suất đầu ra $p(y|x)$ có entropy cao (tức là phân phối xác suất khá đồng đều giữa các lớp), nghĩa là mô hình không tự tin dự đoán ảnh đó thuộc về một lớp cụ thể nào đó.
    Suy ra, **Đây là một ảnh xấu và không rõ ràng nội dung.**
- **Để đo sự đa dạng:** IS đưa một tập hợp ảnh qua mô hình Inception-v3 và lấy tổng hợp prediction của mô hình:
    - Nếu tổng hợp của các phân phối xác suất đầu ra $p(y|x)$ có entropy thấp (tức là có một đỉnh nhọn ở một lớp), nghĩa là mô hình sinh ra các ảnh tập trung ở một lớp cụ thể nào đó.
    Suy ra, **Mô hình đang thiếu sự đa dạng.**
    - Nếu tổng hợp của các phân phối xác suất đầu ra $p(y|x)$ có entropy cao (tức là phân phối xác suất khá đồng đều giữa các lớp), nghĩa là mô hình sinh ra các ảnh phân bố đều trên nhiều lớp khác nhau.
    Suy ra, **Mô hình đang có sự đa dạng.**

> 🖼️ **[Ảnh placeholder #16 — IS quality vs diversity]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. A rabbit judge wearing a referee cap, sitting at a desk. Left of the desk: one cat photo entering an Inception-v3 box that outputs a sharp prediction bar chart with one tall bar 'CAT 99%' (low entropy, good quality), with a green check. Right of the desk: a stack of 100 diverse animal photos entering the same Inception-v3 box, outputting a flat bar chart spread across many classes (high entropy, good diversity), with another green check. Speech bubble from rabbit: 'IS = chất lượng × đa dạng'. Soft pastel colors, minimal clean background."*

Cách thức tính toán:

- Lấy một tập hợp ảnh được sinh ra (VD: 50,000 ảnh) từ mô hình của bạn.
- Đưa từng ảnh qua mô hình pre-trained Inception-v3 để nhận được vector xác suất 1000 chiều $p(y|x)$ (tương ứng với 1000 lớp của ImageNet).
- Tính toán phân phối xác suất trung bình (marginal distribution) bằng cách lấy trung bình của tất cả các vector $p(y|x)$ thu được $p(y) = E_x[p(y|x)]$.
- Tính toán Kullback-Leibler (KL) Divergence giữa $p(y|x)$ và $p(y)$ cho mỗi ảnh, sau đó lấy trung bình KL Divergence trên tất cả các ảnh.
Chúng ta muốn $p(y|x)$ (đặc trưng cho chất lượng - phải có entropy thấp) rất khác biệt so với $p(y)$ (đặc trưng cho sự đa dạng - phải có entropy cao).
- Công thức cuối cùng, hàm $exp$ được sử dụng để đưa kết quả về một thang đo dễ đọc hơn: $IS = \exp(\mathbb{E}_x[D_{KL}(p(y|x) \| p(y))])$
- Kết luận: Điểm IS càng cao càng tốt.
Nó cho thấy các ảnh sinh ra vừa có chất lượng cao (dự đoán tự tin) vừa đa dạng (bao phủ nhiều lớp).

#### Ví dụ

Xét hai mô hình sinh ảnh trên CIFAR-10:

- **Mô hình A (tốt):** sinh 50,000 ảnh đa dạng từ 10 lớp của CIFAR-10. Mỗi ảnh được Inception-v3 dự đoán tự tin vào một lớp ($p(y|x)$ entropy thấp), và tổng hợp lại $p(y)$ phân bố đều trên cả 10 lớp (entropy cao) → IS ≈ 9.5 → 11.
- **Mô hình B (mode collapse):** chỉ sinh ảnh con mèo. Mỗi ảnh đẹp, $p(y|x)$ entropy thấp ("cat"), nhưng $p(y)$ cũng tập trung ở lớp "cat" → KL Divergence nhỏ → IS ≈ 2–3.

Giá trị IS điển hình trên một số benchmark:

| Mô hình / Dataset | IS |
|---|---|
| CIFAR-10 ảnh thật | ~11.24 |
| BigGAN (ImageNet 128×128) | ~166 (1000 lớp) |
| StyleGAN2 (FFHQ) | ~5.0 (chỉ 1 lớp = mặt người, nên IS không phù hợp) |
| DCGAN (CIFAR-10) | ~6.4 |

#### Ưu và nhược điểm

**Ưu điểm:**
- **Đơn giản để tính toán.**
- **Không cần sử dụng ảnh thật trong quá trình đánh giá.**

**Nhược điểm:**
- **Không so sánh với dữ liệu thật:** IS chỉ nhìn vào các ảnh được sinh ra mà không đối chiếu chúng với phân phối của ảnh thật.
- **Dễ bị "qua mặt":** Một mô hình có thể học cách tạo ra **một ảnh hoàn hảo cho mỗi lớp** trong 1000 lớp của ImageNet. Nó sẽ nhận được điểm IS rất cao, nhưng thực tế nó không có khả năng sinh ra các biến thể khác của đối tượng.
- **Phụ thuộc vào ImageNet và Inception-v3:** Nó hoạt động không tốt với các bộ dữ liệu không có các lớp tương tự ImageNet và bị giới hạn bởi khả năng của mô hình Inception-v3.

### 3.2. Fréchet Inception Distance (FID)

Fréchet Inception Distance (FID) được giới thiệu vào năm 2017 trong bài báo [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) (Heusel et al.) và đã khắc phục được một số nhược điểm của IS.
FID đo lường "khoảng cách" giữa phân phối của các ảnh được sinh ra và phân phối của các ảnh thật.
Nó không chỉ nhìn vào đầu ra của lớp phân loại mà còn xem xét các đặc trưng sâu hơn bên trong mạng nơ-ron.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/fid.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

FID cũng sử dụng mô hình Inception-v3, tương tự IS nhưng thay vì lấy lớp đầu ra (softmax), FID lấy các vector đặc trưng (feature vectors) từ một layer trong model — thường là pool3 với 2048 chiều.
Vector này được coi là một biểu diễn cô đọng về nội dung của ảnh và FID sẽ sử dụng vector này để đo đạc đánh giá.

> 🖼️ **[Ảnh placeholder #17 — FID: hai phân phối Gaussian]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. A 2D feature space plotted on soft pastel grid background. Two overlapping bell-shaped Gaussian distribution clouds: one blue labeled 'Real images N(μ_r, Σ_r)' with little kawaii cat icons inside, one pink labeled 'Generated images N(μ_g, Σ_g)' with little kawaii fox-painted cats inside. A double-headed arrow between the two means showing 'Fréchet distance'. A rabbit judge holding a small scoreboard 'FID = 3.5 (rất tốt)'. Soft pastel colors, minimal clean background."*

Các bước tính FID:

- Chuẩn bị trước một tập hợp ảnh thật từ bộ dữ liệu của bạn (ví dụ: 10,000 ảnh).
- Lấy một tập hợp ảnh được sinh ra (VD: 50,000 ảnh) từ mô hình của bạn.
- Đưa từng ảnh của cả hai tập hợp ảnh qua mô hình Inception-v3 và thu thập các vector đặc trưng cho mỗi tập.
- Giả sử, các vector đặc trưng của mỗi ảnh trong mỗi tập tuân theo một phân phối Gaussian, ta tính toán vector giá trị kỳ vọng $\mu$ và ma trận hiệp phương sai $\Sigma$ cho cả hai tập: ảnh thật $(\mu_r, \Sigma_r)$ và ảnh sinh ra $(\mu_g, \Sigma_g)$.
- FID chính là khoảng cách Fréchet giữa hai phân phối Gaussian với công thức: $FID = \|\mu_r - \mu_g\|^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$
    - $\|\mu_r - \mu_g\|^2$ là khoảng cách bình phương giữa hai vector trung bình, đo lường sự khác biệt về nội dung trung bình của hai tập ảnh.
    - $Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$ là trace của ma trận, đo lường sự khác biệt về ma trận hiệp phương sai, tức là sự khác biệt về sự đa dạng và tương quan giữa các đặc trưng.
- Kết luận: Điểm FID càng thấp càng tốt.
FID bằng 0 có nghĩa là phân phối của ảnh sinh ra và ảnh thật giống hệt nhau (trong không gian đặc trưng của Inception).

#### Ví dụ

Giá trị FID điển hình trên một số benchmark text-to-image / image generation:

| Mô hình / Dataset | FID |
|---|---|
| Ảnh thật (lower-bound) | ~0 (idealize) |
| StyleGAN3 (FFHQ 1024) | ~3.0 |
| StyleGAN2 (FFHQ 1024) | ~3.5 |
| Progressive GAN (CelebA-HQ) | ~8.0 |
| Latent Diffusion (LDM-4, ImageNet) | ~3.6 |
| DCGAN (CIFAR-10) | ~35–40 |
| VAE thuần (CIFAR-10) | ~50+ |

Quy ước trực giác:
- **FID < 5:** chất lượng xuất sắc, gần như không phân biệt được với ảnh thật.
- **FID 5–15:** chất lượng tốt.
- **FID 15–50:** vẫn có thể nhận ra là ảnh sinh.
- **FID > 50:** chất lượng kém, dễ phát hiện artifact.

#### Ưu và nhược điểm

**Ưu điểm:**
- **So sánh trực tiếp với dữ liệu thật:** Đây là cải tiến quan trọng nhất so với IS.
- **Nhạy cảm với Mode Collapse:** Nếu mô hình chỉ sinh ra một vài loại ảnh, ma trận hiệp phương sai Σ_g sẽ rất khác so với Σ_r, dẫn đến điểm FID cao.
- **Tương quan tốt hơn với đánh giá của con người:** Điểm FID thấp thường tương ứng với các ảnh có chất lượng và độ đa dạng cao theo cảm nhận của con người.

**Nhược điểm:**
- **Yêu cầu tính toán lớn:** Cần một lượng lớn mẫu (thường là 10,000 đến 50,000) để có được điểm số ổn định.
- **Phụ thuộc vào ImageNet và Inception-v3:** Nó hoạt động không tốt với các bộ dữ liệu không có các lớp tương tự ImageNet và bị giới hạn bởi khả năng của mô hình Inception-v3.
- **Giả định Gaussian không phải lúc nào cũng đúng:** Phân phối feature thực tế hiếm khi là Gaussian thuần — đây là động lực ra đời của CMMD.

### 3.3. CLIP-MMD (CMMD)

CMMD (CLIP Maximum Mean Discrepancy) được giới thiệu vào năm 2024 trong bài báo [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603) (Jayasumana et al.) và đã nổi lên như một thước đo (metric) tiêu chuẩn mới, được kỳ vọng sẽ khắc phục những hạn chế của các phương pháp trước đây như FID.

CMMD là sự kết hợp giữa hai thành phần:
- **Mô hình CLIP (Contrastive Language-Image Pre-training):** Thay vì sử dụng mạng Inception-v3, CMMD dùng CLIP để trích xuất đặc trưng vì CLIP có khả năng hiểu hình ảnh gần với thị giác và ngữ nghĩa của con người hơn.
- **Maximum Mean Discrepancy (MMD):** Đây là một phương pháp thống kê dùng để đo lường khoảng cách giữa hai phân phối dữ liệu (giữa ảnh thật và ảnh do máy sinh ra), thay thế cho độ đo Fréchet.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/cmmd.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

**Maximum Mean Discrepancy (MMD)** là một thước đo khoảng cách giữa hai phân phối xác suất $P$ và $Q$ dựa trên *mean embedding* trong một không gian Hilbert nhân tái sinh (Reproducing Kernel Hilbert Space - RKHS) được xác định bởi một kernel $k$.

Công thức của MMD bình phương:

$$MMD^2(P, Q) = \mathbb{E}_{x, x' \sim P}[k(x, x')] + \mathbb{E}_{y, y' \sim Q}[k(y, y')] - 2 \, \mathbb{E}_{x \sim P, y \sim Q}[k(x, y)]$$

Trực giác:
- Nếu $P = Q$, ba kỳ vọng triệt tiêu lẫn nhau và $MMD^2 = 0$.
- Nếu $P \neq Q$, $MMD^2 > 0$ và càng lớn khi hai phân phối càng khác nhau.

Kernel thường dùng là **Gaussian RBF**: $k(x, y) = \exp\left(-\dfrac{\|x - y\|^2}{2\sigma^2}\right)$.

Khác với FID — vốn giả định feature là Gaussian và tính khoảng cách Fréchet giữa hai Gaussian — MMD **không yêu cầu giả định phân phối**, nó so sánh đầy đủ mọi moment thông qua kernel trick.

> 🖼️ **[Ảnh placeholder #18 — CMMD với CLIP encoder và kernel MMD]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. A two-pan balance scale in the center. Left pan holds a basket labeled 'Ảnh thật' with kawaii real cat photos. Right pan holds a basket labeled 'Ảnh sinh ra' with kawaii AI-generated cats. Above the scale floats a glowing CLIP encoder box (drawn as a stylish robot with two eyes — one looking at images, one looking at text), turning both image batches into colorful embedding dots. A small 'Kernel MMD' sticker on the balance arm. A rabbit judge holding a card 'CMMD = 0.45 (low → similar)'. Soft pastel teal and pink colors, minimal clean background."*

Quy trình tính CMMD:

1. Chuẩn bị 2 tập ảnh — ảnh thật (reference) và ảnh sinh ra.
2. Trích đặc trưng bằng **CLIP image encoder** (ví dụ ViT-L/14), không phải Inception-v3 như FID.
3. Áp dụng công thức MMD trên hai tập embedding CLIP vừa thu được.
4. Trả về giá trị CMMD — **càng thấp càng tốt**.

#### Ví dụ

So sánh CMMD vs FID trên cùng một dataset:

| Điều kiện | FID | CMMD |
|---|---|---|
| 50,000 sample | 3.2 | 0.42 |
| 5,000 sample | 4.8 (lệch +50%) | 0.45 (lệch +7%) |
| 500 sample | 12.0 (không tin cậy) | 0.48 (vẫn ổn) |

Một trường hợp nổi tiếng FID "phán đoán sai" mà CMMD khắc phục được:
- Thêm nhiễu Gaussian rất nhỏ (mắt thường không thấy) vào ảnh thật → **FID tăng vọt từ 0 lên hàng chục** vì giả định Gaussian bị vi phạm.
- CMMD chỉ tăng nhẹ vì nó so sánh phân phối CLIP feature thực sự, không giả định Gaussian.

Một ví dụ khác: progressive distillation Stable Diffusion → 4 steps. FID nói chất lượng giảm mạnh, nhưng human eval và CMMD đồng thuận rằng chất lượng giảm rất nhẹ — chứng minh CMMD tương quan tốt hơn với cảm nhận con người.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Không cần giả định Gaussian:** so sánh phân phối feature thực, ổn định hơn FID khi phân phối bị skew.
- **Ổn định với sample nhỏ:** CMMD cho điểm tin cậy với chỉ 500–5,000 sample, trong khi FID cần 10,000–50,000.
- **CLIP encoder hiểu ngữ nghĩa:** vector CLIP gần gũi với thị giác con người hơn Inception-v3 (vốn chỉ train trên 1000 lớp ImageNet).
- **Ít bias hơn FID:** không bị "fool" bởi noise/blur nhỏ.

**Nhược điểm:**
- **Vẫn phụ thuộc vào CLIP:** kế thừa các bias của CLIP (chủng tộc, giới tính, văn hoá phương Tây) đã có trong dữ liệu train.
- **Nhạy với kernel choice:** kết quả phụ thuộc vào loại kernel và bandwidth $\sigma$.
- **Chưa phổ biến bằng FID:** cộng đồng quen với FID, nên báo cáo CMMD đôi khi cần kèm FID để so sánh.

### 3.4. Learned Perceptual Image Patch Similarity (LPIPS)

LPIPS (Learned Perceptual Image Patch Similarity), hay còn gọi là "Perceptual Loss", được giới thiệu trong bài báo [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) (Zhang et al., 2018), là một thước đo được thiết kế để đánh giá sự tương đồng giữa hai hình ảnh theo cách gần giống với cách con người cảm nhận.

Thay vì so sánh từng pixel một cách máy móc, LPIPS sử dụng một mạng nơ-ron sâu (Deep Neural Network) đã được huấn luyện trước để trích xuất các đặc trưng phức tạp và so sánh chúng.
Điểm LPIPS càng thấp thì hai hình ảnh càng giống nhau về mặt tri giác.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/lpips.jpeg" style="width: 800px;"/>

#### Các metrics trước LPIPS

Ta sử dụng các thước đo như sau để đánh giá mức độ tương đồng của hai hình ảnh.
Tuy nhiên, các thước đo này có một nhược điểm lớn: chúng không tương quan tốt với nhận thức của con người.
- **L1 Loss (Mean Absolute Error - MAE):** Tính trung bình của trị tuyệt đối của sự khác biệt giữa các pixel.
- **L2 Loss (Mean Squared Error - MSE):** Tính trung bình của bình phương sự khác biệt giữa các pixel.
- **PSNR (Peak Signal-to-Noise Ratio):** Dựa trên MSE.
- **SSIM (Structural Similarity Index):** Cố gắng đo lường sự thay đổi về cấu trúc, độ sáng và độ tương phản.

#### Mô tả ý tưởng và cơ chế hoạt động

Quy trình tính điểm LPIPS cho cặp ảnh $(x_1, x_2)$:

- Lấy hai ảnh cần so sánh $x_1$ và $x_2$.
- Đưa từng ảnh qua mô hình pre-trained CNN (thường là VGG, AlexNet hoặc SqueezeNet) để nhận được các vector đặc trưng ở nhiều lớp của mô hình.
LPIPS mong muốn nắm bắt các đặc trưng bậc thấp như cạnh, góc, màu sắc... ở những layer nông và nắm bắt các đặc trưng bậc cao, mang tính ngữ nghĩa hơn như các bộ phận của vật thể ở những layer sâu.
- Các vector đặc trưng ở mỗi lớp của hai ảnh sẽ được chuẩn hóa (unit-normalize) và tính khoảng cách L2.
Kết quả là một giá trị đo lường sự khác biệt về đặc trưng tại lớp đó.
- Tính tổng có trọng số các giá trị đo lường sự khác biệt ở các lớp để ra được điểm LPIPS cuối cùng.
    - Các nhà phát triển LPIPS đã huấn luyện một mạng tuyến tính nhỏ để học các trọng số cho mỗi lớp.
    - Các trọng số này được học từ bộ dữ liệu **BAPPS** (Berkeley-Adobe Perceptual Patch Similarity Dataset) gồm các phán đoán của con người về sự tương đồng của các cặp ảnh.
    - Mục tiêu là để trọng số này phản ánh tầm quan trọng của từng loại đặc trưng đối với nhận thức của con người.
    - Điểm LPIPS cuối cùng là tổng của các khoảng cách ở mỗi lớp nhân với trọng số tương ứng đã học được.
- LPIPS thường được dùng kèm với các metric khác như PSNR/SSIM/FID để có đánh giá toàn diện.

> 🖼️ **[Ảnh placeholder #19 — LPIPS vs L2 trên 3 ảnh mèo]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. Three cute cat photos arranged in a row at the top, labeled: 'Original (gốc)', 'Shifted 1px (dịch nhẹ)', 'Blurred (mờ)'. Below each pair-comparison, two horizontal bar charts. Top chart labeled 'L2 distance (sai)': long bar for Original-vs-Shifted (red 'high!'), tiny bar for Original-vs-Blurred (green 'low'). Bottom chart labeled 'LPIPS (đúng tri giác)': tiny bar for Original-vs-Shifted (green 'low'), long bar for Original-vs-Blurred (red 'high'). A rabbit judge below pointing at LPIPS chart with a thumbs up. Soft pastel colors, minimal clean background."*

#### Ví dụ minh hoạ hạn chế của L1/L2

Xét 3 ảnh:
- **Ảnh gốc (Original):** Một bức ảnh con mèo sắc nét.
- **Ảnh dịch chuyển (Shifted):** Cùng bức ảnh đó nhưng được dịch sang phải 1 pixel.
- **Ảnh mờ (Blurred):** Một phiên bản bị làm mờ của ảnh gốc.

Đánh giá 3 ảnh trên:
- **Theo mắt người:** Ảnh gốc (1) và ảnh dịch chuyển (2) gần như giống hệt nhau. Ảnh mờ (3) thì có chất lượng kém hơn hẳn.
- **Theo L2 Loss (MSE):** Điểm L2 giữa ảnh gốc (1) và ảnh dịch chuyển (2) sẽ rất cao (tức là rất khác nhau) vì mọi pixel đều bị lệch.
Ngược lại, điểm L2 giữa ảnh gốc (1) và ảnh mờ (3) có thể sẽ thấp hơn (tức là giống nhau hơn), điều này hoàn toàn trái ngược với cảm nhận của chúng ta.
- **Theo LPIPS:** điểm LPIPS giữa (1)-(2) rất thấp (đúng với mắt người), trong khi LPIPS giữa (1)-(3) cao hơn rõ rệt — phản ánh đúng tri giác.

Vấn đề này xảy ra vì L1/L2 chỉ quan tâm đến giá trị pixel tại đúng một vị trí, chúng không hiểu được "khái niệm" con mèo, kết cấu lông hay các đặc trưng bậc cao.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Tương quan cao với tri giác con người:** Cách thức đánh giá của LPIPS gần với cách con người quan sát và đánh giá.
- **Bất biến với các thay đổi nhỏ:** LPIPS không quá nhạy cảm với các phép dịch chuyển, xoay nhẹ hoặc biến dạng nhỏ mà không làm thay đổi nội dung chính của ảnh.
- **Đánh giá được cả cấu trúc và phong cách:** Bằng cách sử dụng các lớp sâu, LPIPS có thể so sánh sự tương đồng về mặt ngữ nghĩa và phong cách, chứ không chỉ là kết cấu bề mặt.

**Nhược điểm:**
- **Chi phí tính toán cao:** LPIPS đòi hỏi phải thực hiện tính toán với mô hình CNN nên tốn nhiều tài nguyên và thời gian hơn.
- **Phụ thuộc vào mô hình CNN:** Kết quả của LPIPS phụ thuộc vào mạng CNN được sử dụng.
- **Không nắm bắt được mọi khía cạnh:** LPIPS có thể không đánh giá tốt các lỗi về mặt logic hoặc ngữ cảnh toàn cục trong một bức ảnh (VD: một người có ba tay).

### 3.5. Human evaluation

Đánh giá bởi con người (Human Evaluation) đóng một vai trò cực kỳ quan trọng và không thể thiếu trong lĩnh vực tạo sinh ảnh (Image Generation).
Mặc dù chúng ta có các chỉ số đánh giá tự động nhưng chúng vẫn còn nhiều hạn chế và không thể nắm bắt được toàn bộ chất lượng của một bức ảnh do AI tạo ra.

Đánh giá của con người được coi là thước đo cuối cùng và chính xác nhất về chất lượng của một mô hình tạo sinh ảnh.
Mọi chỉ số tự động đều được phát triển và kiểm chứng bằng cách so sánh xem chúng có tương quan tốt với nhận xét của con người hay không.
Nếu một chỉ số tự động cho điểm cao một mô hình mà con người lại đánh giá thấp, thì chỉ số đó được xem là chưa hiệu quả.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/human_evaluation.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

Có hai dạng đánh giá phổ biến:

- **Đánh giá so sánh (Pairwise / 2AFC):** Người đánh giá được cho xem hai bức ảnh và được yêu cầu chọn ra bức ảnh tốt hơn dựa trên một tiêu chí cụ thể (VD: tính chân thật, độ thẩm mỹ, độ khớp prompt ...). Đây là cách đo lường relative quality, dễ cho người đánh giá hơn vì không cần thang đo tuyệt đối.
- **Đánh giá theo thang đo (Likert / Absolute):** Người đánh giá cho điểm một bức ảnh theo thang điểm (VD: từ 1 đến 5 hoặc 1 đến 10) cho các tiêu chí như chất lượng hình ảnh, tính thẩm mỹ, sự phù hợp với prompt.

Để bảo đảm tin cậy, người ta thường thuê nhiều annotator (3-5 người cho mỗi ảnh), đo **inter-rater agreement** (ví dụ Krippendorff's alpha, Cohen's kappa) và loại bỏ annotator có agreement thấp.

> 🖼️ **[Ảnh placeholder #20 — Pairwise human evaluation]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. Three rabbit judges sitting at a desk wearing referee caps, each holding a clipboard. In front of them on the desk, two cat photos side by side labeled 'Ảnh A' and 'Ảnh B'. A speech bubble from the middle rabbit asks 'Ảnh nào đẹp hơn?'. Above each rabbit's head, a thumb-up emoji pointing at their chosen image. On the right side of the panel, a Likert scale bar (1-5 stars) with a glowing star on '4'. Soft pastel colors, warm friendly judging atmosphere, minimal clean background."*

#### Ví dụ

Một số phương pháp đánh giá phổ biến trong thực tế:

- **Two-Alternative Forced Choice (2AFC):** cho 2 ảnh A/B từ 2 mô hình khác nhau, hỏi "ảnh nào đẹp hơn?" → tính **win-rate** của mỗi mô hình. Ví dụ: so sánh SDXL vs DALL-E 3 trên cùng 100 prompt → SDXL thắng 42%, DALL-E 3 thắng 58%.
- **Likert scale 1-5:** chấm điểm cho từng ảnh trên 3 trục: *Image Quality*, *Aesthetic*, *Text-Image Alignment*. Ví dụ: Stable Diffusion 1.5 trên COCO captions: Quality 3.8, Aesthetic 3.2, Alignment 3.5.
- **Elo rating (LMArena style):** mỗi cặp so sánh được dùng để update Elo rating của các mô hình — giống như cờ vua. Áp dụng phổ biến trong các leaderboard như **Chatbot Arena** (text), **GenAI Arena** (image), **WildBench**. Mô hình SOTA hiện tại thường có Elo 1300-1500.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Tiêu chuẩn vàng:** human evaluation là chuẩn cuối cùng để hiệu chỉnh và validate mọi metric tự động.
- **Nắm bắt được sắc thái:** con người dễ dàng nhận ra các lỗi tinh tế mà metric tự động bỏ qua (sai cấu trúc tay, biểu cảm sai, văn hoá không phù hợp).
- **Đo prompt-alignment trực tiếp:** đặc biệt quan trọng với text-to-image, nơi metric như FID không đánh giá được mức độ khớp prompt.

**Nhược điểm:**
- **Tốn kém và tốn thời gian:** Cần phải thuê nhiều người để đánh giá hàng ngàn, hàng triệu bức ảnh.
- **Tính chủ quan và thiên vị:** Đánh giá có thể khác nhau giữa những người khác nhau do sở thích, nền tảng văn hóa.
Do đó, ta cần đo đạc chỉ số độ đồng thuận giữa những người đánh giá.
- **Thiếu nhất quán:** Cùng một người có thể đưa ra những đánh giá khác nhau vào những thời điểm khác nhau.
- **Không scale được:** không thể dùng human evaluation làm signal trong vòng lặp huấn luyện mô hình.

### 3.6. Một số chỉ số đánh giá phụ khác

#### 3.6.1. CLIP Score

CLIP Score được giới thiệu trong bài báo [CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://arxiv.org/abs/2104.08718) (Hessel et al., 2021), được xây dựng dựa trên mô hình CLIP — là một metric định lượng mức độ tương đồng về mặt ngữ nghĩa giữa một hình ảnh và một đoạn mô tả văn bản, được sử dụng rất nhiều trong quá trình đánh giá mô hình Text-to-Image.
Điểm số càng cao, hình ảnh càng khớp với mô tả.

Chi tiết hơn về CLIP và một số biến thể nâng cấp đã được mình viết trong [bài viết này](/blog/transfer-learning-weakly-semi-un-va-self-supervised-learning).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/clip_score.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

- Lấy một cặp: ảnh được sinh ra và prompt dùng để sinh ảnh đó.
- Dùng Text Encoder của CLIP để biến prompt thành text_embedding.
- Dùng Image Encoder của CLIP để biến ảnh thành image_embedding.
- Tính toán độ tương đồng cosine (cosine similarity) giữa hai vector text_embedding và image_embedding.
- Kết quả của phép tính này chính là CLIP Score:
    - **Điểm cao (gần 1):** Hai vector gần như chỉ về cùng một hướng trong không gian vector, nghĩa là hình ảnh và văn bản có sự tương đồng ngữ nghĩa cao.
    - **Điểm thấp (gần 0 hoặc âm):** Hai vector chỉ về các hướng khác nhau, cho thấy hình ảnh và văn bản không liên quan.

> 🖼️ **[Ảnh placeholder #21 — CLIP Score: text-image cosine]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. Top-left: a paper card with text 'một quả táo đỏ' goes into a 'Text Encoder' box that outputs a glowing pink arrow vector. Bottom-left: a kawaii red apple photo goes into an 'Image Encoder' box that outputs a glowing teal arrow vector. The two arrow vectors meet on the right in a small circular embedding space — pointing in almost the same direction. A small angle θ between them is labeled '≈ 18°' and 'cosine ≈ 0.95 ✓'. A rabbit judge holds a card 'CLIP Score: 0.32 (high)'. Soft pastel colors, minimal clean background."*

#### Ví dụ

| Prompt | Ảnh được sinh | CLIP Score |
|---|---|---|
| "một con mèo cam ngồi trên ghế gỗ" | mèo cam đang ngồi trên ghế gỗ | ~0.32 (cao, khớp) |
| "một con mèo cam ngồi trên ghế gỗ" | một con chó nâu trên giường | ~0.15 (thấp, không khớp) |
| "vương quốc trên mây với rồng đỏ" | thành phố nổi với rồng đỏ | ~0.30 (cao, khớp) |
| "ảnh đen trắng phong cách Picasso" | ảnh màu chân thực | ~0.20 (thấp, sai style) |

Quy ước thang đo điển hình:
- **0.25 - 0.35:** ảnh khớp prompt tốt (mức chấp nhận được trong production).
- **0.20 - 0.25:** khớp một phần.
- **< 0.20:** không khớp.

Lưu ý: CLIP Score thường nhân với 100 (CLIP Score thô × 2.5) để có thang dễ đọc hơn (25–35 thay vì 0.25–0.35).

#### Ưu và nhược điểm

**Ưu điểm:**
- **Hiểu ngữ nghĩa:** Nó không chỉ so khớp từ khóa, mà còn hiểu được các khái niệm, phong cách, và mối quan hệ phức tạp.
- **Reference-free:** không cần ảnh ground-truth, chỉ cần prompt và ảnh sinh ra.
- **Tự động và nhanh:** dễ tích hợp vào pipeline đánh giá hàng loạt.

**Nhược điểm:**
- **Bias của CLIP:** CLIP được huấn luyện trên dữ liệu từ internet, nên nó cũng "học" cả những thiên kiến có sẵn trong dữ liệu đó (VD: thiên kiến về giới tính, chủng tộc).
- **Không phải là thước đo về thẩm mỹ:** Một ảnh có thể có CLIP Score rất cao (khớp hoàn hảo với prompt) nhưng trông lại không đẹp hoặc kỳ dị về mặt bố cục vì CLIP không phải là một nhà phê bình nghệ thuật.
- **Có thể bị "đánh lừa":** Đôi khi, các hình ảnh chứa văn bản (VD: ảnh có chữ "Apple" viết trên đó) có thể đạt điểm cao khi prompt là "quả táo", mặc dù đó không phải là thứ người dùng muốn.

#### 3.6.2. NIMA Score

NIMA là viết tắt của Neural Image Assessment là một mô hình được thiết kế để đánh giá chất lượng của một hình ảnh theo cách mà con người cảm nhận được giới thiệu trong bài báo [NIMA: Neural Image Assessment](https://arxiv.org/pdf/1709.05424) (Talebi & Milanfar, 2018).

Trong khi đó, NIMA cố gắng mô phỏng sự đánh giá của con người, không chỉ xem xét chất lượng kỹ thuật (technical quality) mà còn cả chất lượng thẩm mỹ (aesthetic quality).
Một bức ảnh có thể sắc nét về mặt kỹ thuật nhưng bố cục, màu sắc lại không hài hòa, NIMA có thể nhận ra điều này.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/nima_score.jpeg" style="width: 800px;"/>

#### Mô tả ý tưởng và cơ chế hoạt động

NIMA sử dụng một mô hình như Inception, VGG hay MobileNet và được tinh chỉnh trên một bộ dữ liệu lớn gồm các hình ảnh đã được con người chấm điểm về mặt thẩm mỹ như bộ dữ liệu **AVA - Aesthetic Visual Analysis** (~250,000 ảnh).

Output của NIMA không phải là một con số duy nhất, mà là một phân phối điểm số (từ 1 đến 10).
Từ phân phối này, chúng ta có thể tính ra điểm trung bình và độ lệch chuẩn.
Điểm trung bình càng cao, hình ảnh càng được đánh giá là có chất lượng và tính thẩm mỹ tốt.

Trong bài toán sinh ảnh, mục tiêu là tạo ra những hình ảnh không chỉ đúng với mô tả (prompt) mà còn phải đẹp, chân thực và hấp dẫn về mặt thị giác. Đây chính là lúc NIMA phát huy vai trò của mình.

NIMA có thể tự động chấm điểm cho tất cả các ảnh được tạo ra.
Và ta có thể chỉ giữ lại những ảnh có NIMA score cao, loại bỏ những ảnh bị lỗi, mờ, hoặc có bố cục xấu hoặc ta có thể xếp hạng kết quả cho người dùng, ưu tiên hiển thị những ảnh có điểm NIMA cao nhất trước.

#### Ví dụ

| Loại ảnh | NIMA trung bình |
|---|---|
| Ảnh phong cảnh chuyên nghiệp, bố cục rule-of-thirds tốt | ~6.5–7.5 |
| Ảnh chân dung studio sắc nét, ánh sáng đẹp | ~6.0–7.0 |
| Ảnh selfie thông thường | ~4.5–5.5 |
| Ảnh mờ, lệch bố cục, ánh sáng kém | ~3.0–4.0 |
| Ảnh bị artifact nghiêm trọng (overexposure, blur nặng) | ~1.5–3.0 |

**Workflow điển hình trong production:**
1. Sinh 100 ảnh từ Stable Diffusion với cùng một prompt + 100 seed khác nhau.
2. Chấm NIMA cho cả 100 ảnh.
3. Giữ top-10 ảnh có NIMA cao nhất để hiển thị cho người dùng.
4. (Tuỳ chọn) dùng top-1 làm "best result" mặc định.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Tương quan tốt với cảm nhận của con người:** Gần gũi với cách con người đánh giá một bức ảnh hơn các chỉ số kỹ thuật thuần túy.
- **Tự động và nhanh chóng:** Cho phép đánh giá hàng loạt mà không cần sự can thiệp của con người.
- **Linh hoạt:** Có thể được sử dụng trong quá trình huấn luyện và dự đoán của mô hình.

**Nhược điểm:**
- **Thiên kiến (Bias):** NIMA được huấn luyện trên một bộ dữ liệu cụ thể, do đó, nó có thể có "thiên kiến" và chấm điểm cao hơn cho những loại ảnh theo phong cách mà bộ dữ liệu này cho là đẹp, trong khi có thể chấm điểm thấp hơn cho các phong cách khác.
- **Không hiểu prompt:** NIMA chỉ chấm thẩm mỹ tuyệt đối, không biết ảnh có khớp với prompt hay không — cần kết hợp với CLIP Score.

#### 3.6.3. ArcFace Score

ArcFace không được tạo ra cho bài toán sinh ảnh mà là một công nghệ đột phá trong lĩnh vực nhận dạng khuôn mặt (Face Recognition) được giới thiệu trong bài báo [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698) (Deng et al., 2019).

Mục tiêu của ArcFace là tạo ra một vector đặc trưng (feature embedding) cho mỗi khuôn mặt.
Vector này có đặc điểm rất đặc biệt:
- Các khuôn mặt của cùng một người sẽ có vector đặc trưng rất gần nhau trong không gian vector.
- Các khuôn mặt của những người khác nhau sẽ có vector đặc trưng rất xa nhau.

Khi có hai vector đặc trưng của hai khuôn mặt, chúng ta có thể tính cosine similarity giữa chúng, nằm trong khoảng từ -1 đến 1 (1 khi hai vector hoàn toàn tương đồng, 0 khi hai vector hoàn toàn không tương đồng, -1 khi hai vector ngược chiều nhau), chính là ArcFace Score.
- ArcFace Score gần 1: Hai khuôn mặt gần như chắc chắn là của cùng một người.
- ArcFace Score gần 0 hoặc âm: Hai khuôn mặt là của hai người khác nhau.

Dựa vào ArcFace Score, ta có thể đo lường mức độ tương đồng về nhận dạng giữa hai khuôn mặt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/arcface_score.jpeg" style="width: 800px;"/>

Chính vì khả năng "đo lường nhận dạng" xuất sắc này, các nhà nghiên cứu đã mượn ArcFace để giải quyết một trong những thách thức lớn nhất của việc sinh ảnh khuôn mặt: **Làm sao để ảnh được tạo ra trông giống hệt một người cụ thể?**

ArcFace score rất hữu ích trong việc sử dụng image generation trong các ứng dụng sinh ảnh mới từ một gương mặt cho trước.
ArcFace score cung cấp một chỉ số định lượng và khách quan, thay vì chỉ dựa vào cảm nhận của mắt người, chúng ta có thể tính ArcFace score giữa mỗi ảnh được tạo ra và ảnh gốc.
Ảnh nào có score cao nhất thì được xem là ảnh bảo toàn nhận dạng tốt nhất.
Điều này rất hữu ích trong việc tự động lọc và chọn lựa kết quả.

#### Mô tả ý tưởng và cơ chế hoạt động

- **Tạo vector tham chiếu:** Một hoặc một nhóm ảnh gốc của người A được đưa qua mô hình ArcFace đã được huấn luyện sẵn để trích xuất vector đặc trưng.
Nếu ta có một nhóm ảnh gốc của người A, ta có thể lấy trung bình các vector đặc trưng để tạo ra vector tham chiếu.
- **Bắt đầu sinh ảnh:** Sử dụng mô hình image generation để sinh ra loạt ảnh mới của người A đó.
- **Trích xuất vector và so sánh:** Các ảnh mới được sinh ra của người A sẽ được đưa qua mô hình ArcFace để trích xuất vector đặc trưng.
Ta lấy các vector đặc trưng này tính toán cosine similarity với vector tham chiếu ở Bước 1 để ra được ArcFace Score cho mỗi ảnh mới được sinh.
- **Lọc các ảnh không đạt yêu cầu:** Ta cần phân tích để chọn ra một ngưỡng ArcFace Score Threshold phù hợp để lọc ra những ảnh mới sinh "giống" nhất với những ảnh gốc của người A.

> 🖼️ **[Ảnh placeholder #22 — ArcFace identity preservation]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. In the center, a large reference photo of person A drawn as a kawaii bear character labeled 'Reference vector v_ref'. Around it in a circle, 5 generated bear portraits each with a small score label: 'sim 0.82 ✓', 'sim 0.71 ✓', 'sim 0.65 ✓', 'sim 0.42 ✗', 'sim 0.18 ✗' — the high-score ones drawn as similar-looking bears, the low-score ones as obviously different bear faces. A glowing 'threshold = 0.5' dashed circle separates accepted from rejected. A rabbit judge nodding approvingly at the high-score ones. Soft pastel colors, minimal clean background."*

#### Ví dụ

| Tình huống | Cosine similarity | Kết luận |
|---|---|---|
| Ảnh tham chiếu A vs ảnh sinh ra "A đang cười" | ~0.75 | Cùng người (đạt) |
| Ảnh tham chiếu A vs ảnh sinh ra "A đeo kính" | ~0.70 | Cùng người (đạt) |
| Ảnh tham chiếu A vs ảnh sinh ra "A 60 tuổi" | ~0.55 | Cùng người, biên (ranh ngưỡng) |
| Ảnh tham chiếu A vs ảnh sinh ra phong cách Picasso | ~0.30 | Khác hoặc style mạnh |
| Ảnh tham chiếu A vs ảnh sinh ra của người B | ~0.10 | Khác người (loại) |

Threshold thường dùng trong ứng dụng identity-preserving generation (DreamBooth, IP-Adapter, InstantID...): **0.5–0.6** — tuỳ vào mức độ nghiêm ngặt cần thiết.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Độ chính xác cao:** ArcFace là một trong những phương pháp nhận dạng khuôn mặt hàng đầu, do đó nó đảm bảo việc bảo toàn nhận dạng rất tốt.
- **Bền vững với thay đổi:** Nó có khả năng nhận ra một người ngay cả khi có sự thay đổi về góc mặt, ánh sáng, biểu cảm.
- **Định lượng được**: Cung cấp một con số cụ thể để đo lường, giúp tự động hóa việc đánh giá và tối ưu.

**Nhược điểm:**
- **Bộ ảnh tham chiếu rõ ràng:** Ta cần chuẩn bị bộ ảnh tham chiếu rõ ràng với từng người. Một bộ ảnh tham chiếu bị che khuất hay góc mặt quay sẽ khiến cho kết quả đánh giá ArcFace Score bị sai lệch rất đáng kể.
- **Không hiểu về ngữ cảnh:** ArcFace chỉ quan tâm đến nhận dạng và không hiểu các yêu cầu khác như "vẽ theo phong cách Picasso". Điều này đôi khi tạo ra sự xung đột giữa việc "giống người thật" và "giống phong cách nghệ thuật".
- **Nhạy cảm với ArcFace Score Threshold:** Để chọn ra được những tấm ảnh giống với chủ thể nhất, ta cần chọn ArcFace Score Threshold phù hợp với mỗi chủ thể của hình ảnh.

## 4. Các thách thức của bài toán image generation

Cũng giống như các bài toán khác trong lĩnh vực liên quan đến Generative AI nói chung, image generation cũng gặp phải nhiều thách thức và khó khăn trong quá trình triển khai hiện tại và tương lai.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/challenges.jpeg" style="width: 800px;"/>

> 🖼️ **[Ảnh placeholder #23 — Image generation challenges]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. A friendly cat police officer wearing a small cap and badge, standing in front of a large kawaii image generation machine (drawn as a colorful TV-like box with a robot arm holding a paintbrush). The cat officer holds up a row of caution signs in their paws: 'Bias', 'Bản quyền', 'Deepfake', 'Chi phí tính toán', 'Nội dung độc hại'. Around the machine, small worried-looking emoji clouds. Soft pastel yellow and orange caution-tape colors mixed with friendly pastel, minimal clean background, gentle warning vibe (not scary)."*

- **Thiên lệch (Bias):** Dữ liệu huấn luyện thường không đa dạng (VD: gương mặt đa số là người da trắng ở nhiều dataset).
Mô hình học theo và có thể tái tạo bias.
- **Nội dung độc hại (Safety):** Các mô hình sinh ảnh có thể tạo ra hình ảnh phản cảm, bạo lực hoặc giả mạo thông tin.
- **Giả mạo thông tin:** Mô hình image generation có thể tạo video, ảnh giả mạo gây hại.
- **Bản quyền:** Các mô hình học từ bộ dữ liệu lớn có thể vi phạm bản quyền ảnh nghệ sĩ.
- **Chi phí và tác động môi trường:** Huấn luyện và sử dụng mô hình image generation lớn tiêu tốn tài nguyên nhiều.
- **Khả năng giải thích:** Để kiểm tra, kiểm soát kết quả của mô hình image generation, cần nghiên cứu về cách giải thích logic, nguồn gốc từng thành phần.
- **Đánh giá chất lượng:** Cần thoát khỏi sự phụ thuộc vào đánh giá của con người, xây dựng các hệ thống tự động đánh giá.
