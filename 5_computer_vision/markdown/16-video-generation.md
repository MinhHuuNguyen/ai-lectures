---
time: 06/26/2022
title: Bài toán video generation
description: Sinh video (video generation) là bước phát triển tự nhiên của sinh ảnh thay vì tạo ra một bức ảnh tĩnh, mô hình phải tạo ra cả một chuỗi khung hình vừa đẹp về thị giác, vừa nhất quán và mượt mà về chuyển động theo thời gian. Bài toán này kế thừa toàn bộ nền tảng của các mô hình tạo sinh (GAN, Transformer, Diffusion) nhưng phải giải quyết thêm thách thức cốt lõi là tính nhất quán thời gian (temporal consistency) cùng chi phí tính toán bùng nổ. Bài viết tập trung vào các nhóm phương pháp sinh video, các thước đo đánh giá và những lưu ý đặc thù khi làm việc với dữ liệu video.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/16-video-generation/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: false
---

## 1. Giới thiệu chung về video generation

Video generation là bài toán xây dựng các mô hình có khả năng **tạo ra video mới** — một chuỗi khung hình liên tiếp — sao cho chân thực, mượt mà và phù hợp với điều kiện đầu vào (nhiễu ngẫu nhiên, văn bản, ảnh, hoặc video khác).
Đây là phần mở rộng của bài toán [image generation](/blog/bai-toan-image-generation): toàn bộ nền tảng về các mô hình tạo sinh (VAE, GAN, Transformer, Diffusion) và các metric cơ bản (IS, FID, CLIP Score) đã được trình bày trong bài đó; bài này tập trung vào những gì **đặc thù cho video**.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/16-video-generation/banner.jpeg" style="width: 1000px;"/>

### Khác biệt cốt lõi so với sinh ảnh

Sinh một video khó hơn sinh một tập ảnh độc lập rất nhiều, vì xuất hiện thêm các ràng buộc:
- **Nhất quán thời gian (temporal consistency):** đối tượng phải giữ nguyên hình dạng, màu sắc, danh tính qua các frame; không được "nhấp nháy" (flickering) hay biến dạng đột ngột.
- **Chuyển động hợp lý (plausible motion):** chuyển động phải tuân theo vật lý và ngữ nghĩa — người đi bộ phải bước đều, nước phải chảy xuống.
- **Chi phí bùng nổ:** không gian dữ liệu lớn hơn ảnh theo thừa số $T$ (số frame), khiến cả huấn luyện lẫn suy luận đắt đỏ hơn nhiều bậc.
- **Mạch lạc dài hạn (long-range coherence):** video dài đòi hỏi nội dung nhất quán xuyên suốt hàng trăm frame.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/16-video-generation/temporal_consistency.jpeg" style="width: 900px;"/>

### Các bài toán con

- **Unconditional video generation:** sinh video từ nhiễu ngẫu nhiên, không điều kiện.
- **Text-to-Video (T2V):** sinh video từ mô tả văn bản (VD: "một chú gấu trúc đang lướt ván trên sóng").
- **Image-to-Video (I2V) / Video Prediction:** từ một (hoặc vài) khung hình đầu, sinh ra phần video tiếp theo — "làm cho ảnh chuyển động".
- **Video-to-Video (V2V):** biến đổi video nguồn (đổi phong cách, chỉnh sửa nội dung, tô màu) nhưng giữ chuyển động gốc.
- **Frame Interpolation:** chèn thêm frame trung gian để tăng FPS / làm slow-motion.
- **Sinh đồng thời hình và tiếng (audio-video joint generation):** tạo video kèm âm thanh khớp nội dung (tiếng bước chân, tiếng nhạc cụ) — hướng mới nổi với các mô hình như Veo 3.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/16-video-generation/applications.jpeg" style="width: 1000px;"/>

## 2. Nhóm các phương pháp giải bài toán video generation

Các nhóm phương pháp sinh video phần lớn **mở rộng từ các họ mô hình sinh ảnh** đã trình bày ở bài [image generation](/blog/bai-toan-image-generation), bổ sung cơ chế mô hình hoá chiều thời gian. Phần dưới chỉ nhấn mạnh điểm khác biệt cho video, không lặp lại nền tảng GAN/diffusion/transformer.

### 2.1. Nhóm dựa trên GAN

#### Mô tả ý tưởng và cơ chế hoạt động

Mở rộng trò chơi đối kháng Generator–Discriminator (xem bài 8) sang chiều thời gian. Hai hướng chính:
- **Sinh khối không-thời gian:** Generator tạo cả khối video bằng tích chập 3D; Discriminator phán đoán thật/giả trên cả clip (**VGAN**, **TGAN**).
- **Tách nội dung và chuyển động:** **MoCoGAN** phân tách latent thành phần *content* (giữ cố định, quyết định "ai/cái gì") và phần *motion* (biến thiên theo thời gian, quyết định "chuyển động thế nào"), giúp điều khiển tốt hơn.

#### Ưu và nhược điểm

**Ưu điểm:** sinh nhanh (một lần feed-forward), kế thừa độ sắc nét của GAN.
**Nhược điểm:** huấn luyện bất ổn và mode collapse (xem bài 8) còn trầm trọng hơn ở video; khó scale lên video dài, độ phân giải cao.

#### Một số mô hình tiêu biểu trong nhóm

- **VGAN (Vondrick et al., 2016)** — [paper](https://arxiv.org/abs/1609.02612) — Generator hai luồng foreground (3D conv) + background (2D conv).
- **TGAN - Temporal GAN (Saito et al., 2017)** — [paper](https://arxiv.org/abs/1611.06624) — tách temporal generator và image generator.
- **MoCoGAN (Tulyakov et al., 2017)** — [paper](https://arxiv.org/abs/1707.04993) — phân tách content/motion trong không gian latent.
- **DVD-GAN (Clark et al., 2019)** — [paper](https://arxiv.org/abs/1907.06571) — scale GAN sinh video lên độ phân giải và độ dài lớn hơn với dual discriminator (không gian + thời gian).

### 2.2. Nhóm Autoregressive / Transformer

#### Mô tả ý tưởng và cơ chế hoạt động

Tương tự sinh ảnh bằng transformer (bài 8): nén video thành **chuỗi token rời rạc** (bằng VQ-VAE/VQ-GAN mở rộng theo thời gian), rồi dùng transformer sinh token tuần tự hoặc song song (masked), cuối cùng decode về pixel.
Lợi thế của hướng này là **sinh video dài** bằng cách nối tiếp (autoregressive rollout) và dễ điều kiện hoá trên văn bản nhờ kiến trúc thống nhất với mô hình ngôn ngữ.

#### Ưu và nhược điểm

**Ưu điểm:** scale tốt, hiểu prompt phức tạp, sinh được video độ dài thay đổi (Phenaki sinh video dài theo chuỗi prompt).
**Nhược điểm:** sinh tuần tự chậm; tokenize làm mất chi tiết; dễ tích luỹ lỗi (error accumulation) khi rollout dài.

#### Một số mô hình tiêu biểu trong nhóm

- **VideoGPT (Yan et al., 2021)** — [paper](https://arxiv.org/abs/2104.10157) — VQ-VAE 3D + transformer autoregressive.
- **NÜWA (Wu et al., 2021)** — [paper](https://arxiv.org/abs/2111.12417) — mô hình đa nhiệm cho ảnh/video từ văn bản hoặc phác thảo.
- **CogVideo (Hong et al., 2022)** — [paper](https://arxiv.org/abs/2205.15868) — transformer text-to-video quy mô lớn (tiếng Anh/Trung).
- **Phenaki (Villegas et al., 2022)** — [paper](https://arxiv.org/abs/2210.02399) — sinh video dài, có cốt truyện từ chuỗi prompt.
- **VideoPoet (Kondratyuk et al., 2023)** — [paper](https://arxiv.org/abs/2312.14125) — LLM đa phương thức sinh video (kèm cả audio).

### 2.3. Nhóm Diffusion-based

#### Mô tả ý tưởng và cơ chế hoạt động

Là hướng mạnh và phổ biến nhất hiện nay, mở rộng Diffusion Model (bài 8) bằng cách thêm **các lớp chú ý theo thời gian (temporal attention)** hoặc **tích chập 3D** vào U-Net khử nhiễu, để mô hình khử nhiễu *đồng thời cả khối video* và duy trì nhất quán thời gian.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/16-video-generation/video_diffusion_unet.jpeg" style="width: 900px;"/>

Để giảm chi phí, phần lớn mô hình thực hiện diffusion trong **không gian latent nén theo cả không gian lẫn thời gian** (latent video diffusion). Một số mô hình (Imagen Video, Make-A-Video) dùng **cascaded diffusion** — sinh video độ phân giải/FPS thấp rồi nâng cấp dần qua các tầng super-resolution không-thời gian.

#### Ưu và nhược điểm

**Ưu điểm:** chất lượng và độ đa dạng vượt trội, huấn luyện ổn định, ít mode collapse (xem bài 8); hệ sinh thái lớn (AnimateDiff tận dụng được mô hình ảnh Stable Diffusion sẵn có).
**Nhược điểm:** sinh chậm (lặp nhiều bước khử nhiễu) — nặng hơn nhiều so với sinh ảnh; tốn tài nguyên huấn luyện khổng lồ.

#### Một số mô hình tiêu biểu trong nhóm

- **Video Diffusion Models (Ho et al., 2022)** — [paper](https://arxiv.org/abs/2204.03458) — mô hình diffusion video đầu tiên, dùng 3D U-Net.
- **Imagen Video (Ho et al., 2022)** — [paper](https://arxiv.org/abs/2210.02303) — cascaded diffusion text-to-video độ phân giải cao.
- **Make-A-Video (Singer et al., 2022)** — [paper](https://arxiv.org/abs/2209.14792) — tận dụng dữ liệu ảnh-văn bản, học chuyển động từ video không nhãn.
- **Stable Video Diffusion (Blattmann et al., 2023)** — [paper](https://arxiv.org/abs/2311.15127) — latent video diffusion mã nguồn mở, image-to-video.
- **AnimateDiff (Guo et al., 2023)** — [paper](https://arxiv.org/abs/2307.04725) — gắn "motion module" vào Stable Diffusion đã pretrain để biến mô hình ảnh thành mô hình video.

### 2.4. Spatiotemporal DiT và các mô hình SOTA

#### Mô tả ý tưởng và cơ chế hoạt động

Hướng mới nhất thay U-Net bằng **Diffusion Transformer (DiT)** hoạt động trên các **patch không-thời gian (spatiotemporal patches)** trong một không gian latent được nén mạnh. Kiến trúc thuần transformer này scale rất tốt theo dữ liệu và tham số, là nền tảng của các mô hình sinh video chất lượng cao gần đây.

<!-- PLACEHOLDER IMAGE: spatiotemporal_dit.jpeg
Prompt: "Original schematic of a spatiotemporal Diffusion Transformer (DiT) for video: a compressed video latent is split into spacetime patches, flattened into a token sequence, processed by transformer (DiT) blocks, then decoded back into video. Clean labeled diagram, white background." (Minh hoạ khái niệm theo Sora technical report / DiT, Peebles & Xie, 2022 / W.A.L.T — vẽ sơ đồ gốc, KHÔNG sao chép hình trong paper.)
Sau khi sinh ảnh, thêm vào đây:
<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/16-video-generation/spatiotemporal_dit.jpeg" style="width: 1000px;"/>
-->

#### Một số mô hình tiêu biểu

- **Sora (OpenAI, 2024)** — diffusion transformer trên spacetime patches, sinh video dài tới ~1 phút, chất lượng đột phá.
- **Veo / Veo 3 (Google DeepMind, 2024–2025)** — text/image-to-video chất lượng cao; Veo 3 **sinh kèm âm thanh** đồng bộ.
- **Kling (Kuaishou, 2024)** — mô hình sinh video độ dài lớn, chuyển động ổn định.
- **Movie Gen (Meta, 2024)** — [paper](https://arxiv.org/abs/2410.13720) — bộ mô hình sinh video kèm âm thanh, chỉnh sửa video bằng văn bản.

## 3. Các metrics trong video generation

Đánh giá video sinh ra phải tính tới cả **chất lượng từng frame**, **tính nhất quán thời gian** và **độ khớp với điều kiện**. Nhiều metric kế thừa từ sinh ảnh (FID, IS, CLIP Score — xem bài 8), bổ sung chiều thời gian.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/16-video-generation/fvd.jpeg" style="width: 900px;"/>

### 3.1. Fréchet Video Distance (FVD)

#### Mô tả ý tưởng và cơ chế hoạt động

**FVD** (Unterthiner et al., 2018) — [paper](https://arxiv.org/abs/1812.01717) — là metric chính cho sinh video, mở rộng trực tiếp từ FID (bài 8). Khác biệt: thay vì trích đặc trưng từng ảnh bằng Inception-v3, FVD trích **đặc trưng không-thời gian của cả clip** bằng mạng **I3D** (pretrain trên Kinetics), rồi tính khoảng cách Fréchet giữa hai phân phối Gaussian của tập video thật và tập video sinh:

$$FVD = \|\mu_r - \mu_g\|^2 + Tr\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

Nhờ dùng đặc trưng I3D, FVD nhạy với cả chất lượng hình ảnh lẫn **tính hợp lý của chuyển động** — điều mà FID per-frame bỏ qua. **FVD càng thấp càng tốt.**

#### Ví dụ

| Tình huống | FID per-frame | FVD |
|---|---|---|
| Từng frame đẹp nhưng chuyển động giật/nhấp nháy | thấp (tưởng tốt) | cao (phát hiện lỗi) |
| Video mượt, chuyển động hợp lý | thấp | thấp |

Đây chính là lý do FVD được ưa chuộng: nó "bắt" được các lỗi thời gian mà metric trên từng frame không thấy.

#### Ưu và nhược điểm

**Ưu điểm:** đánh giá đồng thời chất lượng không gian và thời gian; là chuẩn de-facto cho sinh video.
**Nhược điểm:** phụ thuộc I3D và dữ liệu Kinetics; nhạy với độ dài clip và cách lấy mẫu frame; kế thừa giả định Gaussian của Fréchet (xem hạn chế FID ở bài 8).

### 3.2. Các metric bổ trợ

#### Mô tả ý tưởng và cơ chế hoạt động

- **FID / IS per-frame:** áp FID/IS (bài 8) lên từng frame để đo chất lượng *không gian*, nhưng **không** phản ánh tính nhất quán thời gian — phải dùng kèm FVD.
- **CLIPSIM / CLIP Score:** đo độ khớp giữa video và prompt văn bản bằng cách trung bình CLIP Score (bài 8) trên các frame — quan trọng cho text-to-video.
- **Tính nhất quán thời gian (temporal consistency / warp error):** dùng optical flow để "warp" frame trước sang frame sau và đo sai khác — phát hiện flickering.
- **VBench (Huang et al., 2023)** — [paper](https://arxiv.org/abs/2311.17982) — bộ benchmark **đa chiều**, tách đánh giá thành nhiều khía cạnh (chất lượng hình ảnh, độ mượt chuyển động, nhất quán đối tượng, độ khớp prompt...), tương quan tốt với cảm nhận con người.
- **Human Evaluation:** vẫn là tiêu chuẩn vàng (xem bài 8) — so sánh cặp (2AFC) hoặc Elo rating giữa các mô hình trên các trục chân thực, chuyển động, độ khớp prompt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/16-video-generation/vbench.jpeg" style="width: 700px;"/>

#### Ví dụ

Một mô hình text-to-video thường được báo cáo đồng thời: **FVD** (chất lượng tổng thể) + **CLIPSIM** (khớp prompt) + **VBench** (phân tích đa chiều) + **human eval** (chốt cuối).

#### Ưu và nhược điểm

**Ưu điểm:** kết hợp nhiều metric cho bức tranh toàn diện hơn một con số đơn lẻ.
**Nhược điểm:** chưa có metric đơn nào "chốt" được toàn bộ chất lượng video; các metric tự động vẫn lệch so với cảm nhận con người, đặc biệt về chuyển động.

## 4. Lưu ý khi làm việc với video generation

Ngoài các thách thức chung của Generative AI (bias, an toàn, bản quyền, chi phí — xem mục thách thức bài [image generation](/blog/bai-toan-image-generation)), sinh video có những lưu ý đặc thù:

- **Tính nhất quán thời gian là ưu tiên số một:** flickering và biến dạng đối tượng giữa các frame là lỗi dễ thấy nhất; cần temporal attention/3D conv và đánh giá bằng FVD chứ không chỉ FID per-frame.
- **Chi phí tính toán bùng nổ:** gần như luôn phải làm việc trong **không gian latent nén không-thời gian** thay vì pixel; cân nhắc kỹ độ phân giải × FPS × độ dài vì chúng nhân nhau.
- **Sinh video dài:** mô hình thường chỉ sinh tốt vài giây; video dài cần kỹ thuật **rollout autoregressive** hoặc **chia khối (chunking)** có điều kiện gối nhau, kèm cơ chế chống tích luỹ lỗi.
- **Điều khiển chuyển động và camera:** ngoài nội dung, người dùng cần điều khiển *chuyển động* (motion control) và *góc/đường đi camera* (camera control) — một trục điều khiển không tồn tại ở sinh ảnh.
- **Đồng bộ hình–tiếng (audio-video):** xu hướng mới là sinh đồng thời video và âm thanh khớp nhau (tiếng bước chân, môi khớp lời nói). Đây là bài toán đa phương thức khó, đòi hỏi đồng bộ chặt giữa hai luồng.
- **Dữ liệu huấn luyện và caption video:** dữ liệu video–văn bản chất lượng cao khan hiếm và đắt; chú thích (caption) video tốn kém hơn ảnh, ảnh hưởng trực tiếp tới khả năng điều khiển bằng prompt.
- **An toàn và deepfake:** sinh video chân thực làm gia tăng rủi ro giả mạo (deepfake), đòi hỏi watermark, kiểm duyệt và các biện pháp truy vết nguồn gốc.
