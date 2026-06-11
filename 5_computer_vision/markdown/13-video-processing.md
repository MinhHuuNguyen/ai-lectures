---
time: 06/24/2022
title: Bài toán xử lý video
description: Video là dữ liệu giàu thông tin bậc nhất trong thị giác máy tính — một video không chỉ là tập hợp các khung hình tĩnh mà còn mang theo chiều thời gian (chuyển động, sự kiện, quan hệ nhân quả) và thường đi kèm kênh âm thanh. Làm việc với dữ liệu video đòi hỏi những kỹ thuật riêng để biểu diễn chuyển động, mô hình hoá quan hệ không-thời gian, kết hợp hình ảnh với âm thanh, đồng thời phải đối mặt với chi phí tính toán và lưu trữ khổng lồ. Bài viết này trình bày cách biểu diễn và tiền xử lý video, các nhóm mô hình hoá không-thời gian, vai trò của âm thanh trong hiểu video, các thước đo đánh giá và những thách thức đặc thù của lĩnh vực.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: false
---

## 1. Giới thiệu chung về xử lý video

Xử lý video (video processing) là nhánh của thị giác máy tính tập trung vào việc phân tích và hiểu **dữ liệu video** — chuỗi các khung hình (frames) liên tiếp theo thời gian, thường đi kèm một kênh âm thanh.
Nếu một bức ảnh được biểu diễn bằng tensor 3 chiều $H \times W \times C$ (cao × rộng × kênh màu), thì một video được biểu diễn bằng tensor 4 chiều $T \times H \times W \times C$, trong đó $T$ là số khung hình theo trục thời gian.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/banner.jpeg" style="width: 1000px;"/>

Điểm khác biệt cốt lõi giữa video và ảnh tĩnh chính là **chiều thời gian (temporal dimension)**.
Chính chiều thời gian này tạo ra cả cơ hội lẫn thách thức:
- **Chuyển động và sự kiện:** Nhiều bài toán chỉ có thể giải được khi nhìn vào chuỗi thời gian. Ví dụ, chỉ với một khung hình ta khó phân biệt "đang mở cửa" hay "đang đóng cửa", "ngồi xuống" hay "đứng lên" — thông tin nằm ở **sự thay đổi giữa các khung hình**.
- **Quan hệ nhân quả và ngữ cảnh thời gian:** Một hành động dài (nấu ăn, chơi thể thao) gồm nhiều giai đoạn nối tiếp nhau, đòi hỏi mô hình nắm được phụ thuộc thời gian dài (long-range temporal dependency).
- **Dư thừa thông tin (redundancy):** Hai khung hình liên tiếp thường gần như giống hệt nhau, gây lãng phí tính toán nếu xử lý từng frame độc lập.
- **Chi phí tính toán và lưu trữ:** Một video chỉ vài giây ở 30 FPS đã có hàng trăm khung hình. Khối lượng dữ liệu và phép tính lớn hơn ảnh tĩnh nhiều bậc.

Một số khái niệm nền tảng khi làm việc với video:
- **Frame:** một khung hình tĩnh trong video.
- **FPS (Frames Per Second):** số khung hình mỗi giây (phổ biến 24, 25, 30, 60).
- **Độ phân giải (resolution):** kích thước không gian mỗi frame (VD: $1920 \times 1080$).
- **Codec / nén (compression):** chuẩn nén video (H.264, H.265, VP9...) lưu trữ video hiệu quả bằng cách chỉ ghi keyframe và phần sai khác (motion vector + residual) giữa các frame.
- **Keyframe (I-frame):** khung hình được mã hoá độc lập, không phụ thuộc frame khác.
- **Shot / Scene / Clip:** đơn vị ngữ nghĩa — *shot* là đoạn quay liên tục từ một camera, *scene* gồm nhiều shot cùng bối cảnh, *clip* là đoạn ngắn được cắt ra để đưa vào mô hình.

### Ứng dụng của xử lý video

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/applications.jpeg" style="width: 1000px;"/>

Xử lý video có mặt trong rất nhiều lĩnh vực thực tiễn:
- **Giám sát an ninh (surveillance):** phát hiện hành vi bất thường, đếm người, theo dõi đối tượng trong camera giám sát.
- **Xe tự hành (autonomous driving):** nhận diện và dự đoán chuyển động của người đi bộ, phương tiện qua chuỗi khung hình.
- **Phân tích thể thao:** theo dõi cầu thủ và bóng, thống kê chiến thuật, tự động tạo highlight.
- **Kiểm duyệt nội dung (content moderation):** phát hiện nội dung bạo lực, phản cảm, deepfake trên các nền tảng video.
- **Truy xuất và tìm kiếm video (video retrieval):** tìm đoạn video theo nội dung hoặc theo mô tả văn bản.
- **Y tế:** phân tích video nội soi, siêu âm tim, theo dõi vận động phục hồi chức năng.
- **AR/VR và robot:** hiểu cảnh động theo thời gian thực để tương tác với môi trường.

### Các bài toán con khi làm việc với video

Tương tự như image generation có nhiều bài toán con (synthesis, translation, inpainting...), làm việc với dữ liệu video cũng bao gồm một họ các bài toán:
- **Nhận diện hành động (Action Recognition) / Video Classification:** gán nhãn hành động/sự kiện cho cả đoạn video (VD: "nhảy dây", "rót nước").
- **Định vị hành động theo thời gian (Temporal Action Localization):** xác định hành động *xảy ra ở khoảng thời gian nào* trong video dài.
- **Phát hiện đối tượng theo video (Video Object Detection):** phát hiện đối tượng trên từng frame nhưng tận dụng tính liên tục thời gian để ổn định kết quả.
- **Theo dõi đối tượng (Object Tracking / Multi-Object Tracking - MOT):** gán và duy trì danh tính (ID) của các đối tượng xuyên suốt video.
- **Phân vùng đối tượng trong video (Video Object Segmentation - VOS):** phân vùng pixel của đối tượng và duy trì nhất quán qua các frame.
- **Mô tả video (Video Captioning) / Hỏi đáp video (Video QA):** sinh mô tả ngôn ngữ cho video (xem thêm bài [image captioning](/blog/bai-toan-image-captioning)).
- **Hiểu đa phương thức hình-tiếng (Audio-Visual Understanding):** kết hợp khung hình và âm thanh để hiểu nội dung.

Bài toán **sinh video (video generation)** — tạo ra video mới từ nhiễu, văn bản hoặc ảnh — được trình bày riêng trong bài [video generation](/blog/bai-toan-video-generation) vì nó thuộc nhóm mô hình tạo sinh (generative).

## 2. Biểu diễn và tiền xử lý dữ liệu video

Trước khi đưa vào mô hình, video cần được biểu diễn ở dạng phù hợp. Đây là bước đặc thù nhất khi làm việc với dữ liệu video, vì ta phải cân bằng giữa lượng thông tin thời gian giữ lại và chi phí tính toán.

### 2.1. Lấy mẫu khung hình (Frame Sampling)

Một video dài có thể chứa hàng nghìn khung hình, nhưng đưa toàn bộ vào mô hình là bất khả thi về bộ nhớ. Vì vậy ta phải **lấy mẫu** một tập con khung hình.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/frame_sampling.jpeg" style="width: 900px;"/>

- **Dense sampling (lấy mẫu dày):** lấy một clip ngắn liên tiếp (VD: 16–64 frame liên tục). Giữ được chuyển động chi tiết (fine-grained motion) nhưng chỉ "nhìn" được một khoảng thời gian ngắn.
- **Sparse / uniform sampling (lấy mẫu thưa):** chia video thành các đoạn đều nhau và lấy một frame đại diện mỗi đoạn, bao phủ toàn bộ độ dài video với ít frame.
- **Segment-based sampling (TSN):** chiến lược của **Temporal Segment Networks** — chia video thành $K$ đoạn, lấy ngẫu nhiên một snippet mỗi đoạn rồi tổng hợp dự đoán. Vừa bao phủ toàn cục, vừa rẻ.

Việc chọn chiến lược lấy mẫu phụ thuộc vào bài toán: hành động ngắn, nhanh (đập tay) cần dense sampling; sự kiện dài (nấu ăn) cần sparse sampling để bao phủ toàn bộ ngữ cảnh.

### 2.2. Optical Flow — biểu diễn chuyển động

**Optical Flow (luồng quang học)** là trường vector mô tả **chuyển động biểu kiến** của từng pixel giữa hai khung hình liên tiếp. Mỗi pixel được gán một vector $(u, v)$ chỉ hướng và độ lớn dịch chuyển. Đây là cách biểu diễn chuyển động tường minh, tách biệt thông tin "động" khỏi thông tin "tĩnh" (màu sắc, kết cấu).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/optical_flow.jpeg" style="width: 800px;"/>

Cơ sở lý thuyết là giả định **độ sáng không đổi (brightness constancy)**: cùng một điểm vật thể giữ nguyên cường độ sáng khi di chuyển một lượng nhỏ giữa hai frame liên tiếp:

$$I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)$$

Khai triển Taylor bậc nhất dẫn tới **phương trình ràng buộc optical flow**:

$$I_x u + I_y v + I_t = 0$$

trong đó $I_x, I_y, I_t$ là đạo hàm riêng của cường độ ảnh theo $x, y, t$ và $(u, v) = (\frac{dx}{dt}, \frac{dy}{dt})$ là vector chuyển động cần tìm.
Một phương trình với hai ẩn $(u, v)$ là **bài toán thiếu xác định (aperture problem)**, nên cần thêm giả định để giải:
- **Phương pháp cổ điển:** **Lucas-Kanade** (giả định flow đồng nhất trong một lân cận nhỏ) cho flow thưa; **Horn-Schunck** và **Farnebäck** cho flow dày (dense).
- **Phương pháp học sâu:** **FlowNet** (Dosovitskiy et al., 2015) — [paper](https://arxiv.org/abs/1504.06852) — học optical flow end-to-end bằng CNN; **RAFT** (Teed & Deng, 2020) — [paper](https://arxiv.org/abs/2003.12039) — kiến trúc recurrent với all-pairs correlation, đạt độ chính xác SOTA.

Optical flow là đầu vào quan trọng cho nhiều phương pháp nhận diện hành động (đặc biệt nhánh temporal của two-stream network ở §3.2). Sai số của optical flow được đo bằng **EPE (End-Point Error)** — trình bày ở §5.5.

### 2.3. Khai thác thông tin nén (Compressed Video)

Một hướng tiền xử lý hiệu quả là **tận dụng chính cấu trúc của video nén** thay vì giải nén toàn bộ về RGB. Các codec hiện đại đã lưu sẵn **motion vectors** (xấp xỉ chuyển động) và **residuals** (phần sai khác) giữa các frame.

Các phương pháp như **CoViAR - Compressed Video Action Recognition** (Wu et al., 2018) — [paper](https://arxiv.org/abs/1712.00636) — đọc trực tiếp I-frame, motion vector và residual từ luồng nén, tránh phải giải mã và tính optical flow đắt đỏ. Ưu điểm là nhanh hơn nhiều lần; nhược điểm là motion vector từ codec thô và nhiễu hơn optical flow chuyên dụng.

## 3. Nhóm các phương pháp mô hình hoá video

Câu hỏi trung tâm của mọi mô hình video là: **làm sao mô hình hoá được chiều thời gian?** Dưới đây là các nhóm phương pháp chính, sắp xếp theo tiến trình phát triển.

### 3.1. 2D CNN + tổng hợp thời gian

#### Mô tả ý tưởng và cơ chế hoạt động

Ý tưởng đơn giản nhất là tái sử dụng các CNN 2D mạnh mẽ đã thành công với ảnh: trích đặc trưng từng frame bằng CNN 2D rồi **tổng hợp (fusion)** thông tin theo thời gian. Karpathy et al. (2014) — [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Karpathy_Large-scale_Video_Classification_2014_CVPR_paper.pdf) — khảo sát nhiều chiến lược:
- **Single-frame:** chỉ phân loại trên một frame, bỏ qua thời gian (baseline).
- **Late fusion:** xử lý độc lập hai frame cách xa nhau rồi hợp đặc trưng ở lớp cuối.
- **Early fusion:** gộp nhiều frame ngay ở lớp đầu (theo kênh).
- **Slow fusion:** trộn thông tin thời gian từ từ qua nhiều lớp — hiệu quả nhất trong nhóm.

Một bước tiến quan trọng là **Temporal Segment Networks - TSN** (Wang et al., 2016) — [paper](https://arxiv.org/abs/1608.00859): chia video thành $K$ đoạn, mỗi đoạn lấy một snippet, cho qua cùng một CNN 2D (chia sẻ trọng số), rồi tổng hợp bằng hàm consensus (trung bình) để ra dự đoán cấp video. Nhờ đó TSN mô hình hoá được toàn bộ độ dài video với chi phí rất thấp.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Rẻ và tận dụng pretrain ảnh:** dùng lại trực tiếp backbone CNN 2D (ResNet...) đã pretrain trên ImageNet.
- **Hiệu quả cho hành động "tĩnh về ngữ cảnh":** với hành động mà bối cảnh đã đủ gợi ý (chơi đàn piano → có cây đàn), single-frame/TSN đã rất mạnh.

**Nhược điểm:**
- **Mô hình hoá chuyển động yếu:** tổng hợp đơn giản (trung bình) khó nắm được thứ tự thời gian và chuyển động chi tiết.
- **Kém với hành động phụ thuộc thứ tự:** không phân biệt tốt "mở" vs "đóng", "đẩy" vs "kéo".

#### Một số mô hình tiêu biểu trong nhóm

- **Karpathy Fusion Networks (2014)** — [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Karpathy_Large-scale_Video_Classification_2014_CVPR_paper.pdf) — khảo sát single/early/late/slow fusion trên Sports-1M.
- **TSN - Temporal Segment Networks (Wang et al., 2016)** — [paper](https://arxiv.org/abs/1608.00859) — sparse segment sampling + consensus, nền tảng cho nhiều mô hình sau.
- **TSM - Temporal Shift Module (Lin et al., 2019)** — [paper](https://arxiv.org/abs/1811.08383) — "dịch" một phần kênh đặc trưng dọc trục thời gian, đạt khả năng mô hình hoá thời gian gần 3D CNN với chi phí của 2D CNN.

### 3.2. Two-Stream Networks

#### Mô tả ý tưởng và cơ chế hoạt động

**Two-Stream Network** (Simonyan & Zisserman, 2014) — [paper](https://arxiv.org/abs/1406.2199) — xuất phát từ quan sát thị giác con người: nhận thức gồm luồng *what* (nội dung) và *where/motion* (chuyển động). Mô hình gồm hai nhánh CNN song song:
- **Nhánh không gian (Spatial stream):** nhận **một frame RGB**, học diện mạo (appearance) — vật thể, bối cảnh.
- **Nhánh thời gian (Temporal stream):** nhận **stack optical flow** của nhiều frame liên tiếp, học chuyển động.

Hai nhánh được huấn luyện riêng và **hợp nhất (fusion)** điểm số ở cuối (trung bình hoặc SVM). Việc tách riêng nhánh chuyển động bằng optical flow giúp mô hình nắm bắt động học tốt hơn hẳn nhóm 2D CNN thuần.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/two_stream.jpeg" style="width: 900px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **Mô hình hoá chuyển động mạnh:** optical flow cung cấp tín hiệu động học tường minh, chính xác.
- **Cải thiện rõ rệt so với 2D CNN thuần** trên các benchmark như UCF101, HMDB51.

**Nhược điểm:**
- **Chi phí tính optical flow lớn:** phải tính flow trước, tốn thời gian và lưu trữ, khó chạy thời gian thực.
- **Hai nhánh tách rời:** không học chung end-to-end một cách tự nhiên; fusion thủ công.

#### Một số mô hình tiêu biểu trong nhóm

- **Two-Stream ConvNet (Simonyan & Zisserman, 2014)** — [paper](https://arxiv.org/abs/1406.2199) — mô hình gốc.
- **Two-Stream Fusion (Feichtenhofer et al., 2016)** — [paper](https://arxiv.org/abs/1604.06573) — nghiên cứu cách và vị trí hợp nhất hai nhánh (spatial + temporal).
- **TSN two-stream (Wang et al., 2016)** — kết hợp ý tưởng two-stream với segment sampling.

### 3.3. 3D CNNs

#### Mô tả ý tưởng và cơ chế hoạt động

Thay vì xử lý thời gian như bước hậu kỳ, **3D CNN** mở rộng phép tích chập sang chiều thời gian: kernel trượt đồng thời trên cả không gian và thời gian $(T \times H \times W)$, học đặc trưng không-thời gian (spatio-temporal) một cách thống nhất ngay từ pixel.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/cnn_3d.jpeg" style="width: 800px;"/>

Với đầu vào là khối video $X \in \mathbb{R}^{T \times H \times W \times C}$, một tầng tích chập 3D với kernel $k_t \times k_h \times k_w$ tính:

$$Y(t, x, y) = \sum_{i=0}^{k_t-1} \sum_{j=0}^{k_h-1} \sum_{l=0}^{k_w-1} W(i, j, l) \cdot X(t+i, x+j, y+l)$$

Khác biệt then chốt so với conv 2D là chiều cộng theo $i$ (thời gian): mỗi neuron đầu ra "nhìn" được một cửa sổ thời gian, nên học được chuyển động trực tiếp.
Nhược điểm là số tham số và chi phí tính toán tăng mạnh (nhân thêm thừa số $k_t$). Để giảm tải, **R(2+1)D** tách conv 3D thành conv không gian 2D nối tiếp conv thời gian 1D.

Một mẹo huấn luyện quan trọng là **inflation (I3D)**: khởi tạo kernel 3D bằng cách "bơm phồng" trọng số 2D đã pretrain trên ImageNet (lặp lại theo trục thời gian), tận dụng được pretrain ảnh cho mô hình video.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Học không-thời gian thống nhất, end-to-end:** không cần tính optical flow riêng.
- **Chất lượng cao:** I3D, SlowFast, X3D nằm trong nhóm SOTA về nhận diện hành động.

**Nhược điểm:**
- **Tốn tài nguyên:** nhiều tham số, ngốn bộ nhớ và FLOPs, khó train.
- **Cần dữ liệu lớn:** dễ overfit nếu thiếu dữ liệu, nên thường phải pretrain trên Kinetics.

#### Một số mô hình tiêu biểu trong nhóm

- **C3D (Tran et al., 2015)** — [paper](https://arxiv.org/abs/1412.0767) — 3D CNN tổng quát đầu tiên với kernel $3\times3\times3$, đặt nền móng cho nhóm.
- **I3D - Inflated 3D ConvNet (Carreira & Zisserman, 2017)** — [paper](https://arxiv.org/abs/1705.07750) — inflation từ Inception 2D, giới thiệu bộ dữ liệu Kinetics.
- **R(2+1)D (Tran et al., 2018)** — [paper](https://arxiv.org/abs/1711.11248) — tách conv không gian và thời gian để tăng hiệu quả và độ chính xác.
- **SlowFast (Feichtenhofer et al., 2019)** — [paper](https://arxiv.org/abs/1812.03982) — hai nhánh: nhánh *Slow* (ít frame, nhiều kênh, bắt ngữ nghĩa) và nhánh *Fast* (nhiều frame, ít kênh, bắt chuyển động).
- **X3D (Feichtenhofer, 2020)** — [paper](https://arxiv.org/abs/2004.04730) — mở rộng mạng có hệ thống theo nhiều chiều (độ sâu, thời gian, độ phân giải) để đạt hiệu quả tham số cao.

### 3.4. RNN / LSTM cho video

#### Mô tả ý tưởng và cơ chế hoạt động

Vì video là dữ liệu tuần tự, một hướng tự nhiên là dùng mạng hồi quy (RNN/LSTM) để mô hình hoá phụ thuộc thời gian. Kiến trúc phổ biến là **CNN + LSTM**: CNN 2D trích đặc trưng từng frame, chuỗi đặc trưng này được đưa qua LSTM để tổng hợp thông tin theo thời gian và đưa ra dự đoán.

**LRCN - Long-term Recurrent Convolutional Networks** (Donahue et al., 2015) — [paper](https://arxiv.org/abs/1411.4389) — là đại diện tiêu biểu, áp dụng cho cả nhận diện hành động và mô tả video.
Một biến thể quan trọng là **ConvLSTM** (Shi et al., 2015) — [paper](https://arxiv.org/abs/1506.04214) — thay phép nhân ma trận trong LSTM bằng phép tích chập, giữ được cấu trúc không gian 2D của đặc trưng (hữu ích cho dự báo không-thời gian như dự báo mưa từ radar).

#### Ưu và nhược điểm

**Ưu điểm:**
- **Xử lý chuỗi độ dài thay đổi:** RNN linh hoạt với video dài ngắn khác nhau.
- **Mô hình hoá thứ tự tường minh:** nắm được trình tự thời gian của các sự kiện.

**Nhược điểm:**
- **Khó song song hoá và khó train:** RNN tuần tự, dễ gặp vanishing gradient với chuỗi dài.
- **Bị 3D CNN và Transformer vượt qua:** trên các benchmark hành động hiện đại, nhóm này không còn là SOTA.

#### Một số mô hình tiêu biểu trong nhóm

- **LRCN (Donahue et al., 2015)** — [paper](https://arxiv.org/abs/1411.4389) — CNN + LSTM cho nhận diện và mô tả video.
- **Beyond Short Snippets (Ng et al., 2015)** — [paper](https://arxiv.org/abs/1503.08909) — khảo sát feature pooling và LSTM cho video dài.
- **ConvLSTM (Shi et al., 2015)** — [paper](https://arxiv.org/abs/1506.04214) — LSTM tích chập cho dữ liệu không-thời gian.

### 3.5. Video Transformers

#### Mô tả ý tưởng và cơ chế hoạt động

Sau thành công của **Vision Transformer (ViT)** trên ảnh, ý tưởng coi video như một chuỗi **token không-thời gian** đã mở ra nhóm mô hình mạnh nhất hiện nay. Video được chia thành các *patch* không-thời gian (spatio-temporal patches/tubelets), nhúng thành token và đưa qua các tầng self-attention.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/video_transformer.jpeg" style="width: 900px;"/>

Thách thức lớn nhất là chi phí self-attention: với $N$ token, độ phức tạp là $O(N^2)$, mà $N \approx T \times \frac{H}{p} \times \frac{W}{p}$ rất lớn với video. Các mô hình giải quyết bằng cách **tách attention (factorized attention)**:
- **TimeSformer** (Bertasius et al., 2021) — [paper](https://arxiv.org/abs/2102.05095) — đề xuất **divided space-time attention**: thực hiện attention theo không gian và theo thời gian *tách rời*, giảm chi phí từ $O((T \cdot S)^2)$ xuống $O(T^2 + S^2)$ (với $S$ là số token không gian).
- **ViViT** (Arnab et al., 2021) — [paper](https://arxiv.org/abs/2103.15691) — khảo sát nhiều cách factorize encoder không gian/thời gian.

Một hướng quan trọng khác là **học tự giám sát (self-supervised)** để giảm phụ thuộc vào nhãn: **VideoMAE** che ngẫu nhiên phần lớn token và học tái tạo, cho phép pretrain hiệu quả trên video không nhãn.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Mô hình hoá phụ thuộc thời gian dài:** self-attention kết nối mọi cặp token, nắm được quan hệ thời gian xa.
- **Khả năng scale tốt:** mạnh dần khi tăng dữ liệu và tham số, giống quy luật scaling của các mô hình lớn.
- **Thống nhất với đa phương thức:** dễ kết hợp text, audio vào cùng backbone Transformer.

**Nhược điểm:**
- **Chi phí $O(N^2)$ rất cao:** dù đã factorize, video transformer vẫn ngốn tài nguyên.
- **Đói dữ liệu:** cần pretrain quy mô lớn (hoặc self-supervised) mới phát huy hết sức mạnh.

#### Một số mô hình tiêu biểu trong nhóm

- **TimeSformer (Bertasius et al., 2021)** — [paper](https://arxiv.org/abs/2102.05095) — divided space-time attention.
- **ViViT - Video Vision Transformer (Arnab et al., 2021)** — [paper](https://arxiv.org/abs/2103.15691) — các biến thể factorized transformer cho video.
- **MViT - Multiscale Vision Transformers (Fan et al., 2021)** — [paper](https://arxiv.org/abs/2104.11227) — transformer đa tỉ lệ, hiệu quả cho video.
- **Video Swin Transformer (Liu et al., 2021)** — [paper](https://arxiv.org/abs/2106.13230) — attention trong cửa sổ không-thời gian cục bộ.
- **VideoMAE (Tong et al., 2022)** — [paper](https://arxiv.org/abs/2203.12602) — masked autoencoder cho video, pretrain tự giám sát hiệu quả dữ liệu.

## 4. Âm thanh trong video (Audio & Audio-Visual)

Một video hoàn chỉnh không chỉ có hình ảnh mà còn có **âm thanh** — một phương thức (modality) giàu thông tin thường bị bỏ qua. Tiếng vỗ tay, tiếng nhạc cụ, lời nói, tiếng động cơ... cung cấp tín hiệu bổ sung (và đôi khi quyết định) để hiểu nội dung video. Mục này trình bày cách làm việc với kênh âm thanh và kết hợp hình-tiếng.

### 4.1. Biểu diễn âm thanh

Âm thanh gốc là **dạng sóng (waveform)** — chuỗi 1D các giá trị biên độ theo thời gian, lấy mẫu ở tần số cao (VD: 16 kHz, 44.1 kHz). Xử lý trực tiếp waveform khó vì chuỗi rất dài và thông tin tần số bị ẩn. Vì vậy, cách phổ biến là chuyển sang miền thời gian-tần số bằng **biến đổi Fourier thời gian ngắn (STFT)** để được **spectrogram**.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/audio_spectrogram.jpeg" style="width: 900px;"/>

- **Spectrogram:** ảnh 2D với trục hoành là thời gian, trục tung là tần số, độ sáng là năng lượng — biến âm thanh thành "ảnh" để dùng được CNN.
- **Mel-spectrogram:** spectrogram được ánh xạ sang **thang Mel** (thang tần số mô phỏng cảm nhận của tai người, nhạy hơn ở tần số thấp). Đây là biểu diễn được dùng nhiều nhất.
- **MFCC (Mel-Frequency Cepstral Coefficients):** đặc trưng cô đọng cổ điển, phổ biến trong nhận dạng tiếng nói trước kỷ nguyên deep learning.

Chính nhờ biểu diễn spectrogram dạng ảnh mà toàn bộ kho công cụ CNN/Transformer của thị giác máy tính có thể tái sử dụng cho âm thanh.

### 4.2. Mô hình hoá âm thanh

- **1D CNN trên waveform:** tích chập trực tiếp trên dạng sóng (VD: SoundNet, Wav2Vec) — học đặc trưng từ tín hiệu thô.
- **2D CNN trên spectrogram:** coi mel-spectrogram như ảnh và áp dụng CNN. **VGGish** (Hershey et al., 2017) — [paper](https://arxiv.org/abs/1609.09430) — và **PANNs** (Kong et al., 2019) — [paper](https://arxiv.org/abs/1912.10211) — là các backbone âm thanh phổ biến, pretrain trên **AudioSet**.
- **Audio Transformer:** **AST - Audio Spectrogram Transformer** (Gong et al., 2021) — [paper](https://arxiv.org/abs/2104.01778) — áp dụng ViT trực tiếp lên patch của spectrogram, đạt SOTA phân loại âm thanh.

### 4.3. Các bài toán audio-visual

Khi kết hợp cả hai phương thức, ta có một họ bài toán hình-tiếng:
- **Nhận diện sự kiện/hành động audio-visual:** dùng cả hình và tiếng để phân loại (VD: phân biệt "cắt giấy" và "xé giấy" nhờ âm thanh). Bộ dữ liệu **AudioSet** và **AVE (Audio-Visual Event)** là chuẩn phổ biến.
- **Định vị nguồn âm thanh (Sound Source Localization):** xác định *vùng nào trong khung hình* phát ra âm thanh đang nghe.
- **Đọc môi / nhận dạng tiếng nói nghe-nhìn (Lip Reading / Audio-Visual Speech Recognition):** kết hợp chuyển động môi với âm thanh để tăng độ chính xác, đặc biệt khi ồn.
- **Phát hiện người đang nói (Active Speaker Detection):** xác định ai trong khung hình đang nói tại mỗi thời điểm.
- **Phân vùng audio-visual (Audio-Visual Segmentation):** phân vùng pixel của đối tượng đang phát ra âm thanh.

### 4.4. Hợp nhất đa phương thức (Multimodal Fusion)

Câu hỏi cốt lõi là **kết hợp hình và tiếng ở đâu và như thế nào**:
- **Early fusion:** ghép đặc trưng hình-tiếng từ sớm rồi xử lý chung — nắm được tương tác chi tiết nhưng nhạy với lệch đồng bộ.
- **Late fusion:** xử lý hai nhánh độc lập, chỉ hợp nhất điểm số ở cuối — đơn giản, ổn định.
- **Cross-attention fusion:** dùng attention cho một phương thức "truy vấn" phương thức kia, là cách kết hợp linh hoạt và mạnh nhất hiện nay.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/audio_visual_fusion.jpeg" style="width: 900px;"/>

Một ý tưởng nền tảng là **tương ứng nghe-nhìn tự giám sát (audio-visual correspondence)**: hình và tiếng trong cùng một video vốn đã *đồng bộ tự nhiên*, nên có thể dùng chính sự đồng bộ này làm tín hiệu giám sát miễn phí. **L3-Net "Look, Listen and Learn"** (Arandjelović & Zisserman, 2017) — [paper](https://arxiv.org/abs/1705.08168) — học bằng cách dự đoán một đoạn hình và một đoạn tiếng có khớp nhau hay không. **AudioCLIP** (Guzhov et al., 2021) — [paper](https://arxiv.org/abs/2106.13043) — mở rộng CLIP sang cả âm thanh, tạo không gian nhúng chung cho ảnh, văn bản và âm thanh.

## 5. Các metrics trong xử lý video

Mỗi bài toán video có thước đo riêng. Dưới đây là các metric quan trọng nhất, nhóm theo bài toán.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/metrics.jpeg" style="width: 1000px;"/>

### 5.1. Nhận diện hành động (Action Recognition)

#### Mô tả ý tưởng và cơ chế hoạt động

Action recognition về bản chất là bài toán phân loại, nên dùng các metric phân loại:
- **Top-1 Accuracy:** tỉ lệ video mà lớp dự đoán có xác suất cao nhất *đúng* với nhãn.
- **Top-5 Accuracy:** tỉ lệ video mà nhãn đúng nằm trong 5 lớp dự đoán xác suất cao nhất — hữu ích khi có nhiều lớp dễ nhầm (Kinetics có 400–700 lớp).
- **mAP (mean Average Precision):** dùng cho dữ liệu **đa nhãn** (một video có nhiều hành động cùng lúc), như bộ **Charades**.

#### Ví dụ

Giá trị Top-1 accuracy điển hình của một số mô hình:

| Mô hình / Dataset | Top-1 Accuracy |
|---|---|
| TSN (UCF101) | ~94% |
| I3D (Kinetics-400) | ~71% |
| SlowFast (Kinetics-400) | ~79% |
| VideoMAE (Kinetics-400) | ~87% |
| Something-Something V2 (rất phụ thuộc thời gian) | ~60–75% |

Lưu ý: UCF101 dễ vì nhiều hành động đoán được qua bối cảnh; Something-Something khó vì bắt buộc phải hiểu chuyển động và thứ tự thời gian.

#### Ưu và nhược điểm

**Ưu điểm:** đơn giản, trực quan, dễ so sánh giữa các mô hình.
**Nhược điểm:** accuracy đơn lẻ không phản ánh được mô hình "hiểu chuyển động" hay chỉ "đoán theo bối cảnh"; với dữ liệu mất cân bằng lớp, accuracy có thể gây hiểu lầm (nên dùng kèm mAP/per-class).

### 5.2. Định vị hành động theo thời gian (Temporal Action Localization)

#### Mô tả ý tưởng và cơ chế hoạt động

Bài toán này không chỉ phân loại mà còn phải xác định **khoảng thời gian** $[t_{start}, t_{end}]$ của hành động. Thước đo là **mAP@tIoU**, tương tự mAP trong object detection nhưng IoU được tính trên **trục thời gian 1D**:

$$tIoU = \frac{|\text{khoảng dự đoán} \cap \text{khoảng thật}|}{|\text{khoảng dự đoán} \cup \text{khoảng thật}|}$$

Một dự đoán được tính là đúng nếu $tIoU$ với ground-truth vượt một ngưỡng (VD: 0.5). mAP thường được báo cáo trung bình trên nhiều ngưỡng tIoU (VD: 0.3, 0.4, 0.5, 0.6, 0.7).

#### Ví dụ

| Dataset | Ngưỡng tIoU | mAP điển hình |
|---|---|---|
| THUMOS14 | @0.5 | ~50–70% |
| ActivityNet-1.3 | trung bình @[0.5:0.95] | ~35–40% |

#### Ưu và nhược điểm

**Ưu điểm:** đánh giá đồng thời cả phân loại lẫn độ chính xác định vị thời gian.
**Nhược điểm:** ranh giới hành động trong thực tế thường mơ hồ (khó gán nhãn chính xác $t_{start}/t_{end}$), khiến điểm số nhạy với cách annotate.

### 5.3. Theo dõi đa đối tượng (Multi-Object Tracking - MOT)

#### Mô tả ý tưởng và cơ chế hoạt động

MOT vừa phải phát hiện đúng đối tượng vừa phải **duy trì danh tính (ID)** xuyên suốt video. Các metric chính:
- **MOTA (Multi-Object Tracking Accuracy):** tổng hợp lỗi bỏ sót (FN), báo nhầm (FP) và **chuyển ID (ID switch)**:

$$MOTA = 1 - \frac{\sum_t (FN_t + FP_t + IDSW_t)}{\sum_t GT_t}$$

- **MOTP (Multi-Object Tracking Precision):** độ chính xác định vị (trung bình IoU của các cặp khớp đúng).
- **IDF1:** F1-score trên việc gán đúng danh tính — nhạy với tính nhất quán ID hơn MOTA.
- **HOTA (Higher Order Tracking Accuracy)** (Luiten et al., 2020) — [paper](https://arxiv.org/abs/2009.07736) — cân bằng tường minh giữa chất lượng *phát hiện* (DetA) và chất lượng *liên kết* danh tính (AssA): $HOTA = \sqrt{DetA \cdot AssA}$. Hiện được xem là metric toàn diện nhất.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/mot_metrics.jpeg" style="width: 900px;"/>

#### Ví dụ

| Mô hình / Benchmark | MOTA | IDF1 | HOTA |
|---|---|---|---|
| SORT (MOT17) | ~43 | ~40 | — |
| ByteTrack (MOT17) | ~80 | ~77 | ~63 |
| Ảnh thật (lý tưởng) | 100 | 100 | 100 |

#### Ưu và nhược điểm

**Ưu điểm:** HOTA/IDF1 phản ánh tốt cả phát hiện lẫn duy trì danh tính.
**Nhược điểm:** MOTA bị chi phối bởi số lượng phát hiện (lấn át lỗi ID switch); một bộ metric đơn lẻ khó phản ánh đủ — thường phải báo cáo nhiều chỉ số cùng nhau.

### 5.4. Phân vùng đối tượng trong video (Video Object Segmentation)

#### Mô tả ý tưởng và cơ chế hoạt động

VOS được đánh giá bằng chỉ số **J&F**, trung bình của hai thành phần (chuẩn của benchmark **DAVIS**):
- **Region Similarity $\mathcal{J}$:** chỉ số Jaccard (IoU) giữa mask dự đoán và mask thật — đo độ phủ vùng.
- **Contour Accuracy $\mathcal{F}$:** F-measure trên đường biên — đo độ chính xác của viền đối tượng.

$$\mathcal{J\&F} = \frac{\mathcal{J} + \mathcal{F}}{2}$$

#### Ví dụ

| Mô hình / Benchmark | J&F |
|---|---|
| OSVOS (DAVIS 2017) | ~60 |
| STM (DAVIS 2017) | ~82 |
| XMem (DAVIS 2017) | ~86 |

#### Ưu và nhược điểm

**Ưu điểm:** kết hợp cả độ phủ vùng ($\mathcal{J}$) lẫn chất lượng biên ($\mathcal{F}$), tương quan tốt với cảm nhận.
**Nhược điểm:** không trực tiếp đo tính **nhất quán thời gian** (mask có thể "nhấp nháy" giữa các frame mà vẫn cho J&F cao trên từng frame).

### 5.5. Optical Flow

#### Mô tả ý tưởng và cơ chế hoạt động

Sai số optical flow được đo bằng **EPE (End-Point Error)** — khoảng cách Euclid trung bình giữa vector flow dự đoán và vector flow thật tại mỗi pixel:

$$EPE = \frac{1}{N} \sum_{i=1}^{N} \left\| \mathbf{f}^{pred}_i - \mathbf{f}^{gt}_i \right\|_2$$

Trên benchmark **KITTI**, còn dùng **Fl-all** — tỉ lệ pixel có flow sai (lệch quá ngưỡng cả về độ lớn lẫn tỉ lệ).

#### Ví dụ

| Mô hình / Benchmark | EPE |
|---|---|
| FlowNet2 (Sintel) | ~3.0–4.0 |
| RAFT (Sintel clean) | ~1.6 |
| RAFT (Sintel final) | ~2.7 |

(EPE càng thấp càng tốt — đơn vị là pixel.)

#### Ưu và nhược điểm

**Ưu điểm:** trực tiếp, dễ hiểu, là chuẩn lâu đời.
**Nhược điểm:** EPE trung bình bị chi phối bởi các pixel sai lớn ở vùng biên/che khuất; không phản ánh tốt độ mượt tổng thể.

### 5.6. Âm thanh và audio-visual

#### Mô tả ý tưởng và cơ chế hoạt động

- **Accuracy / mAP:** cho phân loại âm thanh và sự kiện audio-visual (VD: mAP trên **AudioSet** với ~527 lớp âm thanh đa nhãn).
- **WER (Word Error Rate):** cho nhận dạng tiếng nói (kể cả audio-visual speech recognition) — tỉ lệ lỗi từ (chèn + xoá + thay thế) so với câu thật, **càng thấp càng tốt**.

#### Ví dụ

| Bài toán / Benchmark | Metric | Giá trị điển hình |
|---|---|---|
| Phân loại âm thanh (AudioSet) | mAP | ~0.43 (PANNs) |
| Audio-Visual Speech Recognition (LRS3, môi trường ồn) | WER | thấp hơn audio-only đáng kể |

#### Ưu và nhược điểm

**Ưu điểm:** tái dùng được các metric chuẩn của phân loại và ASR.
**Nhược điểm:** chưa có metric thống nhất đo riêng "mức độ đóng góp của việc kết hợp hình-tiếng"; thường phải so sánh ablation audio-only / visual-only / audio-visual.

## 6. Các thách thức của xử lý video

Làm việc với dữ liệu video kế thừa mọi khó khăn của xử lý ảnh, cộng thêm những thách thức đặc thù do chiều thời gian và đa phương thức.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/13-video-processing/challenges.jpeg" style="width: 1000px;"/>

- **Chi phí tính toán và lưu trữ:** mọi phép tính bị nhân thêm thừa số $T$ (số frame). Một mô hình ảnh nhẹ có thể trở nên cực kỳ đắt khi áp lên video — đây là rào cản lớn nhất.
- **Phụ thuộc thời gian dài (long-range temporal):** hiểu một sự kiện kéo dài hàng phút đòi hỏi kết nối thông tin qua rất nhiều frame, vượt khả năng của các mô hình chỉ nhìn được vài giây.
- **Chi phí gán nhãn lớn:** annotate video (đặc biệt là mask theo frame cho VOS, hay ranh giới thời gian cho localization) tốn kém hơn ảnh tĩnh nhiều lần.
- **Tính nhất quán thời gian (temporal consistency):** kết quả xử lý từng frame độc lập dễ bị "nhấp nháy" (flickering) — nhãn/mask/bounding box nhảy lung tung giữa các frame liền kề.
- **Nhiễu chuyển động thực tế:** camera rung/di chuyển (camera motion), che khuất (occlusion), nhoè chuyển động (motion blur), thay đổi ánh sáng — làm chuyển động biểu kiến phức tạp.
- **Ràng buộc thời gian thực (real-time / streaming):** nhiều ứng dụng (giám sát, xe tự hành) yêu cầu xử lý ngay khi luồng video tới, không được chờ toàn bộ video.
- **Đồng bộ hình-tiếng (audio-visual synchronization):** lệch đồng bộ giữa kênh hình và kênh tiếng làm hỏng các mô hình đa phương thức.
- **Dịch chuyển miền và mất cân bằng dữ liệu (domain shift & imbalance):** mô hình train trên một loại video (VD: phim) hoạt động kém trên loại khác (VD: camera giám sát góc rộng); phân bố hành động trong dữ liệu thường lệch nặng.
