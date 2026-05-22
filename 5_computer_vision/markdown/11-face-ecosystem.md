---
time: 06/16/2022
title: Hệ sinh thái các bài toán với dữ liệu khuôn mặt
description: Trong thị giác máy tính, hệ sinh thái khuôn mặt (facial ecosystem) không phải là một mô hình đơn lẻ mà là một tập hợp các mô hình chuyên biệt, hoạt động phối hợp với nhau để thực hiện một chuỗi các tác vụ phân tích, nhận dạng và xử lý khuôn mặt người từ hình ảnh hoặc video. Các mô hình này là nền tảng cho vô số ứng dụng trong đời sống, từ mở khóa điện thoại, chấm công, giám sát an ninh cho đến các hiệu ứng trên mạng xã hội. Một quy trình xử lý khuôn mặt điển hình thường bao gồm nhiều bước, mỗi bước được đảm nhiệm bởi một hoặc nhiều loại mô hình khác nhau.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về hệ sinh thái các bài toán với dữ liệu khuôn mặt

### 1.1. Lịch sử phát triển

> 🖼️ **[Ảnh placeholder #1 — Timeline lịch sử]**
> **Prompt:** *"Horizontal landscape infographic in cute kawaii pastel style. A long timeline from left to right showing milestones of face recognition history: 2001 Viola–Jones (a tiny cat detective with magnifying glass and Haar-like rectangles), 2014 DeepFace (a cute robot with neural net brain), 2015 FaceNet (three smiling cat faces connected by triplet lines anchor-positive-negative), 2019 ArcFace (a cat with golden angular margin halo), 2023 Diffusion era (a sparkly cloud generating faces). Soft pastel colors, minimal clean background, small Vietnamese labels under each milestone."*

Lịch sử phát triển của hệ sinh thái khuôn mặt gắn liền với sự tiến hoá của thị giác máy tính nói chung. Giai đoạn cổ điển khởi đầu bằng phương pháp **Viola–Jones (2001)** dùng Haar-like features và AdaboostCascade, mở ra khả năng phát hiện khuôn mặt thời gian thực trên CPU phổ thông. Tiếp đó, **Deformable Parts Model (DPM)** và các đặc trưng thủ công như **LBP**, **HOG** chiếm ưu thế trong giai đoạn 2005–2012.

> 🖼️ **[Ảnh placeholder #2 — So sánh classical vs deep learning]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style, split into two halves. Left half labeled 'Classical': a cat face overlaid with simple black-and-white Haar rectangles and HOG histogram bars, looking a bit old-fashioned with sepia tones. Right half labeled 'Deep Learning': the same cat face passing through a colorful glowing neural network of stacked layers, producing a vibrant embedding vector. A friendly arrow in the middle showing evolution. Soft pastel colors, minimal clean background."*

Bước ngoặt đến từ kỷ nguyên deep learning: **DeepFace (Facebook, 2014)** lần đầu đạt độ chính xác xấp xỉ con người trên LFW, **FaceNet (Google, 2015)** giới thiệu triplet loss học trực tiếp embedding 128 chiều, **MTCNN (2016)** thống nhất phát hiện và căn chỉnh khuôn mặt. Các margin-based softmax như **SphereFace (2017)**, **CosFace (2018)** và đặc biệt **ArcFace (2019)** đã đẩy độ chính xác verification lên >99.8% trên LFW.

> 🖼️ **[Ảnh placeholder #3 — Margin loss intuition]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. A 2D circular embedding space (a soft pink circle) with several clusters of cute cat faces — each cluster a different person, separated by glowing angular margins drawn as rainbow pie slices (representing ArcFace additive angular margin). Small mascot pointing at one slice with caption 'angular margin'. Friendly, soft pastel colors, clean minimal background."*

Từ 2020 trở đi, hệ sinh thái mở rộng sang **GAN/Diffusion** cho face swap (FSGAN, SimSwap, DiffFace), các mô hình **lightweight** (MobileFaceNet, GhostFaceNet) cho thiết bị di động, và **transformer-based** (ViTPose, FaRL) cho các tác vụ phân tích chi tiết.

### 1.2. Ứng dụng thực tế

> 🖼️ **[Ảnh placeholder #4 — Collage ứng dụng]**
> **Prompt:** *"Horizontal landscape collage illustration in cute kawaii pastel style. Six rounded mini-scenes in a single row: (1) a cat smiling at a phone showing Face ID unlock with a heart, (2) a cat tapping a face scanner at an office door (chấm công), (3) a cat at an ATM with a face scan beam, (4) a cat at airport gate with biometric passport, (5) a cat with AR bunny ears and sparkles (social media filter), (6) a cat with stethoscope analyzing facial features. Soft pastel colors, minimal clean background, friendly and warm vibe."*

Hệ sinh thái khuôn mặt đã thâm nhập rộng rãi vào đời sống:

- **Xác thực cá nhân**: mở khoá điện thoại (Apple Face ID, Android Face Unlock), chấm công, kiểm soát ra vào toà nhà.
- **Tài chính – ngân hàng**: xác thực eKYC khi mở tài khoản online, xác nhận giao dịch chuyển khoản, anti-spoofing để chống giả mạo CMND/CCCD.
- **An ninh – công cộng**: hộ chiếu sinh trắc tại sân bay, hệ thống giám sát đô thị, tìm người mất tích.
- **Mạng xã hội & giải trí**: hiệu ứng AR/filter trên TikTok, Snapchat, Instagram; ứng dụng deepfake giải trí (Reface, FaceApp); hậu kỳ phim ảnh.

> 🖼️ **[Ảnh placeholder #5 — eKYC banking flow]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. A step-by-step eKYC flow from left to right: cat user holding up an ID card → cat selfie with phone camera (with face detection box) → green checkmark and 'matched' badge → cat happily holding a new bank account card. Connected by soft arrows. Friendly colors, minimal clean background, Vietnamese label 'eKYC' at the top."*

- **Y tế**: hỗ trợ chẩn đoán các hội chứng di truyền (ví dụ DeepGestalt cho hội chứng Williams, Noonan).
- **Bán lẻ & marketing**: phân tích nhân khẩu học khách hàng (đếm người, ước lượng tuổi/giới tính), đo lường cảm xúc khi xem quảng cáo.
- **Ô tô thông minh**: giám sát sự tập trung của tài xế (Driver Monitoring System) qua hướng nhìn và độ mở mắt.

> 🖼️ **[Ảnh placeholder #6 — Driver Monitoring System]**
> **Prompt:** *"Horizontal landscape illustration in cute kawaii pastel style. Interior of a cute car with a cat driver behind the wheel. A small dashboard camera shines a soft beam onto the driver's face, with an overlay showing gaze direction arrows, eye-open percentage, and a friendly 'Tập trung 95%' indicator. Cute trees flying past the side window. Soft pastel colors, warm and safe atmosphere."*

## 2. Chi tiết từng bài toán trong hệ sinh thái khuôn mặt

### 2.1. Phát hiện khuôn mặt (Face Detection)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/face_detection.jpeg" style="width: 800px;"/>

#### Định nghĩa
- **Input**: ảnh hoặc khung hình video (RGB).
- **Output**: danh sách các hộp giới hạn (bounding box) bao quanh từng khuôn mặt, kèm điểm tin cậy (confidence score). Một số mô hình hiện đại còn trả về thêm 5 điểm mốc cơ bản (2 mắt, 1 mũi, 2 khoé miệng).

> 🖼️ **[Ảnh placeholder #7 — Pipeline input/output]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. A simple pipeline diagram from left to right: an input photo of a group of cute cats at a picnic → an arrow → the same photo with green bounding boxes drawn around each cat face, each labeled with 'conf: 0.98'. Above each box, 5 small dots marking eyes, nose and mouth corners. Soft pastel colors, minimal clean background."*

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **WIDER FACE**: 32.203 ảnh / 393.703 khuôn mặt, chia 3 mức độ khó (Easy/Medium/Hard) theo che khuất, tư thế và quy mô.
- **FDDB**: 2.845 ảnh / 5.171 khuôn mặt, dùng ellipse annotation, là benchmark kinh điển.
- **PASCAL FACE**: 851 ảnh / 1.341 khuôn mặt, tuyển từ tập PASCAL VOC.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **Viola–Jones (2001)**: Haar cascade + AdaBoost, real-time trên CPU.
- **MTCNN (Zhang et al., 2016)**: cascaded CNN ba giai đoạn P-Net → R-Net → O-Net, đồng thời trả về 5 landmark.
- **RetinaFace (Deng et al., 2019)**: single-shot multi-level, dùng FPN + context module, SOTA trên WIDER FACE.
- **SCRFD (2021) / YOLOv5-Face / YOLOv8-Face**: tối ưu tốc độ và độ chính xác cho production.

> 🖼️ **[Ảnh placeholder #8 — Classical vs Modern detector]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. Two side-by-side panels. Left panel labeled 'Viola–Jones (2001)': a cat face overlaid with sliding black-and-white Haar rectangles in a cascade pattern. Right panel labeled 'MTCNN (2016)': three stacked cute neural network blocks labeled P-Net, R-Net, O-Net, with a face passing through and getting tighter boxes plus 5 landmark dots at the end. Soft pastel colors, minimal background."*

#### Ứng dụng của bài toán này
- Bước tiền xử lý bắt buộc cho mọi pipeline khuôn mặt downstream (recognition, attributes, anti-spoofing).
- Tự động lấy nét (auto-focus) và phơi sáng (auto-exposure) trên camera điện thoại.
- Đếm số người trong cửa hàng, hội nghị, lớp học.

#### Những khó khăn còn tồn đọng
- Phát hiện khuôn mặt rất nhỏ (tiny face dưới 20×20 px) trong ảnh đám đông.
- Khuôn mặt bị che khuất nặng (khẩu trang, kính râm, tay) hoặc ở góc nghiêng/profile lớn.

### 2.2. Căn chỉnh khuôn mặt & điểm mốc (Face Alignment / Landmark Detection)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/face_alignment.jpeg" style="width: 800px;"/>

#### Định nghĩa
- **Input**: ảnh khuôn mặt đã được crop (từ bước face detection).
- **Output**: tập toạ độ 2D (hoặc 3D) của các điểm mốc cố định trên khuôn mặt — phổ biến là 5, 68, 98 hoặc 106 điểm xác định vị trí mắt, lông mày, mũi, miệng và đường viền khuôn mặt.

> 🖼️ **[Ảnh placeholder #9 — 68 landmark trên mặt]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. A large cute cat face in the center, with 68 small glowing pink dots placed at the standard 68-landmark positions: 17 along the jawline, 5 on each eyebrow, 6 around each eye, 9 on the nose, 20 around the mouth. The dots are connected by very thin lines forming the face contour. Friendly expression, soft pastel pink background."*

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **300-W**: ghép từ LFPW, HELEN, AFW, XM2VTS — 68 điểm mốc, là benchmark chuẩn.
- **AFLW**: 25.993 khuôn mặt trên 21.997 ảnh, gắn nhãn 21 landmark, đa dạng tư thế và môi trường.
- **WFLW**: 7.500 ảnh, 98 điểm mốc, có thêm attribute label cho occlusion, pose, blur, makeup.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **Cascaded Regression** cổ điển: ESR (Cao 2012), SDM (Xiong & De la Torre 2013).
- **DAN (Trigeorgis et al., 2016)**: deep alignment network nhiều giai đoạn.
- **FAN (Bulat & Tzimiropoulos, 2017)**: Stacked Hourglass dự đoán heatmap cho landmark.
- **HRNet**: giữ độ phân giải cao xuyên suốt, đạt SOTA trên 300-W và WFLW.

> 🖼️ **[Ảnh placeholder #10 — Heatmap prediction]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. Left: a cute cat face input photo. Middle: an arrow into a stacked hourglass-shaped neural network (drawn as hourglass icons). Right: a grid of small heatmap images, each showing a soft Gaussian blob at a different landmark location (eye corner, nose tip, mouth corner). Soft pastel colors, friendly mascot pointing at the heatmaps."*

#### Ứng dụng của bài toán này
- Chuẩn hoá (alignment) ảnh khuôn mặt trước khi đưa vào face recognition — tăng độ chính xác đáng kể.
- Dán filter / hiệu ứng AR (TikTok, Snapchat) bám sát từng cử động cơ mặt.
- Tạo và điều khiển avatar 3D, Memoji.

#### Những khó khăn còn tồn đọng
- Sai số tăng mạnh ở khuôn mặt profile, bị che khuất nặng hoặc biểu cảm cực đoan (cười rộng, nhăn).
- Yêu cầu thời gian thực trên thiết bị di động khi cần track 60+ FPS.

### 2.3. Phát hiện điểm mốc cơ thể người (Human Keypoint Detection)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/human_keypoint_detection.jpeg" style="width: 800px;"/>

#### Định nghĩa
- **Input**: ảnh hoặc video có một hoặc nhiều người.
- **Output**: toạ độ các khớp cơ thể (vai, khuỷu tay, cổ tay, hông, đầu gối, mắt cá…) cho từng người. Phổ biến nhất là chuẩn 17 keypoint của COCO.

> 🖼️ **[Ảnh placeholder #11 — COCO 17 keypoint skeleton]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. A cute cat character doing a yoga warrior pose, with the COCO 17-keypoint skeleton overlaid: small colored dots at nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles — connected by thin rainbow lines forming a skeleton. A small legend on the right listing '17 keypoints'. Soft pastel colors, minimal clean background."*

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **COCO Keypoints**: ~250.000 person instances với 17 keypoint, benchmark phổ biến nhất.
- **MPII Human Pose**: 25.000 ảnh / 40.000 người, gắn nhãn 16 keypoint, gồm hơn 410 hoạt động.
- **CrowdPose**: 20.000 ảnh tập trung vào cảnh đông đúc, đánh giá khả năng phân biệt người chồng lấn.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **OpenPose (Cao et al., 2017)**: bottom-up, đề xuất Part Affinity Fields (PAFs) để ghép keypoint thành người.
- **HRNet (Sun et al., 2019)**: top-down, giữ feature map độ phân giải cao, SOTA trên COCO.
- **AlphaPose**: top-down, kết hợp Regional Multi-Person Pose Estimation (RMPE).
- **ViTPose (2022)**: dùng Vision Transformer thuần, đơn giản nhưng đạt SOTA mới.

> 🖼️ **[Ảnh placeholder #12 — Top-down vs Bottom-up]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. Two side-by-side panels. Left labeled 'Top-down (HRNet)': a scene with 3 cat dancers, first a big bounding box is drawn around each cat, then keypoints appear inside each box. Right labeled 'Bottom-up (OpenPose)': the same 3 cat dancers, all keypoints detected at once in the scene, then rainbow Part Affinity Field lines connect them into 3 separate skeletons. Soft pastel colors, clean minimal background."*

#### Ứng dụng của bài toán này
- Phân tích kỹ thuật vận động viên trong thể thao (golf swing, tư thế chạy).
- Motion capture không cần marker cho hoạt hình, game, VR.
- Giám sát sức khoẻ: phát hiện té ngã ở người cao tuổi, theo dõi vật lý trị liệu tại nhà.

#### Những khó khăn còn tồn đọng
- Cảnh đông người chen lấn (crowded scene) gây nhầm lẫn keypoint giữa các cá thể.
- Tự che khuất (self-occlusion) — ví dụ tay khuất sau lưng — làm mất keypoint.

### 2.4. Nhận diện khuôn mặt (Face Recognition)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/face_recognition.jpeg" style="width: 800px;"/>

#### Định nghĩa
- **Input**: ảnh khuôn mặt đã căn chỉnh (aligned), hoặc cặp ảnh để so sánh.
- **Output**: vector embedding (thường 128–512 chiều). Từ đó, **Identification** trả về ID gần nhất trong gallery (1-vs-N), còn **Verification** trả về quyết định cùng/khác người (1-vs-1) dựa trên cosine similarity.

> 🖼️ **[Ảnh placeholder #13 — Verification vs Identification]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. Two side-by-side panels. Left labeled 'Verification (1-vs-1)': two cat face photos with a big '=?' between them, output is a green check or red cross with a similarity score. Right labeled 'Identification (1-vs-N)': one cat face on the left, a row of 6 candidate cat faces (gallery) on the right with a glowing arrow pointing to the matching one (Rank-1). Soft pastel colors, clean minimal background."*

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **CASIA-WebFace** (494.414 ảnh / 10.575 ID) và **VGGFace2** (3.31M ảnh / 9.131 ID) — dùng để huấn luyện.
- **LFW** (13.233 ảnh / 5.749 cá nhân) — benchmark verification kinh điển.
- **IJB-A/B/C** và **MegaFace** (1M distractor) — đánh giá khả năng mở rộng quy mô lớn.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **DeepFace (Taigman et al., 2014)**: CNN 9 tầng với 3D alignment, đạt ~97% trên LFW.
- **FaceNet (Schroff et al., 2015)**: triplet loss học embedding 128-D, đạt 99.63% LFW.
- **ArcFace (Deng et al., 2019)**: Additive Angular Margin Loss, là baseline phổ biến nhất hiện nay; InsightFace là implementation tham khảo.
- **MobileFaceNet / GhostFaceNet**: mô hình nhẹ (~1M params) cho mobile và embedded.

> 🖼️ **[Ảnh placeholder #14 — Embedding space & triplet loss]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. A 2D embedding space drawn as a soft pink plane. Three labeled cat-face icons: 'Anchor' in the middle, 'Positive' (same identity, different photo) being pulled closer with a green arrow, 'Negative' (different identity) being pushed away with a red arrow. A dashed circle around anchor showing margin. Caption 'Triplet Loss' at top. Soft pastel colors, clean minimal background."*

#### Ứng dụng của bài toán này
- eKYC trong ngân hàng, ví điện tử (MoMo, ZaloPay), sàn chứng khoán.
- Face Unlock trên điện thoại, chấm công văn phòng, kiểm soát ra vào.
- Tìm kiếm người mất tích, điều tra tội phạm, gắn tag ảnh tự động (Facebook, Google Photos).

#### Những khó khăn còn tồn đọng
- **Bias** theo chủng tộc và giới tính — sai số ở phụ nữ da màu có thể cao gấp 10–100 lần so với nam da trắng (NIST FRVT).
- Open-set recognition (gặp người chưa từng có trong gallery) và robust với khẩu trang / kính / lão hoá theo thời gian.

### 2.5. Phân loại / Hồi quy thuộc tính khuôn mặt (Face Attributes Classification – Regression)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/face_attributes_classification.jpeg" style="width: 800px;"/>

Bài toán này là một họ rộng bao gồm: **phân tích biểu cảm / cảm xúc**, **ước lượng tuổi & giới tính**, và **phân loại các thuộc tính nhị phân** (đeo kính, có râu, tóc xoăn, đội mũ, đang cười…). Một số nhãn là phân loại rời rạc (giới tính, cảm xúc), một số là giá trị liên tục — ví dụ tuổi — nên gọi là Classification – Regression.

> 🖼️ **[Ảnh placeholder #15 — Multi-label attribute output]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. A cute smiling cat face with glasses and a tiny moustache in the center, with rounded tag badges floating around it like a tag cloud: 'Smile ✓', 'Glasses ✓', 'Moustache ✓', 'Young Adult', 'Female', 'No Hat ✗', 'Curly hair ✗'. Tags in soft pastel colors with check or cross icons. Clean minimal background."*

#### Định nghĩa
- **Input**: ảnh khuôn mặt đã crop.
- **Output**: đa nhãn — gồm cả nhãn phân loại (vui/buồn/giận, nam/nữ, có/không đeo kính) và giá trị hồi quy (tuổi tính bằng năm).

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **CelebA**: 202.599 ảnh của 10.177 người nổi tiếng, 40 thuộc tính nhị phân — dataset chuẩn cho attribute.
- **FER2013** (35k ảnh, 7 cảm xúc) và **AffectNet** (~1M ảnh, 8 nhãn cảm xúc + dimensional valence/arousal) cho biểu cảm.
- **IMDb-WIKI** (~500k ảnh có nhãn tuổi/giới tính từ wiki) và **UTKFace** (~20k ảnh, tuổi 0–116) cho ước lượng tuổi.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **DEX (Rothe et al., 2015)**: fine-tune VGG-16 cho age regression, thắng ChaLearn LAP.
- **Multi-task CNN**: chia sẻ backbone (ResNet / EfficientNet), nhiều head với sigmoid cross-entropy cho từng attribute.
- **EmotionNet** và các biến thể CNN+LSTM cho phân tích biểu cảm trong video.
- **FaRL (2022)** và các foundation model học từ cặp ảnh-text giúp transfer tốt sang nhiều attribute task.

> 🖼️ **[Ảnh placeholder #16 — Age regression & emotion wheel]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. Left half: a row of 6 cat faces aging from kitten (1 year) to elderly cat (100 years), with a continuous age slider underneath labeled 'Age Regression'. Right half: a circular emotion wheel with 7 cat faces around it showing Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral — each a different pastel color slice. Soft pastel colors, clean minimal background."*

#### Ứng dụng của bài toán này
- Phân tích nhân khẩu học khách hàng trong retail, đo lường cảm xúc người xem quảng cáo.
- Ứng dụng làm đẹp / thử mỹ phẩm ảo (FaceApp, YouCam Makeup).
- Lọc và gợi ý nội dung trên mạng xã hội theo cảm xúc.

#### Những khó khăn còn tồn đọng
- Mất cân bằng dữ liệu nghiêm trọng giữa các nhãn (ít ảnh người cao tuổi, ít ảnh "có râu") và bias chủng tộc.
- Ranh giới giữa các cảm xúc mờ (vừa buồn vừa ngạc nhiên) và lo ngại đạo đức / riêng tư khi suy luận thuộc tính nhạy cảm.

### 2.6. Ước lượng góc quay đầu (Head Pose Estimation)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/head_pose_estimation.jpeg" style="width: 800px;"/>

#### Định nghĩa
- **Input**: ảnh khuôn mặt.
- **Output**: 3 góc Euler — **yaw** (quay trái/phải), **pitch** (gật lên/xuống), **roll** (nghiêng đầu) — tính bằng độ.

> 🖼️ **[Ảnh placeholder #17 — 3 trục yaw/pitch/roll]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. A cute cat head in the center with three colored 3D rotation axes drawn through it: a red horizontal arrow labeled 'Yaw (trái/phải)', a green vertical arrow labeled 'Pitch (lên/xuống)', a blue front-back arrow labeled 'Roll (nghiêng)'. Around it, three small thumbnail cat heads showing each rotation in action. Soft pastel colors, minimal clean background."*

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **300W-LP**: ~61.000 ảnh tổng hợp từ 300-W bằng cách warp 3D, có nhãn pose chính xác.
- **AFLW2000-3D**: 2.000 ảnh thực với 68 landmark 3D và pose ground-truth, dùng để test.
- **BIWI Kinect Head Pose**: 24 chuỗi video của 20 người, có depth từ Kinect.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **Hopenet (Ruiz et al., 2018)**: multi-loss CNN — kết hợp classification (bin angle) + regression cho từng axis.
- **FSA-Net (Yang et al., 2019)**: fine-grained structure aggregation, nhẹ và chính xác.
- **6DRepNet (2022)**: dự đoán trực tiếp ma trận xoay 6D thay vì Euler, giảm singularity.
- Suy luận pose gián tiếp từ 3DMM fitting (kết hợp với 3D Face Reconstruction).

> 🖼️ **[Ảnh placeholder #18 — Ứng dụng DMS]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. Interior view of a cute car: a sleepy cat driver with head tilting to the side (large pitch angle), an in-car camera detecting the pose with overlay arrows showing yaw/pitch/roll values, a warning bubble pops up 'Mất tập trung!' with a friendly alert icon. Soft pastel colors, warm but slightly tense atmosphere."*

#### Ứng dụng của bài toán này
- Driver Monitoring System trên ô tô: phát hiện tài xế ngủ gật, mất tập trung.
- Gaze estimation và human-computer interaction (điều khiển con trỏ bằng đầu).
- Hỗ trợ căn chỉnh ảnh trước khi recognition để chuẩn hoá tư thế.

#### Những khó khăn còn tồn đọng
- Độ chính xác giảm mạnh khi yaw vượt 75°–90° (profile hoặc back view).
- Thiếu dữ liệu thực ở góc cực đoan — phụ thuộc nhiều vào ảnh tổng hợp.

### 2.7. Tái tạo 3D khuôn mặt (3D Face Reconstruction)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/face_3d_reconstruction.jpeg" style="width: 800px;"/>

#### Định nghĩa
- **Input**: 1 hoặc nhiều ảnh 2D của khuôn mặt.
- **Output**: mô hình 3D — thường là mesh (vài nghìn vertex) hoặc tham số của một mô hình 3DMM (Basel Face Model, FLAME) gồm shape, expression, pose, texture.

> 🖼️ **[Ảnh placeholder #19 — 2D → 3DMM → 3D mesh pipeline]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. A 3-stage pipeline from left to right: (1) a flat 2D selfie of a cute cat, (2) an arrow into a neural network box that outputs 3DMM parameters (shape, expression, pose, texture shown as small sliders), (3) a 3D rotating cat mesh model with wireframe visible, slightly turned to show the depth. Soft pastel colors, clean minimal background."*

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **300W-LP**: ảnh kèm ground-truth 3DMM parameters (fitting trên 300-W).
- **FaceWarehouse**: 3D scan của 150 người ở 20 biểu cảm khác nhau.
- **NoW Challenge**: benchmark chuẩn để đo lỗi reconstruction trên scan thực.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **3DMM fitting cổ điển (Blanz & Vetter, 1999)**: tối ưu hoá iterative để khớp 3DMM với ảnh và landmark.
- **PRNet (Feng et al., 2018)**: dự đoán UV position map, từ đó suy ra mesh 3D.
- **DECA (2021)**: regress FLAME params, tách biệt identity và expression, học từ in-the-wild image.
- **MICA (2022)**: chuyên về độ chính xác identity shape, đạt SOTA trên NoW Challenge.

> 🖼️ **[Ảnh placeholder #20 — Ứng dụng AR/VR avatar]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. Two side-by-side scenes. Left: a cute cat wearing VR goggles, in front of them a glowing 3D avatar of themselves mimicking their expression. Right: a cat in a shop holding a phone with AR try-on showing different pairs of sunglasses fitting their 3D face. Soft pastel colors, magical sparkles, clean minimal background."*

#### Ứng dụng của bài toán này
- Tạo avatar 3D cho VR/AR (Apple Vision Pro, Meta Quest), Memoji.
- Thử kính / trang sức / trang điểm ảo trong thương mại điện tử.
- Hậu kỳ phim ảnh: thay diễn viên đóng thế bằng diễn viên chính, làm trẻ hoá nhân vật.

#### Những khó khăn còn tồn đọng
- Khó tái tạo chi tiết nhỏ như nếp nhăn, lỗ chân lông từ ảnh 2D đơn lẻ.
- Ánh sáng phức tạp và môi trường in-the-wild làm sai lệch texture và shape.

### 2.8. Hoán đổi khuôn mặt (Face Swapping / Deepfake)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/face_swap.jpeg" style="width: 800px;"/>

#### Định nghĩa
- **Input**: ảnh/video nguồn (cung cấp **identity**) và ảnh/video đích (cung cấp **pose, expression, lighting**).
- **Output**: ảnh/video đích với danh tính bị thay bằng nguồn, nhưng giữ nguyên biểu cảm, tư thế và điều kiện ánh sáng.

> 🖼️ **[Ảnh placeholder #21 — Source identity + Target pose = Swap]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. A simple equation layout: a portrait of cat A (labeled 'Source: identity') + a portrait of cat B with a unique pose and smile (labeled 'Target: pose, expression') = a portrait that has cat A's face on cat B's pose and expression (labeled 'Swap result'). Big plus and equals signs in friendly bubble font. Soft pastel colors, clean minimal background."*

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **FaceForensics++**: 1.000 video gốc + 4.000 video giả bằng 4 phương pháp (DeepFakes, Face2Face, FaceSwap, NeuralTextures) — chuẩn cho deepfake detection.
- **Celeb-DF (v2)**: 590 video gốc + 5.639 deepfake chất lượng cao của người nổi tiếng.
- **DFDC (Facebook Deepfake Detection Challenge)**: ~100.000 video, dataset deepfake lớn nhất.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **DeepFakes (2017)**: autoencoder chia sẻ encoder, hai decoder riêng cho A và B — mở đầu phong trào.
- **FSGAN (Nirkin et al., 2019)**: subject-agnostic, không cần train riêng cho từng cặp.
- **SimSwap (2020)**: ID injection module, chất lượng cao và tổng quát hoá tốt.
- **DiffFace (2023)** và các phương pháp dựa trên **Diffusion / StyleGAN2** cho chất lượng SOTA.

> 🖼️ **[Ảnh placeholder #22 — Autoencoder DeepFakes architecture]**
> **Prompt:** *"Horizontal landscape diagram illustration, cute kawaii pastel style. A central shared encoder (a friendly funnel shape labeled 'Encoder') with two cat photos feeding in from the left (cat A on top, cat B on bottom). From the encoder, the latent vector splits to TWO separate decoders on the right (labeled 'Decoder A' and 'Decoder B'), each a funnel reversed, reconstructing each cat. Arrows clearly show shared encoder + separate decoders. Soft pastel colors, clean minimal background."*

#### Ứng dụng của bài toán này
- Hậu kỳ điện ảnh: thay mặt diễn viên đóng thế, hồi sinh diễn viên đã mất, làm trẻ/già nhân vật.
- Lồng tiếng đa ngôn ngữ với khẩu hình khớp (visual dubbing).
- Ứng dụng giải trí (Reface, Avatarify) — người dùng tự gắn mặt mình vào meme, MV ca nhạc.

#### Những khó khăn còn tồn đọng
- Artifact ở biên (edge), không khớp ánh sáng / màu da giữa nguồn và đích — đặc biệt ở high-resolution.
- Rủi ro lạm dụng (deepfake porn, tin giả chính trị, lừa đảo) — thúc đẩy nhu cầu deepfake detection và watermarking.

### 2.9. Chống giả mạo khuôn mặt (Face Anti-Spoofing / Presentation Attack Detection)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/face_anti_spoofing.jpeg" style="width: 800px;"/>

#### Định nghĩa
- **Input**: ảnh hoặc video khuôn mặt từ camera.
- **Output**: nhãn nhị phân **real (live) / fake (spoof)**, đôi khi kèm loại tấn công (print, replay, 3D mask).

> 🖼️ **[Ảnh placeholder #23 — Các loại tấn công spoofing]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. A row of 4 attack types in front of a cute phone camera: (1) 'Print': a hand holding a printed paper photo of a cat, (2) 'Replay': a tablet screen showing a cat video, (3) '3D Mask': a cat wearing a silicone mask of another cat's face, (4) 'Real': a genuine smiling cat — marked with a green check. The phone reacts with red X for fake, green check for real. Soft pastel colors, friendly atmosphere despite the security theme."*

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **CASIA-FASD** (600 video, 3 mức chất lượng) và **Replay-Attack** (1.300 video) — benchmark kinh điển.
- **OULU-NPU**: 4.950 video, 4 protocol đánh giá generalization theo điều kiện và thiết bị.
- **CelebA-Spoof** (625.000 ảnh) và **HiFiMask** (75 mask 3D chất lượng cao) — quy mô lớn, đa dạng.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **Auxiliary Supervision (Liu et al., 2018)**: học depth map và rPPG signal làm nhãn phụ.
- **CDCN — Central Difference Convolution Network (Yu et al., 2020)**: thay conv thường bằng CDC để bắt texture vi mô.
- **NAS-FAS**: Neural Architecture Search cho bài toán anti-spoofing.
- **ViTranZFAS (2021)** và các transformer-based — học attention cho cross-domain generalization.

> 🖼️ **[Ảnh placeholder #24 — Depth & rPPG auxiliary cues]**
> **Prompt:** *"Horizontal landscape illustration, cute kawaii pastel style. Two side-by-side panels with a cat face input. Left labeled 'Depth map': the real cat face shows a smooth 3D depth surface in soft pastel blues/purples, while a printed-photo attack shows a flat plane. Right labeled 'rPPG signal': a tiny heart-beat waveform glowing on the real cat's cheek (showing live blood flow), while the photo attack shows a flat line. Soft pastel colors, clean minimal background."*

#### Ứng dụng của bài toán này
- Bảo vệ eKYC trong ngân hàng chống tấn công bằng ảnh in / ảnh trên màn hình điện thoại.
- Mobile Face Unlock (đảm bảo không mở khoá được bằng ảnh chủ nhân).
- Xác thực sinh trắc tại ATM, cổng kiểm soát biên giới.

#### Những khó khăn còn tồn đọng
- Tấn công 3D mask cao cấp (silicon mask) ngày càng giống thật.
- Cross-domain generalization kém — mô hình train trên dataset này thường tụt mạnh khi test trên dataset khác hoặc trên thiết bị mới.

## 3. Các thách thức chung của hệ sinh thái khuôn mặt

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/challenges.jpeg" style="width: 800px;"/>

1. **Thiên lệch dữ liệu và công bằng (Bias & Fairness)** — các dataset lớn (MS-Celeb-1M, VGGFace2) phần lớn là người da trắng và nam giới, dẫn tới sai số chênh lệch hàng chục lần giữa các nhóm chủng tộc/giới tính (Buolamwini & Gebru 2018, NIST FRVT report). Cần re-balance dataset, dùng loss có trọng số hoặc fairness-aware training.

2. **Quyền riêng tư và pháp lý** — khuôn mặt là dữ liệu sinh trắc nhạy cảm. GDPR (EU), CCPA (California) và đặc biệt **EU AI Act (2024)** xếp face recognition vào nhóm rủi ro cao, hạn chế nghiêm ngặt giám sát thời gian thực ở nơi công cộng. Các kỹ thuật privacy-preserving (federated learning, homomorphic encryption, face anonymization) ngày càng quan trọng.

3. **Robustness với điều kiện thực tế (PIE & Occlusion)** — hiệu năng tụt mạnh khi gặp ánh sáng kém, độ phân giải thấp, khuôn mặt nghiêng > 60°, hoặc bị che khuất bởi khẩu trang / kính râm / tóc. Đại dịch COVID-19 đẩy nhu cầu masked face recognition lên cao. Giải pháp: data augmentation chuyên biệt, domain adaptation, adversarial robustness.

4. **Hiệu năng real-time trên edge devices** — các ứng dụng mobile, IoT, camera giám sát đòi hỏi vừa chính xác vừa chạy < 30 ms trên CPU/NPU. Hướng nghiên cứu: mô hình lightweight (MobileFaceNet, GhostFaceNet), knowledge distillation, lượng tử hoá INT8, và tận dụng NPU chuyên dụng của Apple / Qualcomm / MediaTek.
