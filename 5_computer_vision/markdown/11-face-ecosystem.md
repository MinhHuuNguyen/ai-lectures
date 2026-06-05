---
time: 06/16/2022
title: Hệ sinh thái các bài toán với dữ liệu khuôn mặt
description: Trong thị giác máy tính, hệ sinh thái khuôn mặt (facial ecosystem) không phải là một mô hình đơn lẻ mà là một tập hợp các mô hình chuyên biệt, hoạt động phối hợp với nhau để thực hiện một chuỗi các tác vụ phân tích, nhận dạng và xử lý khuôn mặt người từ hình ảnh hoặc video. Các mô hình này là nền tảng cho vô số ứng dụng trong đời sống, từ mở khóa điện thoại, chấm công, giám sát an ninh cho đến các hiệu ứng trên mạng xã hội. Một quy trình xử lý khuôn mặt điển hình thường bao gồm nhiều bước, mỗi bước được đảm nhiệm bởi một hoặc nhiều loại mô hình khác nhau.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: false
---

## 1. Giới thiệu chung về hệ sinh thái các bài toán với dữ liệu khuôn mặt

### 1.1. Lịch sử phát triển

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/history_timeline.jpeg" style="width: 800px;"/>

Lịch sử phát triển của hệ sinh thái khuôn mặt gắn liền với sự tiến hoá của thị giác máy tính nói chung. Giai đoạn cổ điển khởi đầu bằng phương pháp **Viola–Jones (2001)** dùng Haar-like features và AdaboostCascade, mở ra khả năng phát hiện khuôn mặt thời gian thực trên CPU phổ thông. Tiếp đó, **Deformable Parts Model (DPM)** và các đặc trưng thủ công như **LBP**, **HOG** chiếm ưu thế trong giai đoạn 2005–2012.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/classical_vs_deep_learning.jpeg" style="width: 800px;"/>

Bước ngoặt đến từ kỷ nguyên deep learning: **DeepFace (Facebook, 2014)** lần đầu đạt độ chính xác xấp xỉ con người trên LFW, **FaceNet (Google, 2015)** giới thiệu triplet loss học trực tiếp embedding 128 chiều, **MTCNN (2016)** thống nhất phát hiện và căn chỉnh khuôn mặt. Các margin-based softmax như **SphereFace (2017)**, **CosFace (2018)** và đặc biệt **ArcFace (2019)** đã đẩy độ chính xác verification lên >99.8% trên LFW.

Từ 2020 trở đi, hệ sinh thái mở rộng sang **GAN/Diffusion** cho face swap (FSGAN, SimSwap, DiffFace), các mô hình **lightweight** (MobileFaceNet, GhostFaceNet) cho thiết bị di động, và **transformer-based** (ViTPose, FaRL) cho các tác vụ phân tích chi tiết.

### 1.2. Ứng dụng thực tế

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/applications_collage.jpeg" style="width: 800px;"/>

Hệ sinh thái khuôn mặt đã thâm nhập rộng rãi vào đời sống:

- **Xác thực cá nhân**: mở khoá điện thoại (Apple Face ID, Android Face Unlock), chấm công, kiểm soát ra vào toà nhà.
- **Tài chính – ngân hàng**: xác thực eKYC khi mở tài khoản online, xác nhận giao dịch chuyển khoản, anti-spoofing để chống giả mạo CMND/CCCD.
- **An ninh – công cộng**: hộ chiếu sinh trắc tại sân bay, hệ thống giám sát đô thị, tìm người mất tích.
- **Mạng xã hội & giải trí**: hiệu ứng AR/filter trên TikTok, Snapchat, Instagram; ứng dụng deepfake giải trí (Reface, FaceApp); hậu kỳ phim ảnh.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/ekyc_flow.jpeg" style="width: 800px;"/>

- **Y tế**: hỗ trợ chẩn đoán các hội chứng di truyền (ví dụ DeepGestalt cho hội chứng Williams, Noonan).
- **Bán lẻ & marketing**: phân tích nhân khẩu học khách hàng (đếm người, ước lượng tuổi/giới tính), đo lường cảm xúc khi xem quảng cáo.
- **Ô tô thông minh**: giám sát sự tập trung của tài xế (Driver Monitoring System) qua hướng nhìn và độ mở mắt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/driver_monitoring_system.jpeg" style="width: 800px;"/>

## 2. Chi tiết từng bài toán trong hệ sinh thái khuôn mặt

### 2.1. Phát hiện khuôn mặt (Face Detection)

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/face_detection.jpeg" style="width: 800px;"/>

#### Định nghĩa
- **Input**: ảnh hoặc khung hình video (RGB).
- **Output**: danh sách các hộp giới hạn (bounding box) bao quanh từng khuôn mặt, kèm điểm tin cậy (confidence score). Một số mô hình hiện đại còn trả về thêm 5 điểm mốc cơ bản (2 mắt, 1 mũi, 2 khoé miệng).

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **WIDER FACE**: 32.203 ảnh / 393.703 khuôn mặt, chia 3 mức độ khó (Easy/Medium/Hard) theo che khuất, tư thế và quy mô.
- **FDDB**: 2.845 ảnh / 5.171 khuôn mặt, dùng ellipse annotation, là benchmark kinh điển.
- **PASCAL FACE**: 851 ảnh / 1.341 khuôn mặt, tuyển từ tập PASCAL VOC.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **MTCNN (Zhang et al., 2016)**: cascaded CNN ba giai đoạn P-Net → R-Net → O-Net, đồng thời trả về 5 landmark.
- **RetinaFace (Deng et al., 2019)**: single-shot multi-level, dùng FPN + context module, SOTA trên WIDER FACE.
- **SCRFD (2021) / YOLOv5-Face / YOLOv8-Face**: tối ưu tốc độ và độ chính xác cho production.

#### Ứng dụng của bài toán này
- Bước tiền xử lý bắt buộc cho mọi pipeline khuôn mặt downstream (recognition, attributes, anti-spoofing).
- Tự động lấy nét (auto-focus) và phơi sáng (auto-exposure) trên camera điện thoại.
- Đếm số người trong cửa hàng, hội nghị, lớp học.

#### Những khó khăn còn tồn đọng
- Phát hiện khuôn mặt rất nhỏ (tiny face dưới 20×20 px) trong ảnh đám đông.
- Khuôn mặt bị che khuất nặng (khẩu trang, kính râm, tay) hoặc ở góc nghiêng/profile lớn.

### 2.2. Căn chỉnh khuôn mặt & điểm mốc (Face Alignment / Landmark Detection)

#### Định nghĩa
- **Input**: ảnh khuôn mặt đã được crop (từ bước face detection).
- **Output**: tập toạ độ 2D (hoặc 3D) của các điểm mốc cố định trên khuôn mặt — phổ biến là 5, 68, 98 hoặc 106 điểm xác định vị trí mắt, lông mày, mũi, miệng và đường viền khuôn mặt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/face_landmark_68.jpeg" style="width: 800px;"/>

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **300-W**: ghép từ LFPW, HELEN, AFW, XM2VTS — 68 điểm mốc, là benchmark chuẩn.
- **AFLW**: 25.993 khuôn mặt trên 21.997 ảnh, gắn nhãn 21 landmark, đa dạng tư thế và môi trường.
- **WFLW**: 7.500 ảnh, 98 điểm mốc, có thêm attribute label cho occlusion, pose, blur, makeup.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **HRNet**: giữ độ phân giải cao xuyên suốt, đạt SOTA trên 300-W và WFLW.
- Ngoài ra, các mô hình Face Detection hiện đại như RetinaFace cũng tích hợp head landmark detection, giúp tiết kiệm thời gian xử lý.

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

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **COCO Keypoints**: ~250.000 person instances với 17 keypoint, benchmark phổ biến nhất.
- **MPII Human Pose**: 25.000 ảnh / 40.000 người, gắn nhãn 16 keypoint, gồm hơn 410 hoạt động.
- **CrowdPose**: 20.000 ảnh tập trung vào cảnh đông đúc, đánh giá khả năng phân biệt người chồng lấn.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **OpenPose (Cao et al., 2017)**: bottom-up, đề xuất Part Affinity Fields (PAFs) để ghép keypoint thành người.
- **HRNet (Sun et al., 2019)**: top-down, giữ feature map độ phân giải cao, SOTA trên COCO.
- **AlphaPose**: top-down, kết hợp Regional Multi-Person Pose Estimation (RMPE).
- **ViTPose (2022)**: dùng Vision Transformer thuần, đơn giản nhưng đạt SOTA mới.

#### Ứng dụng của bài toán này
- Phân tích kỹ thuật vận động viên trong thể thao (golf swing, tư thế chạy).
- Motion capture không cần marker cho hoạt hình, game, VR.
- Giám sát sức khoẻ: phát hiện té ngã ở người cao tuổi, theo dõi vật lý trị liệu tại nhà.

#### Những khó khăn còn tồn đọng
- Cảnh đông người chen lấn (crowded scene) gây nhầm lẫn keypoint giữa các cá thể.
- Tự che khuất (self-occlusion) — ví dụ tay khuất sau lưng — làm mất keypoint.

### 2.4. Nhận diện khuôn mặt (Face Recognition)

#### Định nghĩa
- **Input**: ảnh khuôn mặt đã căn chỉnh (aligned), hoặc cặp ảnh để so sánh.
- **Output**: vector embedding (thường 128–512 chiều). Từ đó, **Identification** trả về ID gần nhất trong gallery (1-vs-N), còn **Verification** trả về quyết định cùng/khác người (1-vs-1) dựa trên cosine similarity.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/verification_vs_identification.jpeg" style="width: 800px;"/>

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **CASIA-WebFace** (494.414 ảnh / 10.575 ID) và **VGGFace2** (3.31M ảnh / 9.131 ID) — dùng để huấn luyện.
- **LFW** (13.233 ảnh / 5.749 cá nhân) — benchmark verification kinh điển.
- **IJB-A/B/C** và **MegaFace** (1M distractor) — đánh giá khả năng mở rộng quy mô lớn.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **DeepFace (Taigman et al., 2014)**: CNN 9 tầng với 3D alignment, đạt ~97% trên LFW.
- **FaceNet (Schroff et al., 2015)**: triplet loss học embedding 128-D, đạt 99.63% LFW.
- **ArcFace (Deng et al., 2019)**: Additive Angular Margin Loss, là baseline phổ biến nhất hiện nay; InsightFace là implementation tham khảo.
- **MobileFaceNet / GhostFaceNet**: mô hình nhẹ (~1M params) cho mobile và embedded.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/triplet_loss.jpeg" style="width: 800px;"/>

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

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/age_regression_emotion_wheel.jpeg" style="width: 800px;"/>

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

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **300W-LP**: ~61.000 ảnh tổng hợp từ 300-W bằng cách warp 3D, có nhãn pose chính xác.
- **AFLW2000-3D**: 2.000 ảnh thực với 68 landmark 3D và pose ground-truth, dùng để test.
- **BIWI Kinect Head Pose**: 24 chuỗi video của 20 người, có depth từ Kinect.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **Hopenet (Ruiz et al., 2018)**: multi-loss CNN — kết hợp classification (bin angle) + regression cho từng axis.
- **FSA-Net (Yang et al., 2019)**: fine-grained structure aggregation, nhẹ và chính xác.
- **6DRepNet (2022)**: dự đoán trực tiếp ma trận xoay 6D thay vì Euler, giảm singularity.
- Suy luận pose gián tiếp từ 3DMM fitting (kết hợp với 3D Face Reconstruction).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/head_pose_dms_distracted.jpeg" style="width: 800px;"/>

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

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **300W-LP**: ảnh kèm ground-truth 3DMM parameters (fitting trên 300-W).
- **FaceWarehouse**: 3D scan của 150 người ở 20 biểu cảm khác nhau.
- **NoW Challenge**: benchmark chuẩn để đo lỗi reconstruction trên scan thực.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **3DMM fitting cổ điển (Blanz & Vetter, 1999)**: tối ưu hoá iterative để khớp 3DMM với ảnh và landmark.
- **PRNet (Feng et al., 2018)**: dự đoán UV position map, từ đó suy ra mesh 3D.
- **DECA (2021)**: regress FLAME params, tách biệt identity và expression, học từ in-the-wild image.
- **MICA (2022)**: chuyên về độ chính xác identity shape, đạt SOTA trên NoW Challenge.

#### Ứng dụng của bài toán này
- Tạo avatar 3D cho VR/AR (Apple Vision Pro, Meta Quest), Memoji.
- Thử kính / trang sức / trang điểm ảo trong thương mại điện tử.
- Hậu kỳ phim ảnh: thay diễn viên đóng thế bằng diễn viên chính, làm trẻ hoá nhân vật.

#### Những khó khăn còn tồn đọng
- Khó tái tạo chi tiết nhỏ như nếp nhăn, lỗ chân lông từ ảnh 2D đơn lẻ.
- Ánh sáng phức tạp và môi trường in-the-wild làm sai lệch texture và shape.

### 2.8. Hoán đổi khuôn mặt (Face Swapping / Deepfake)

#### Định nghĩa
- **Input**: ảnh/video nguồn (cung cấp **identity**) và ảnh/video đích (cung cấp **pose, expression, lighting**).
- **Output**: ảnh/video đích với danh tính bị thay bằng nguồn, nhưng giữ nguyên biểu cảm, tư thế và điều kiện ánh sáng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/face_swap_equation.jpeg" style="width: 800px;"/>

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **FaceForensics++**: 1.000 video gốc + 4.000 video giả bằng 4 phương pháp (DeepFakes, Face2Face, FaceSwap, NeuralTextures) — chuẩn cho deepfake detection.
- **Celeb-DF (v2)**: 590 video gốc + 5.639 deepfake chất lượng cao của người nổi tiếng.
- **DFDC (Facebook Deepfake Detection Challenge)**: ~100.000 video, dataset deepfake lớn nhất.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **DeepFakes (2017)**: autoencoder chia sẻ encoder, hai decoder riêng cho A và B — mở đầu phong trào.
- **FSGAN (Nirkin et al., 2019)**: subject-agnostic, không cần train riêng cho từng cặp.
- **SimSwap (2020)**: ID injection module, chất lượng cao và tổng quát hoá tốt.
- **DiffFace (2023)** và các phương pháp dựa trên **Diffusion / StyleGAN2** cho chất lượng SOTA.

#### Ứng dụng của bài toán này
- Hậu kỳ điện ảnh: thay mặt diễn viên đóng thế, hồi sinh diễn viên đã mất, làm trẻ/già nhân vật.
- Lồng tiếng đa ngôn ngữ với khẩu hình khớp (visual dubbing).
- Ứng dụng giải trí (Reface, Avatarify) — người dùng tự gắn mặt mình vào meme, MV ca nhạc.

#### Những khó khăn còn tồn đọng
- Artifact ở biên (edge), không khớp ánh sáng / màu da giữa nguồn và đích — đặc biệt ở high-resolution.
- Rủi ro lạm dụng (deepfake porn, tin giả chính trị, lừa đảo) — thúc đẩy nhu cầu deepfake detection và watermarking.

### 2.9. Chống giả mạo khuôn mặt (Face Anti-Spoofing / Presentation Attack Detection)

#### Định nghĩa
- **Input**: ảnh hoặc video khuôn mặt từ camera.
- **Output**: nhãn nhị phân **real (live) / fake (spoof)**, đôi khi kèm loại tấn công (print, replay, 3D mask).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/spoofing_attack_types.jpeg" style="width: 800px;"/>

#### Một số bộ dữ liệu nổi bật đáng chú ý
- **CASIA-FASD** (600 video, 3 mức chất lượng) và **Replay-Attack** (1.300 video) — benchmark kinh điển.
- **OULU-NPU**: 4.950 video, 4 protocol đánh giá generalization theo điều kiện và thiết bị.
- **CelebA-Spoof** (625.000 ảnh) và **HiFiMask** (75 mask 3D chất lượng cao) — quy mô lớn, đa dạng.

#### Một số mô hình / phương pháp / kỹ thuật đáng chú ý
- **Auxiliary Supervision (Liu et al., 2018)**: học depth map và rPPG signal làm nhãn phụ.
- **CDCN — Central Difference Convolution Network (Yu et al., 2020)**: thay conv thường bằng CDC để bắt texture vi mô.
- **NAS-FAS**: Neural Architecture Search cho bài toán anti-spoofing.
- **ViTranZFAS (2021)** và các transformer-based — học attention cho cross-domain generalization.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/11-face-ecosystem/anti_spoofing_depth_rppg.jpeg" style="width: 800px;"/>

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
