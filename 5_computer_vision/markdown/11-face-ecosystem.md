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

Các bài toán về khuôn mặt trong CV có thể được phân chia thành nhiều nhóm chính. Phát hiện khuôn
mặt (face detection) tìm tọa độ hộp giới hạn các khuôn mặt trong ảnh. Căn chỉnh/điểm mốc (alignment/
landmark detection) xác định các điểm cố định (mắt, mũi, miệng, v.v.) trên khuôn mặt. Nhận diện khuôn
mặt (identification) xác định danh tính cá nhân dựa trên ảnh khuôn mặt; xác thực (verification) kiểm tra
hai ảnh có cùng một người hay không (cách tiếp cận đóng bộ và mở bộ).  Phân cụm khuôn mặt
(clustering) nhóm các ảnh khuôn mặt của cùng một người mà không có nhãn.  Theo dõi khuôn mặt
(tracking) gán ID cho cùng một cá thể qua các khung hình trong video. Nhóm phân tích khuôn mặt bao
gồm phân tích biểu cảm/cảm xúc, ước lượng tuổi/giới tính, ước lượng tư thế đầu, và dự đoán các thuộc tính
khuôn mặt (có mũ, kính, ria mép, cười, v.v.). Tái tạo 3D (3D face reconstruction) xây dựng mô hình 3D của
khuôn   mặt   từ   ảnh   2D.   Nhóm  tạo dựng khuôn mặt  gồm  tái hiện chuyển động khuôn mặt  (face
reenactment/animation) – di chuyển chuyển động từ video này sang khuôn mặt khác, và hoán đổi khuôn
mặt (deepfake) – thay đổi danh tính khuôn mặt trong ảnh/video. Cuối cùng, các bài toán về bảo mật/
riêng tư như chống giả mạo (anti-spoofing) xác định ảnh/video khuôn mặt giả mạo (in ảnh, màn hình,
mặt nạ 3D), ẩn danh khuôn mặt (phá huỷ hoặc thay đổi tính nhận dạng) và nhận dạng bảo mật quyền
riêng tư (công nghệ nhận diện không làm lộ thông tin nhạy). Các tác vụ đa phương thức có thể kết hợp
thông tin khuôn mặt với giọng nói hoặc văn bản để nhận diện/biểu cảm nâng cao.

## 2. Chi tiết từng bài toán trong hệ sinh thái khuôn mặt

### 2.1. Phát hiện khuôn mặt (Face Detection) và Căn chỉnh khuôn mặt (Face Alignment)

Định nghĩa: Nhận đầu vào là ảnh (hoặc khung hình video) và trả về danh sách các hộp giới hạn chứa
khuôn mặt. Đầu ra thường là tọa độ bounding box và/hoặc xác suất.
Các bộ dữ liệu: WIDER FACE (32.2k ảnh, 393.7k khuôn mặt với nhiều mức độ che khuất, biến đổi tư thế
và điều kiện ánh sáng) . FDDB (2.845 ảnh, 5.171 khuôn mặt) . PASCAL FACE (851 ảnh, 1.341 khuôn
mặt) . AFLW (21.997 ảnh, 25.993 khuôn mặt) . IJB-A (24.327 ảnh, 49.759 khuôn mặt) và MALF
(5.250 ảnh, 11.931 khuôn mặt) cũng được dùng trong thử nghiệm. Hiệu suất thường được đánh giá
bằng AP (Average Precision) hoặc tỉ lệ phát hiện tại một ngưỡng FPPI (False Positives Per Image) tương
tự PASCAL VOC.
Mô hình tiêu biểu: Các phương pháp cổ điển bao gồm Viola–Jones (2001) dùng Haar cascade  và
DPM (deformable parts model) . Trong thời đại học sâu, các mạng CNN như MTCNN, RetinaFace
(Deng et al., 2019) hay biến thể của YOLO/Faster R-CNN (Redmon et al.) được sử dụng rộng rãi.
RetinaFace (Cao et al., 2019) đạt SOTA trên WIDER. Phương pháp thường tiền xử lý bao gồm thay đổi
kích thước ảnh, tăng cường dữ liệu (phóng to, xoay). Xử lý hậu: lọc NMS (non-maximum suppression) để
gộp các kết quả trùng nhau. Thách thức chính là độ chính xác ở khuôn mặt nhỏ, bị che khuất hay đổi tư
thế lớn. Hiệu suất trên tập WIDER có thể đạt AP ~90–95% cho các khó khăn thấp nhưng vẫn giảm mạnh
ở nhóm mặt bị che khuất hoặc góc nghiêng lớn. Các bộ phân loại cần khả năng tổng quát hoá tốt nhằm
xử lý ảnh khuôn mặt ở nhiều điều kiện khác nhau

Định nghĩa: Cho ảnh khuôn mặt (crop bounding box), đầu ra là tọa độ các điểm mốc cố định trên
khuôn mặt (ví dụ 68 điểm định danh vị trí mắt, lông mày, mũi, miệng). Thông tin này thường dùng để
xoay/thay đổi ảnh về vị trí chuẩn hoặc làm tiền xử lý cho nhận diện.
Bộ dữ liệu: 300-W (2013) ghép từ các tập LFPW (1.287 ảnh) , HELEN (2.330 ảnh) , AFW (255 ảnh)
và XM2VTS, với 68 điểm mốc. AFLW (2011) có 21 landmark trên 25k ảnh với nhiều tư thế khác nhau .
COFW là tập ảnh bị che khuất (634 train, 689 test) với 29 điểm. WFLW (7.500 ảnh, 98 điểm) cung cấp
nhiều biểu cảm và điều kiện phức tạp. Metrix: NME (Normalized Mean Error) – trung bình sai số điểm đã
được chuẩn hoá theo khoảng cách giữa hai mắt (hoặc biên), AUC (area under CED curve) cho tiêu chí %
ảnh có lỗi < ngưỡng.
Mô hình tiêu biểu: Trước đây có ASM/AAM, sau đó là Cascaded Regression (ESR, SDM). Trong DL: DAN
1 2
3 4
5
6
7
8 9
10
2
(Trigeorgis et al., 2016) sử dụng mạng sâu đa giai đoạn , FAN (Bulat & Tzimiropoulos 2017), HRNet.
Các kiến trúc thường huấn luyện để dự đoán heatmap hoặc offset của landmark. Tiền xử lý: phát hiện
và crop khuôn mặt, có thể xoay thô. Hậu xử lý: vẽ đường viền, dùng PCA chỉnh sửa landmark. Thách
thức: khuôn mặt nghiêng, biến dạng khuôn mặt (cười, nhăn), che khuất (tóc, tay). Độ chính xác cao trên
ảnh thẳng hoặc ở ngoài trời, nhưng sai số tăng mạnh khi ánh sáng xấu hoặc che khuất nặng.

### 2.2. Nhận diện khuôn mặt (Face Recognition)

Định nghĩa: Nhận diện (identification) là gán ID cho một ảnh khuôn mặt từ một tập ID có sẵn; xác thực
(verification) trả về điểm tương đồng (score) giữa hai ảnh để quyết định có cùng cá nhân hay không.
Trong nhận diện đóng bộ (closed-set ID), danh sách ID biết trước; mở bộ (open-set) có thể gặp người
mới.
Đầu vào/đầu ra: Đầu vào là ảnh (đã căn chỉnh) hoặc embedding khuôn mặt; đầu ra là vector đặc trưng,
hoặc ID với độ tin cậy cao nhất. Verification trả về độ tương đồng (cosine, Euclid) hoặc quyết định nhị
phân.
Bộ dữ liệu: LFW (13.233 ảnh, 5.749 cá nhân)  – tập điển hình để đo xác thực 1-vs-1 (kaggle LFW); IJB-
A/B (cả ảnh tĩnh và video, ~25k mặt người, 10k ID) benchmark nghiêm ngặt; CASIA-WebFace (10.575 ID,
494.414 ảnh) , MS-Celeb-1M (100k ID, 10M ảnh), và VGGFace2 (9.131 ID, >3M ảnh)  là tập lớn dùng
huấn luyện. Tập test phổ biến: MegaFace (1M ảnh nhiễu) để đánh giá khả năng mở rộng.  Metrics:
Verification: độ chính xác (TAR@FAR ở các ngưỡng FAR thấp như 0.1%, 0.01%), ROC AUC. Identification:
tỉ lệ nhận đúng (Rank-1, Rank-5) hoặc CMC.
Mô hình tiêu biểu: Các mạng CNN sâu đóng vai trò chủ đạo. Khởi đầu với DeepFace (Taigman et al.
2014)   sử dụng CNN và multi-patch, DeepID (Sun et al. 2014) kết hợp CNN và loss Softmax/
contrastive, tiếp theo FaceNet (Schroff et al. 2015) với loss triplet achieving state-of-art . Các mô hình
gần đây sử dụng ResNet (ArcFace – Deng et al. 2019), bản quyền (CosFace – Wang et al. 2018) tăng
cường margin của Softmax để cải thiện phân biệt. InsightFace (dựa trên ArcFace) là phương pháp phổ
biến hiện nay. Các mô hình lightweight: MobileFaceNet, FaceNet-based trên MobileNet (Chen et al.,
2018) để chạy trên thiết bị di động. Tiền xử lý: phát hiện và căn chỉnh khuôn mặt, chuẩn hoá độ sáng
(EqualizeHist), loại bỏ ánh sáng. Hậu xử lý: bình thường hoá embedding, PCA LDA để giảm chiều. Thách
thức: tập huấn luyện có thể bị bias (nhiều người Âu/Mỹ hơn ở LFW, MS-Celeb,...), dẫn đến sai số ở người
thiểu số . Cùng với đó, kiểm soát độ nhạy cảm với che khuất, tụ tập nhiều ảnh (cluster) của cùng
người khi nhiều ảnh gốc. Nhận dạng mở (open-set) cũng là thách thức lớn.

### 2.3. Phân lớp các thuộc tính khuôn mặt (Face Attributes Classification)

 Phân tích biểu cảm và cảm xúc (Facial Expression/Affect Analysis)
Định nghĩa: Xác định trạng thái cảm xúc hoặc biểu cảm (vui, buồn, giận, ngạc nhiên, v.v.) từ ảnh hoặc
video khuôn mặt. Đầu ra là nhãn lớp hoặc xác suất các biểu cảm.
Bộ dữ liệu: CK+ (590 video mẫu của 123 người, 7 biểu cảm) – phòng thí nghiệm. FER2013 (35k ảnh, 7
11
12
13 14
15
15
16
3
nhãn) từ Kaggle, ảnh trong tự nhiên (Jang et al. 2013). AffectNet (400k ảnh, 8 nhãn cảm xúc) . RAF-
DB (~30k ảnh, 7 nhãn), JAFFE (213 ảnh, 7 nhãn) cũ hơn. Metrics: độ chính xác (accuracy) hoặc F1-score
với từng lớp.
Mô hình tiêu biểu: CNN đơn thuần (ResNet, VGGFace) hoặc CNN+LSTM cho video (dynamic). Mạng mới:
EmotionNet (Mollahosseini et al. 2016), CapsuleNet cũng được thử nghiệm. Tiền xử lý: căn chỉnh và crop
chặt khuôn mặt. Thách thức: sự đa dạng biểu cảm quá nhỏ so với ID, nền và ánh sáng gây nhiễu. Biểu
cảm có thể pha trộn (vừa buồn vừa ngạc nhiên) gây khó cho phân lớp.
2.6. Ước lượng tuổi và giới tính (Age/Gender Estimation)
Định nghĩa: Từ ảnh khuôn mặt, dự đoán tuổi hoặc giới tính (classification hoặc regression). Đầu ra: tuổi
gần đúng (số tuổi hay nhóm tuổi) và/hoặc nhãn giới tính.
Bộ dữ liệu: Adience (26.580 ảnh của 2.284 người, nhóm tuổi 8 nhãn, 2 giới tính) ; IMDb-WIKI (cả
triệu ảnh, gắn nhãn tuổi từ IMDb và Wikipedia, nhiễu); UTKFace (~20k, tuổi 0-116); ChaLearn 2015 (dữ
liệu video 5 lớp tuổi). Metrics: MAE (Mean Absolute Error) cho tuổi, accuracy cho tuổi theo nhóm,
accuracy cho giới tính.
Mô hình tiêu biểu: Mạng CNN (dạng classifier hoặc regressor). Ví dụ: DEX (Rothe et al., 2015) sử dụng
VGG-16 fine-tune. Hiện nay thường fine-tune ResNet hay EfficientNet, đôi khi ensemble. Thách thức:
chênh lệch ở các nhóm tuổi (ít ảnh người lớn tuổi), mặt trẻ khó phân biệt rõ tuổi, ánh sáng và góc mặt
cũng ảnh hưởng. Tương tự, giới tính ở tuổi niên thiếu có thể khó phân biệt, hay kem nền che mờ đặc
điểm.

Định nghĩa: Phân loại nhãn thuộc tính (ví dụ “có râu”, “đeo kính”, “cười”, giới tính, v.v.) từ ảnh khuôn
mặt.
Bộ dữ liệu: CelebA (202.599 ảnh của 10.177 người nổi tiếng, 40 thuộc tính nhị phân như giới tính, nụ
cười, đeo kính)¹; LFWA (13.143 ảnh, 40 thuộc tính), chúng được dùng phổ biến. Metrics: Multi-label
19
19
5
accuracy hoặc mAP.
Mô hình tiêu biểu:  Fine-tune CNN (ResNet, EfficientNet) với đầu ra đa nhãn. Architectures: CelebA
Challenge network (Liu et al. 2015). Loss thường là sigmoid cross-entropy trên mỗi thuộc tính. Thách
thức: mất cân bằng dữ liệu (một số thuộc tính ít ảnh, ví dụ có râu), mâu thuẫn trong định nhãn (biểu
cảm có thể lẫn với cười), và tính riêng tư (thông tin nhạy cảm).


### 2.4. Ước lượng góc quay đầu (Head Pose Estimation)

Định nghĩa: Dự đoán góc xoay đầu (yaw, pitch, roll) của khuôn mặt. Đầu ra: góc nghiêng (thường đo
bằng độ).
Bộ dữ liệu: AFLW2000-3D (2.000 ảnh vẽ lại 3D với 68 landmark và góc đánh dấu). Biwi Kinect Head Pose
(20 người, video có info 3D). 300W-LP (6.000 ảnh tổng hợp với góc nghiêng). Metrics: MSE (mean
squared error) của góc hoặc phân loại thô (ví dụ frontal, 30°, 60°).
Mô hình tiêu biểu: Hopenet (Ruiz et al., 2018) dùng CNN dự đoán góc. Các mạng CNN (ResNet, VGG)
cũng có thể tinh chỉnh cho bài này. Thách thức: khuôn mặt gần như đáy ảnh, che phủ một phần làm
lệch đo, cần dữ liệu đa dạng góc. Độ chính xác giảm khi góc lớn

### 2.5. Tái tạo 3D khuôn mặt (3D Face Reconstruction)

Định nghĩa: Từ ảnh 2D, khôi phục mô hình 3D của khuôn mặt (mặt nạ 3D hoặc thông số 3DMM).
Bộ dữ liệu: 300W-LP (ảnh có ground-truth 3D từ fitting), FaceWarehouse (3D scans của 150 người ở
nhiều biểu cảm), NoW Challenge (nhiều người, nhiều ảnh đầu). Metrics: lỗi Euclid trung bình của đỉnh
khi chuyển ảnh mẫu sang mô hình 3D tham chiếu.
Mô hình tiêu biểu: Phương pháp truyền thống dùng fitting 3DMM (Blanz & Vetter) kết hợp Landmark.
DL: PRNet (Feng et al. 2018) ước lượng 3D shape map. Định chế lại mạng Hourglass cũng được dùng
(Yang et al.). Thử nghiệm tiên tiến: Pixel2Mesh (Wang 2018) tạo lưới 3D trực tiếp, hoặc HoloFace
(Rathgeb 2017). Thách thức: phụ thuộc vào dữ liệu 3D có sẵn, khó tạo mức chi tiết cao. Góc khuất mặt
và ánh sáng phức tạp dễ gây sai khung 3D

### 2.6. Hoán đổi khuôn mặt (Face Swapping)

Định nghĩa: Đổi danh tính của khuôn mặt trong ảnh/video mà vẫn giữ nguyên biểu cảm và tư thế.
Input: ảnh/video nguồn (identity) và ảnh/video đích (pose), output: khuôn mặt đích với identity của
nguồn. Deepfake thường hiểu là hoán đổi khuôn mặt tự động bằng DL.
Bộ dữ liệu: FaceForensics++ (1339 video gốc và video bị giả mạo 4 phương pháp) . Celeb-DF (590
video thật và deepfake tổng hợp) dùng đánh giá fake vs real. DFDC (Facebook Deepfake Challenge)
4700 video. Đánh giá: SSIM/PSNR cho ảnh, FID score, và phổ biến nhất là đánh giá bằng các mô hình
nhận diện (mức độ nạn nhân bị nhận dạng lầm như thật).
Mô hình tiêu biểu: Các Framework GAN/Autoencoder: nổi bật là DeepFakes (2017) – mã nguồn mở đầu
tiên, FSGAN (Nirkin et al. 2019) sử dụng Inpainting đa giai đoạn, FaceSwap (kỹ thuật GAN cơ bản). Gần
đây có DiffusionFace (Nguyen 2022) dùng diffusion models và StyleGAN2-based (Nirkin CVPR’21). Các
mô hình thường học không gian liên tục của khuôn mặt để tách thông tin danh tính và biểu cảm. Tiền
xử lý: định vị khuôn mặt 2D/3D, ánh xạ bản đồ đánh dấu. Hậu xử lý: ghép nối, chỉnh màu, tổng hợp mịn
viền (ví dụ Poisson blending). Thách thức: xử lý tốt khi màu da, ánh sáng giữa nguồn và đích khác biệt,
artifact ở biên khuôn mặt (Edge artifacts). Công nghệ deepfake ngày càng dễ thực hiện nên các phương
pháp phát hiện (Deepfake detection) cũng trở nên quan trọng để đối phó

### 2.7. Chống giả mạo (Anti-Spoofing / Presentation Attack Detection)

Định nghĩa: Xác định liệu một ảnh/video khuôn mặt có phải là ảnh/thiết bị giả mạo (được trình chiếu)
hay không. Input: ảnh/video khuôn mặt, output: real/live vs fake.
Bộ dữ liệu:  CASIA-FASD (600 video: printing/màn hình 2D, 3 mức chất lượng)¹, Replay-Attack (1.300
video, nhiều biến thể scan), MSU-MFSD (280 video). OULU-NPU (10.000 ảnh từ các hình thức giả mạo
khác nhau), HiFiMask, CelebA-Spoof (625k ảnh nhiều hiệu ứng). Metrics: APCER/BPCER (Attack/Bona fide
classification error rate) hoặc ACER = (APCER+BPCER)/2, Accuracy.
Mô hình tiêu biểu: Phương pháp xưa: LBP/CNN phân lớp nhị phân (rõ nhất là CNN trực tiếp trên ảnh).
Mạng mới: AUXiliary (Li et al. 2018) đưa nhãn độ sâu/texture, CDCN (Yu et al. 2020) sử dụng mạng học
tập texture. RNN/CNN cho video (như Triplet loss). Ngoài ra sử dụng thông tin phản xạ (ví dụ PPG
signals   qua   video)   để   phát   hiện   sống.   Tiền   xử   lý:   thường   crop   vùng   khuôn   mặt;   có   thể   dùng
augmentation các giả mạo. Thách thức: thật–giả ngày càng khắt khe (deepfake cận cảnh), nên các mô
hình phải phát hiện các sai sót vi mô (ví dụ đốm nhiễu, tần số khác biệt) và đảm bảo generalize giữa các
kiểu tấn công

### 3. Các thách thức khi giải các bài toán trong face ecosystem

Thiên lệch dữ liệu và Fairness: Dữ liệu khuôn mặt thường ưu tiên người da trắng và nam giới
, dẫn đến bất bình đẳng trong độ chính xác (Buolamwini & Gebru 2018). Nghiên cứu mới
(NIST FRVT) cho thấy sai số khác biệt lớn giữa các nhóm da đen/tất/tẻ. Có xu hướng dùng điều
chỉnh dữ liệu (oversample nhóm thiểu số) hoặc loss có trọng số để giảm bias. 
Riêng tư: Cần tuân thủ quy định GDPR và quan tâm đến việc lưu trữ ảnh khuôn mặt (Face
recognition là nhạy cảm). Deepfake/hoán đổi khuôn mặt làm gia tăng mối lo ngại bảo mật (tin
giả, phishing) . Phát triển kỹ thuật privacy-preserving (feature encoding, federated learning)
ngày càng quan trọng. 
• Khó khăn ảnh (occlusion/pose/lighting): Hầu hết mô hình bị giảm hiệu năng khi khuôn mặt
che nửa hoặc ngửa, hoặc trong điều kiện ánh sáng kém. Cần data augmentation (tạo khuôn mặt
che, ánh sáng giả lập) và robust learning (phòng thủ adversarial). 
Domain adaptation: Chênh lệch giữa ảnh minh họa (studio) và ảnh thực tế (webcam) đòi hỏi
dùng huấn luyện chéo domain (fine-tune trên trường mới), học không giám sát (few-shot). GAN
được dùng để chuyển phong cách ảnh (style transfer). 
Dữ liệu tổng hợp: GAN và diffusion models tạo dữ liệu khuôn mặt không tốn công thu thập (Ví
dụ DCFace nhân danh tính mới) . Tuy có thể mở rộng đa dạng nhưng cần kiểm tra tính xác
thực (độ thật) của ảnh tạo. Nghiên cứu như FRCSyn (2024) đánh giá nhận dạng trên dữ liệu tổng
hợp. 
Học tự giám sát (Self-supervised): Bài toán khuôn mặt có thể tận dụng SSL (ví dụ MoCo,
SimCLR) trên ảnh không gắn nhãn để học embedding tổng quát, rồi fine-tune cho nhận dạng.
Các nghiên cứu mới phát triển contrastive loss đặc thù khuôn mặt (ArcFace, CosFace cũng là
dạng học metric). 
Hiệu năng và thời gian thực: Ứng dụng thực tế (camera giám sát, điện thoại) đòi hỏi các mô
hình nhẹ và xử lý nhanh. Mạng như MobileFaceNet, GhostNet,... đã ra đời để cân bằng độ chính
xác và tốc độ. Cải tiến phần cứng (NPU) cũng hỗ trợ triển khai nhanh. 
Đạo đức và pháp lý: Nhận diện khuôn mặt trong giám sát công cộng bị tranh cãi. Nhiều nơi (Mỹ,
EU) đã ban hoặc hạn chế sử dụng FRT. Các vấn đề pháp lý (như tuân thủ quyền riêng tư, minh
bạch thuật toán) là thách thức quan trọng khi triển khai hệ thống nhận diện khuôn mặt

1.  **PIE (Pose, Illumination, Expression):** Độ chính xác giảm khi khuôn mặt ở các góc nghiêng khác nhau, ánh sáng quá tối/quá sáng, hoặc có biểu cảm lạ.
2.  **Occlusion (Che khuất):** Khó nhận dạng khi khuôn mặt bị che bởi khẩu trang, kính râm, tóc,...
