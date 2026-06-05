---
time: 06/20/2022
title: Trực quan hoá mô hình CNN (CNN Visualization)
description: Trực quan hoá mô hình CNN (CNN Visualization hay Model Explainability) là lĩnh vực nghiên cứu và kỹ thuật nhằm hiểu được những gì một mạng nơ-ron tích chập thực sự "nhìn thấy" và "học được". Thay vì chấp nhận mô hình như một hộp đen, các phương pháp visualization cho phép nhà nghiên cứu xác định vùng ảnh nào ảnh hưởng đến quyết định của mô hình, kernel nào phản ứng với đặc trưng gì, và tại sao mô hình lại đưa ra một dự đoán cụ thể. Đây là nền tảng của Explainable AI (XAI), không thể thiếu trong các ứng dụng đòi hỏi minh bạch như y tế, tài chính và pháp lý.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: true
---

## 1. Giới thiệu về trực quan hoá mô hình CNN

Khi phát triển các mô hình machine learning, đặc biệt là các mạng CNN sâu, chúng ta mong muốn biết được model đang học tốt ở phần nào và chưa tốt ở phần nào.
**Trực quan hoá mô hình (Model Visualization hay Explainability)** là tập hợp các kỹ thuật giúp "mở hộp đen" của mạng nơ-ron, cho phép chúng ta nhìn vào bên trong để hiểu cơ chế ra quyết định.

Khác với các bài toán computer vision khác (phân loại, phát hiện, nhận dạng), visualization không có đầu ra dạng nhãn hay hộp bao — đầu ra là **heatmap, ảnh, biểu đồ** giúp con người hiểu được mô hình.
Điều này đặt ra thách thức riêng: làm sao đánh giá được "lời giải thích" tốt hay xấu?

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/visualization_overview.jpeg" style="width: 1000px;"/>

Vai trò chính của model visualization là giúp chúng ta hiểu rõ quyết định của mô hình, giúp giải thích được lý do tại sao mô hình lại đưa ra quyết định đó.
Từ đó, ta sẽ xác định được đâu là những điểm mạnh mà mô hình đã đạt được và đâu là những điểm yếu của mô hình và cải thiện từ đó.

Ứng dụng của model visualization trải rộng trên nhiều lĩnh vực:
- **Gỡ lỗi mô hình (Model debugging):** Phát hiện mô hình đang "gian lận" — học các đặc trưng tắt lối (shortcut features) không phải nguyên nhân thực sự. Ví dụ kinh điển: mô hình phân loại sói vs. chó husky thực ra học theo nền tuyết phủ thay vì hình dạng con vật.
- **Y tế và chẩn đoán ảnh (Medical Imaging):** Giúp bác sĩ tin tưởng AI — heatmap cho thấy mô hình nhìn vào đúng vùng khối u, không phải artefact của máy chụp.
- **Pháp lý và kiểm toán (Audit / Compliance):** EU AI Act và GDPR yêu cầu "quyền được giải thích" — doanh nghiệp phải giải trình tại sao AI từ chối một đơn vay hoặc hồ sơ xin việc.
- **Nghiên cứu kiến trúc:** So sánh xem ViT hay CNN "nhìn" ảnh khác nhau như thế nào để có quyết định thiết kế tốt hơn.
- **Phát hiện thiên lệch (Bias detection):** Kiểm tra xem mô hình nhận diện khuôn mặt có đang chú ý vào màu da hay phụ kiện không liên quan.

### Phân loại các phương pháp visualization

Có hai chiều phân loại chính cho các phương pháp visualization:

**Theo thời điểm áp dụng:**
- **Trực quan hoá trong quá trình huấn luyện (during training):** Scalars (loss, accuracy, learning rate), Histograms (phân phối weight/gradient), Images (ảnh đầu vào và dự đoán). Công cụ điển hình: **TensorBoard**, **Weights & Biases (WandB)**.
- **Trực quan hoá sau khi huấn luyện (post-hoc):** Phân tích mô hình đã được huấn luyện để giải thích dự đoán cụ thể. Đây là nhóm phong phú và quan trọng nhất về mặt nghiên cứu.

**Theo phạm vi giải thích:**
- **Global explanation:** Giải thích toàn bộ mô hình học được gì (ví dụ: kernel ở lớp 1 phản ứng với cạnh ngang, cạnh dọc...).
- **Local explanation:** Giải thích tại sao mô hình dự đoán như vậy cho **một ảnh cụ thể** (ví dụ: vùng nào trong ảnh con mèo này khiến model dự đoán "mèo").

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/visualization_taxonomy.jpeg" style="width: 1000px;"/>

Trong các phần tiếp theo, ta sẽ đi từ nhóm đơn giản nhất (trực quan hoá trong quá trình huấn luyện) đến nhóm phức tạp nhất (perturbation-based), theo hành trình từ hiểu mô hình ở mức thô đến mức tinh tế.

## 2. Nhóm các phương pháp trực quan hoá mô hình CNN

Ta sẽ tổ chức các phương pháp theo **mức độ phân tích**: bắt đầu từ theo dõi quá trình học, rồi đến trực quan hoá kiến trúc bên trong, tiếp theo là nhóm dùng gradient để sinh heatmap, và cuối cùng là perturbation-based — nhóm model-agnostic hiện đại nhất.

### 2.1. Trực quan hoá trong quá trình huấn luyện

Đây là nhóm trực quan hoá **thực dụng nhất** và được mọi practitioner sử dụng hàng ngày. Mục tiêu không phải là giải thích tại sao mô hình ra quyết định, mà là **theo dõi quá trình học** để phát hiện sớm các vấn đề như overfitting, gradient vanishing/exploding, hay dead neurons.

#### Mô tả ý tưởng và cơ chế hoạt động

Nhóm này gồm ba loại chính, thường được hiển thị qua các công cụ như **TensorBoard** hoặc **Weights & Biases (WandB)**:

**Scalars (Vô hướng):** Ghi lại và vẽ đồ thị các giá trị vô hướng theo từng bước huấn luyện — `loss` (training và validation), `accuracy`, `learning rate`, `gradient norm`. Nhìn vào đường cong loss giúp ta phát hiện: overfitting (train loss giảm nhưng val loss tăng), underfitting (cả hai đều cao và không giảm), hay learning rate không phù hợp (loss dao động mạnh hoặc không hội tụ).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/training_curves.jpeg" style="width: 900px;"/>

**Histograms (Biểu đồ phân phối):** Ghi lại phân phối của các trọng số (weights), gradient, và activation theo từng epoch. Đây là công cụ mạnh để phát hiện:
- **Gradient vanishing:** Histogram gradient của các lớp đầu gần bằng 0 — mô hình không học được ở lớp đầu.
- **Gradient exploding:** Histogram gradient có giá trị rất lớn — cần gradient clipping.
- **Dead neurons:** Histogram activation của một lớp ReLU toàn bằng 0 — neurons không bao giờ kích hoạt.
- **Weight collapse:** Tất cả weights hội tụ về cùng một giá trị — mô hình gặp symmetry breaking failure.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/histogram_weights_gradients.jpeg" style="width: 900px;"/>

**Images (Ảnh):** Log trực tiếp ảnh đầu vào, ảnh augmentation, ảnh dự đoán của mô hình vào TensorBoard để kiểm tra bằng mắt. Đặc biệt hữu ích để đảm bảo augmentation không làm hỏng ảnh, và nhìn thấy mô hình đang nhầm những ảnh nào.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Không tốn chi phí thêm:** Tất cả thông tin đã có sẵn trong quá trình training — chỉ cần log và hiển thị.
- **Phát hiện vấn đề sớm:** Cho phép dừng training sớm (early stopping) và điều chỉnh hyperparameter kịp thời.
- **Dễ tích hợp:** TensorBoard hỗ trợ TensorFlow/PyTorch; WandB thêm collaboration và cloud sync.

**Nhược điểm:**
- **Không giải thích quyết định:** Chỉ cho biết mô hình đang học như thế nào, không giải thích tại sao mô hình dự đoán như vậy với một ảnh cụ thể.
- **Quá nhiều thông tin có thể gây nhiễu:** Với mô hình hàng trăm layer, số lượng histogram khổng lồ khó theo dõi hết.

#### Một số công cụ tiêu biểu trong nhóm

- **TensorBoard (Google, 2015)** — [link](https://www.tensorflow.org/tensorboard) — công cụ visualization tích hợp của TensorFlow, được PyTorch và Lightning hỗ trợ đầy đủ.
- **Weights & Biases / WandB (2018)** — [link](https://wandb.ai) — nền tảng MLOps cloud, bổ sung experiment tracking, hyperparameter sweeps và model versioning.
- **MLflow (Databricks, 2018)** — [link](https://mlflow.org) — mã nguồn mở, mạnh về quản lý experiment và model registry.

### 2.2. Trực quan hoá Kernel và Feature Map

Sau khi mô hình được huấn luyện xong, câu hỏi đầu tiên là: **các bộ lọc (kernel) đã học được những gì?** Và khi một ảnh đi qua mạng, **mỗi lớp "nhìn thấy" gì?** Nhóm này trả lời hai câu hỏi đó bằng cách hiển thị trực tiếp trọng số kernel và giá trị feature map.

#### Mô tả ý tưởng và cơ chế hoạt động

**Trực quan hoá Kernel:** Hiển thị ma trận trọng số của mỗi bộ lọc dưới dạng hình ảnh, trong đó giá trị pixel biểu thị cường độ của trọng số. Với lớp conv đầu tiên (3 kênh đầu vào RGB), ta có thể hiển thị kernel trực tiếp như ảnh RGB — và quan sát thấy chúng học được các bộ dò cạnh (edge detectors), bộ dò góc (corner detectors) hay bộ dò màu sắc. Những pattern này rất giống với bộ lọc Gabor, điều thú vị là CNN tự học được đặc trưng tương tự những gì con người thiết kế thủ công trong thời kỳ pre-deep learning.

Với các lớp sâu hơn (nhiều kênh đầu vào), kernel không còn có ý nghĩa thị giác trực tiếp — ta cần dùng Activation Maximization (mục 2.3) để hiểu chúng đang học gì.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/kernel_visualization.jpeg" style="width: 800px;"/>

**Trực quan hoá Feature Map:** Với một ảnh đầu vào cụ thể, ta lấy activation của từng kênh tại mỗi lớp conv và hiển thị như ảnh grayscale. Pattern quan sát được theo độ sâu:
- **Lớp đầu (low-level):** Feature map phản ánh các đặc trưng thô — cạnh ngang/dọc, màu sắc, góc. Nhìn vẫn còn giống ảnh gốc.
- **Lớp giữa (mid-level):** Feature map trừu tượng hơn — kết cấu, hình dạng cục bộ.
- **Lớp sâu (high-level):** Feature map rất thưa (sparse), chỉ sáng ở một vài điểm — mô hình đã trích xuất đặc trưng ngữ nghĩa cao như "mắt", "mõm", "đường viền con vật".

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/feature_map_layers.jpeg" style="width: 900px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **Trực tiếp và trực quan:** Không cần thuật toán phức tạp — chỉ cần forward pass và lưu activation.
- **Global explanation mạnh:** Trực quan hoá kernel cho thấy mô hình học được gì một cách toàn cục, không phụ thuộc ảnh đầu vào.
- **Hỗ trợ debug kiến trúc:** Dead filters (kernel toàn màu xám đều) là dấu hiệu lãng phí capacity, giúp cắt giảm số kernel.

**Nhược điểm:**
- **Lớp sâu khó interpret:** Kernel ở lớp 5+ có hàng nghìn kênh đầu vào, không có ý nghĩa thị giác trực tiếp.
- **Feature map phụ thuộc ảnh đầu vào:** Phải chọn ảnh đại diện, khó đưa ra kết luận chung.
- **Không trả lời "tại sao" cho từng quyết định:** Chỉ cho biết "mô hình phản ứng gì", không liên kết trực tiếp với nhãn dự đoán.

#### Một số phương pháp / công cụ tiêu biểu trong nhóm

- **AlexNet filter visualization (Krizhevsky et al., 2012)** — [paper](https://arxiv.org/abs/1404.5997) — bài báo đầu tiên hệ thống hoá việc hiển thị kernel lớp 1 của CNN, cho thấy chúng học Gabor-like patterns.
- **ZFNet / Deconvnet (Zeiler & Fergus, 2014)** — [paper](https://arxiv.org/abs/1311.1901) — đề xuất "deconvolution" để chiếu activation ngược về không gian pixel, lần đầu giải thích được feature map ở lớp sâu.
- **Torchvision hooks** và **Keras `Model.get_layer()`** — API chuẩn trong PyTorch/Keras để trích feature map tại bất kỳ lớp nào bằng forward hooks.

### 2.3. Activation Maximization và DeepDream

Nếu kernel visualization chỉ hiển thị trọng số thô, **Activation Maximization** đặt câu hỏi ngược lại: *ảnh đầu vào nào khiến một neuron/lớp/class cụ thể hoạt động mạnh nhất?* Câu trả lời sinh ra những ảnh kỳ ảo nhưng rất có giá trị về mặt khoa học.

#### Mô tả ý tưởng và cơ chế hoạt động

Thay vì tối ưu tham số của mô hình, ta **cố định mô hình và tối ưu ảnh đầu vào** để tối đa hoá một mục tiêu nào đó (activation của một neuron, score của một class). Bắt đầu từ ảnh nhiễu ngẫu nhiên $x_0$, ta lặp gradient ascent:

$$x_{t+1} = x_t + \alpha \cdot \frac{\partial f(x_t)}{\partial x_t}$$

trong đó $f$ là giá trị cần tối đa hoá (activation của neuron mục tiêu hoặc class score), và $\alpha$ là learning rate. Để ảnh tổng hợp trông tự nhiên hơn (không phải nhiễu tần số cao), người ta thêm regularization: **L2 regularization** trên pixel, **Gaussian blur** sau mỗi bước, hoặc học trong không gian tần số thấp.

**DeepDream (Google, 2015)** là ứng dụng nổi tiếng nhất của Activation Maximization: thay vì bắt đầu từ nhiễu, ta bắt đầu từ một ảnh thực và tối đa hoá activation của một lớp — tạo ra hiệu ứng "ảo giác" kỳ ảo khi mô hình "thấy" chó, mắt và chim ở khắp nơi trong ảnh.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/activation_maximization.jpeg" style="width: 900px;"/>

Một biến thể hữu ích là **Feature Inversion (Mahendran & Vedaldi, 2015)**: tìm ảnh $x^*$ sao cho feature của nó gần nhất với feature của một ảnh tham chiếu. Kết quả cho thấy các lớp sâu lưu giữ thông tin gì: lớp đầu giữ được chi tiết pixel, lớp sâu chỉ giữ cấu trúc ngữ nghĩa — mất hoàn toàn thông tin màu sắc và vị trí chính xác.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Global explanation mạnh mẽ:** Cho thấy trực quan concept mà từng neuron biểu diễn, không phụ thuộc ảnh đầu vào cụ thể.
- **Phát hiện shortcut learning:** Nếu ảnh tổng hợp chứa texture kỳ lạ không có nghĩa ngữ nghĩa, nghĩa là mô hình học shortcut.
- **Nghiên cứu kiến trúc:** So sánh ảnh maximize của ViT vs CNN cho thấy transformer "nhìn" theo cách khác (global vs local features).

**Nhược điểm:**
- **Không realistic:** Ảnh tổng hợp thường chứa nhiều texture lặp lại, trông không tự nhiên với người.
- **Optimization không ổn định:** Kết quả phụ thuộc nhiều vào seed, regularization, learning rate — khó tái tạo.
- **Chỉ giải thích neuron đơn lẻ:** Khó áp dụng để giải thích quyết định cho một ảnh cụ thể.

#### Một số phương pháp tiêu biểu trong nhóm

- **Activation Maximization (Erhan et al., 2009)** — [paper](https://www.researchgate.net/publication/265022827) — bài báo nền tảng đề xuất tối ưu ảnh để hiểu neuron.
- **DeepDream (Mordvintsev et al., Google, 2015)** — [blog](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) — tạo ra ảnh ảo giác từ mô hình phân loại, viral toàn thế giới.
- **Feature Inversion (Mahendran & Vedaldi, 2015)** — [paper](https://arxiv.org/abs/1412.0035) — tái tạo ảnh từ đặc trưng để đo lượng thông tin được lưu giữ ở mỗi lớp.
- **Neural Style Transfer (Gatys et al., 2015)** — [paper](https://arxiv.org/abs/1508.06576) — ứng dụng feature inversion để chuyển phong cách nghệ thuật giữa hai ảnh, đặt nền cho loạt ứng dụng style transfer.

### 2.4. Class Activation Mapping (CAM) và các biến thể

Đây là nhóm phương pháp quan trọng nhất và phổ biến nhất trong **local explanation** — trả lời câu hỏi: *"vùng nào trong ảnh này khiến mô hình dự đoán class X?"* Đầu ra là một **heatmap** (bản đồ nhiệt) có cùng kích thước với ảnh đầu vào, mỗi pixel mang giá trị phản ánh mức độ quan trọng của vùng đó đối với quyết định.

#### Mô tả ý tưởng và cơ chế hoạt động

**CAM — Class Activation Mapping (Zhou et al., 2016)**

CAM là phương pháp đầu tiên sinh heatmap có chất lượng tốt từ CNN phân loại. Ý tưởng: thay thế lớp Fully Connected (FC) cuối bằng **Global Average Pooling (GAP)** kết hợp với một lớp FC nhỏ. Khi đó, score của class $c$ là:

$$S^c = \sum_k w_k^c \cdot \frac{1}{Z} \sum_{i,j} f_k(i,j) = \frac{1}{Z} \sum_{i,j} \underbrace{\sum_k w_k^c \cdot f_k(i,j)}_{\text{CAM}_c(i,j)}$$

Heatmap CAM cho class $c$ tại vị trí $(i,j)$:

$$\text{CAM}_c(i,j) = \sum_k w_k^c \cdot f_k(i,j)$$

trong đó $w_k^c$ là trọng số của lớp FC ứng với class $c$ và feature map $k$, và $f_k(i,j)$ là giá trị feature map thứ $k$ tại vị trí $(i,j)$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/cam_architecture.jpeg" style="width: 900px;"/>

**Hạn chế của CAM:** Yêu cầu kiến trúc mạng phải có GAP ngay trước lớp phân loại. Không áp dụng được cho VGG, AlexNet gốc, hay các mô hình có nhiều lớp FC liên tiếp. Ngoài ra, độ phân giải heatmap bằng resolution của feature map cuối (thường 7×7 hoặc 14×14 trước khi upsample), nên khá thô.

**Grad-CAM — Gradient-weighted CAM (Selvaraju et al., 2017)**

Grad-CAM tổng quát hoá CAM để áp dụng cho **mọi kiến trúc CNN** mà không cần thay đổi cấu trúc. Thay vì dùng trọng số FC, Grad-CAM tính "tầm quan trọng" của mỗi feature map bằng **gradient của class score theo feature map**:

$$\alpha_k^c = \underbrace{\frac{1}{Z} \sum_{i,j}}_{\text{global average pool}} \frac{\partial y^c}{\partial A^k_{i,j}}$$

trong đó $y^c$ là score logit của class $c$ và $A^k$ là feature map thứ $k$ của lớp conv được chọn. Heatmap Grad-CAM:

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c \cdot A^k\right)$$

ReLU được áp dụng vì chỉ quan tâm đến vùng **làm tăng** class score, không phải vùng ức chế.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/gradcam_formula.jpeg" style="width: 1000px;"/>

**Các biến thể nâng cao:**

| Phương pháp | Cải tiến chính |
|---|---|
| **Grad-CAM++ (Chattopadhyay, 2018)** | Dùng weighted average của gradient bậc cao hơn — xử lý tốt hơn khi class xuất hiện nhiều lần trong ảnh (multi-instance) |
| **Score-CAM (Wang, 2020)** | Không dùng gradient — dùng activation làm mask, đo sự thay đổi score → gradient-free, ít nhiễu hơn |
| **Eigen-CAM (Muhammad, 2020)** | Dùng PCA thay vì gradient — nhanh hơn Score-CAM, không cần backward pass |
| **XGrad-CAM (Fu, 2020)** | Chuẩn hoá gradient theo activation → heatmap sắc nét và faithfulness tốt hơn |
| **LayerCAM (Jiang, 2021)** | Tổng hợp heatmap từ nhiều lớp conv khác nhau — chi tiết hơn ở cả local lẫn global level |

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/gradcam_variants.jpeg" style="width: 1000px;"/>

#### Ứng dụng điển hình

- **Medical imaging:** Heatmap Grad-CAM cho thấy mô hình nhìn vào đúng vùng bất thường trong ảnh X-quang ngực (khối u, viêm phổi) — giúp bác sĩ kiểm tra và tin tưởng AI.
- **Weakly-supervised localization:** Dùng CAM để xác định vùng vật thể chỉ từ nhãn cấp ảnh, không cần bounding box trong training.
- **Bias detection:** Phát hiện mô hình dựa vào watermark, copyright notice, hay background thay vì nội dung thực sự.
- **Debugging misclassification:** Khi mô hình phân loại sai, heatmap giúp thấy lý do — ví dụ model nhìn vào hàng rào sau con chó thay vì con chó.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Dễ implement, nhanh:** Chỉ cần một backward pass thêm (Grad-CAM); CAM chỉ cần forward pass.
- **Áp dụng rộng:** Grad-CAM dùng được với mọi CNN có lớp conv — không yêu cầu GAP như CAM gốc.
- **Heatmap có ý nghĩa ngữ nghĩa:** Vùng highlight thực sự liên quan đến class, không phải artifact ngẫu nhiên.

**Nhược điểm:**
- **Độ phân giải thấp:** Heatmap bị giới hạn bởi resolution của feature map cuối (thường 7×7) — mịn khi upsample nhưng mất chi tiết nhỏ.
- **Nhiễu gradient:** Ở mạng rất sâu, gradient bị nhiễu do nhiều phép nhân liên tiếp, heatmap có thể không ổn định.
- **Phụ thuộc lớp được chọn:** Kết quả thay đổi đáng kể tùy vào việc chọn lớp conv nào để tính — không có quy tắc cứng.

#### Một số mô hình / thư viện tiêu biểu

- **CAM (Zhou et al., 2016)** — [paper](https://arxiv.org/abs/1512.04150) — Learning Deep Features for Discriminative Localization.
- **Grad-CAM (Selvaraju et al., 2017)** — [paper](https://arxiv.org/abs/1610.02391) — Visual Explanations from Deep Networks via Gradient-based Localization.
- **Grad-CAM++ (Chattopadhyay et al., 2018)** — [paper](https://arxiv.org/abs/1710.11063) — xử lý multi-instance và partial occlusion tốt hơn.
- **Score-CAM (Wang et al., 2020)** — [paper](https://arxiv.org/abs/1910.01279) — gradient-free, ít bị ảnh hưởng bởi gradient noise.
- **pytorch-grad-cam** — [GitHub](https://github.com/jacobgil/pytorch-grad-cam) — thư viện PyTorch phổ biến nhất, implement toàn bộ CAM family (Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM, LayerCAM...).

### 2.5. Gradient-based Saliency Methods

Trong khi CAM/Grad-CAM làm việc ở **mức feature map** (độ phân giải thấp), nhóm **Gradient-based Saliency** làm việc ở **mức pixel** — tính trực tiếp đạo hàm của class score theo từng pixel đầu vào. Kết quả là **saliency map** có cùng kích thước ảnh đầu vào, từng pixel cho biết mức độ ảnh hưởng của nó đến quyết định của mô hình.

#### Mô tả ý tưởng và cơ chế hoạt động

**Vanilla Gradient / Saliency Map (Simonyan et al., 2013)**

Phương pháp đơn giản nhất: tính gradient của class score $y^c$ theo ảnh đầu vào $x$, rồi lấy giá trị tuyệt đối:

$$S(x) = \left|\frac{\partial y^c}{\partial x}\right|$$

Giá trị gradient lớn tại pixel $(i,j)$ nghĩa là thay đổi nhỏ tại pixel đó làm thay đổi lớn đến đầu ra — pixel đó quan trọng. **Nhược điểm:** Gradient thường rất nhiễu (noisy), đặc biệt ở mạng sâu do phép nhân liên tiếp của nhiều layer.

**Guided Backpropagation (Springenberg et al., 2014)**

Cải tiến Vanilla Gradient: khi lan truyền gradient qua lớp ReLU ngược chiều, chỉ giữ lại **gradient dương** và **neuron dương** (tắt đường truyền của cả hai loại gradient âm). Heuristic này giúp saliency map sắc nét hơn, ít nhiễu hơn, nhưng về lý thuyết không còn "faithfully" biểu diễn đóng góp thực sự của pixel mà chỉ phản ánh ảnh đã được khử nhiễu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/saliency_map_comparison.jpeg" style="width: 1000px;"/>

**Integrated Gradients (Sundararajan et al., 2017)**

Thay vì chỉ lấy gradient tại một điểm $x$, Integrated Gradients tích phân gradient dọc theo đường thẳng từ **baseline** $\bar{x}$ (thường là ảnh đen — tất cả pixel bằng 0) đến ảnh đầu vào $x$:

$$\text{IG}_i(x) = (x_i - \bar{x}_i) \cdot \int_{\alpha=0}^{1} \frac{\partial F(\bar{x} + \alpha(x - \bar{x}))}{\partial x_i} \, d\alpha$$

trong đó $x_i$ là giá trị pixel thứ $i$, và tích phân được xấp xỉ bằng trung bình Riemann qua $m$ bước (thường $m = 50$). **Hai tính chất quan trọng:** Integrated Gradients thỏa mãn axiom **Completeness** (tổng attribution bằng đúng $F(x) - F(\bar{x})$) và **Sensitivity** (nếu feature thay đổi làm đầu ra thay đổi thì attribution ≠ 0) — đảm bảo explanation là faithful theo nghĩa toán học.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/integrated_gradients.jpeg" style="width: 900px;"/>

**SmoothGrad (Smilkov et al., 2017)**

Giải quyết vấn đề nhiễu của Vanilla Gradient bằng cách lấy trung bình gradient qua nhiều ảnh đầu vào đã thêm **Gaussian noise**:

$$\tilde{S}(x) = \frac{1}{n} \sum_{i=1}^{n} S\left(x + \mathcal{N}(0, \sigma^2)\right)$$

Đơn giản nhưng hiệu quả: trung bình làm giảm nhiễu ngẫu nhiên và giữ lại tín hiệu ổn định. SmoothGrad có thể kết hợp với bất kỳ saliency method nào khác (SmoothGrad + Integrated Gradients cho kết quả rất mịn).

#### Ưu và nhược điểm

**Ưu điểm:**
- **Độ phân giải pixel:** Saliency map cùng kích thước ảnh đầu vào, không bị giới hạn bởi feature map resolution như CAM.
- **Integrated Gradients faithful:** Thỏa mãn axioms lý thuyết, đảm bảo attribution có ý nghĩa toán học rõ ràng.
- **Nhanh:** Chỉ cần một (hoặc $m$) backward pass, không cần thay đổi kiến trúc.

**Nhược điểm:**
- **Vanilla Gradient nhiễu:** Không đủ ổn định để tin cậy cho ứng dụng thực tế.
- **Sensitivity to baseline (IG):** Kết quả thay đổi theo baseline được chọn — không có chuẩn nào "đúng" tuyệt đối.
- **Tốn tính toán (SmoothGrad):** Cần $n$ backward pass ($n = 50$ điển hình) — gấp $n$ lần Vanilla Gradient.
- **Dễ bị manipulate:** Các nghiên cứu cho thấy saliency map có thể bị thay đổi hoàn toàn mà không ảnh hưởng đến prediction — đặt câu hỏi về faithfulness thực tế.

#### Một số mô hình / thư viện tiêu biểu

- **Saliency Map / Vanilla Gradient (Simonyan et al., 2013)** — [paper](https://arxiv.org/abs/1312.6034) — bài báo nền tảng giới thiệu gradient-based saliency cho CNN.
- **Guided Backpropagation (Springenberg et al., 2014)** — [paper](https://arxiv.org/abs/1412.6806) — Striving for Simplicity: The All Convolutional Net.
- **Integrated Gradients (Sundararajan et al., 2017)** — [paper](https://arxiv.org/abs/1703.01365) — Axiomatic Attribution for Deep Networks, được Google áp dụng rộng rãi trong production.
- **SmoothGrad (Smilkov et al., 2017)** — [paper](https://arxiv.org/abs/1706.03825) — Removing Noise by Adding Noise.
- **Captum (Facebook/Meta, 2019)** — [GitHub](https://captum.ai) — thư viện PyTorch tổng hợp implement Integrated Gradients, Guided Backprop, GradCAM, SHAP và nhiều phương pháp khác trong một API thống nhất.

### 2.6. Perturbation-based Methods

Thay vì dùng gradient, nhóm này áp dụng **nguyên lý đơn giản hơn**: che/xóa bỏ một vùng ảnh rồi đo mức thay đổi của đầu ra. Nếu che vùng nào mà score giảm mạnh → vùng đó quan trọng. Đây là phương pháp **model-agnostic** — không cần biết kiến trúc bên trong, có thể áp dụng cho mọi mô hình như một hộp đen.

#### Mô tả ý tưởng và cơ chế hoạt động

**Occlusion Sensitivity (Zeiler & Fergus, 2014)**

Phương pháp brute-force đầu tiên: trượt một cửa sổ vuông (patch) màu xám hoặc đen qua ảnh đầu vào, tại mỗi vị trí $(i,j)$ ghi lại class score. Nơi nào score giảm mạnh nhất khi bị che → đó là vùng quan trọng. Kết quả tổng hợp thành **occlusion sensitivity map**:

$$M(i,j) = f^c(x) - f^c\left(x \text{ với patch che tại } (i,j)\right)$$

**Nhược điểm:** Chi phí tính toán $O(H \times W)$ forward pass — chậm với ảnh lớn và patch nhỏ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/occlusion_sensitivity.jpeg" style="width: 900px;"/>

**LIME — Local Interpretable Model-agnostic Explanations (Ribeiro et al., 2016)**

LIME không phân tích toàn bộ ảnh mà **xấp xỉ cục bộ** mô hình phức tạp bằng một mô hình tuyến tính đơn giản xung quanh ảnh đầu vào $x$:

1. **Phân đoạn ảnh thành superpixels** (các vùng liền kề màu tương đồng, thường dùng SLIC).
2. **Tạo nhiều biến thể**: bật/tắt ngẫu nhiên từng superpixel (che đi hoặc giữ nguyên) → thu được tập $\{(z_i, f(z_i))\}$ với $z_i$ là vector nhị phân biểu diễn superpixel nào được giữ.
3. **Fit linear model** với trọng số lớn hơn cho mẫu gần $x$ hơn: $\xi(x) = \arg\min_g \mathcal{L}(f, g, \pi_x)$.
4. **Hệ số tuyến tính** của model fit là attribution cho từng superpixel — dương: ủng hộ class, âm: phản đối class.

**SHAP — SHapley Additive exPlanations (Lundberg & Lee, 2017)**

SHAP dùng nền tảng lý thuyết trò chơi (game theory) — **Shapley values** — để phân bổ đóng góp của từng feature một cách công bằng. Shapley value của feature $i$:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[f(S \cup \{i\}) - f(S)\right]$$

tức là trung bình có trọng số của đóng góp biên của feature $i$ khi thêm vào mọi tập con $S$ của features. SHAP thỏa mãn các axioms quan trọng: **Efficiency** (tổng attribution bằng $f(x) - f(\bar{x})$), **Symmetry**, **Dummy**, và **Additivity**. Với deep learning, **DeepSHAP** và **GradientSHAP** cải tiến tốc độ bằng cách kết hợp SHAP với Integrated Gradients, không cần duyệt $2^{|F|}$ tập con.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/lime_shap_comparison.jpeg" style="width: 900px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **Model-agnostic:** Không cần biết kiến trúc bên trong — áp dụng được cho mọi mô hình (kể cả Random Forest, XGBoost, API đóng).
- **Lý thuyết vững chắc (SHAP):** Nền tảng từ Shapley values đảm bảo các axioms quan trọng, có thể so sánh attribution giữa các features.
- **Trực quan với người dùng cuối (LIME):** Output dạng "vùng nào ủng hộ/phản đối quyết định" dễ hiểu với người không có nền tảng ML.

**Nhược điểm:**
- **Chậm:** Occlusion Sensitivity cần $O(H \times W)$ forward pass; tính Shapley values chính xác tốn $O(2^{|F|})$ — phải dùng xấp xỉ.
- **Không ổn định (LIME):** Kết quả phụ thuộc vào phân đoạn superpixel và sampling ngẫu nhiên — chạy hai lần có thể cho hai kết quả khác nhau.
- **Baseline phụ thuộc (SHAP):** Cần chọn baseline (ảnh đen, trung bình dataset) — ảnh hưởng đáng kể đến kết quả.

#### Một số mô hình / thư viện tiêu biểu

- **Occlusion Sensitivity (Zeiler & Fergus, 2014)** — [paper](https://arxiv.org/abs/1311.1901) — phương pháp perturbation đầu tiên, giới thiệu cùng ZFNet.
- **LIME (Ribeiro et al., 2016)** — [paper](https://arxiv.org/abs/1602.04938) — "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
- **SHAP (Lundberg & Lee, 2017)** — [paper](https://arxiv.org/abs/1705.07874) — A Unified Approach to Interpreting Model Predictions.
- **RISE (Petsiuk et al., 2018)** — [paper](https://arxiv.org/abs/1806.07421) — Randomized Input Sampling for Explanation, perturbation-based thay patch bằng random mask, faithfulness tốt.
- **shap** — [GitHub](https://github.com/shap/shap) — thư viện Python chính thức, hỗ trợ DeepSHAP, GradientSHAP, TreeSHAP.
- **lime** — [GitHub](https://github.com/marcotcr/lime) — thư viện Python chính thức của LIME.

## 3. Các metrics đánh giá phương pháp visualization

Đánh giá chất lượng của một "lời giải thích" (explanation) là thách thức phi thường: không như OCR hay phân loại ảnh, không có ground-truth tuyệt đối cho "heatmap đúng". Các metrics hiện tại đều là **proxy** — đo một khía cạnh của "giải thích tốt" mà không thể bao quát tất cả.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/metrics_overview.jpeg" style="width: 1000px;"/>

### 3.1. Pointing Game

**Pointing Game** là metric đơn giản nhất để đánh giá khả năng **localization** của heatmap — heatmap có chỉ đúng vào vùng đối tượng không?

#### Mô tả ý tưởng và cơ chế hoạt động

Với mỗi ảnh có annotation bounding box (từ dataset như PASCAL VOC, COCO):
1. Tìm điểm **cực đại** của heatmap: $\hat{p} = \arg\max_{(i,j)} M(i,j)$.
2. Kiểm tra: điểm cực đại có nằm **bên trong** bounding box ground-truth không?
3. Tính **Accuracy** = tỷ lệ ảnh có điểm cực đại nằm trong bounding box.

$$\text{Acc}_{\text{point}} = \frac{\#\text{ảnh mà } \hat{p} \in \text{bbox}}{\text{tổng số ảnh}}$$

#### Ví dụ

Trên PASCAL VOC với 20 class: Grad-CAM đạt Pointing Game Accuracy ~67%, Guided Backprop ~75%, Vanilla Gradient ~50%. Điểm cao không nhất thiết nghĩa là explanation tốt nhất — chỉ nghĩa là điểm nóng nhất của heatmap rơi vào vùng đúng.

#### Ưu và nhược điểm

**Ưu điểm:** Đơn giản, có thể tính tự động với mọi dataset có bounding box annotation, không cần user study.

**Nhược điểm:** Chỉ đánh giá một điểm duy nhất — không phản ánh toàn bộ hình dạng heatmap. Một heatmap đúng đỉnh nhưng highlight sai toàn bộ phần còn lại vẫn được điểm tuyệt đối.

### 3.2. Insertion và Deletion Curves

Cặp metric này đánh giá **faithfulness** — heatmap có thực sự phản ánh những gì mô hình dựa vào không?

#### Mô tả ý tưởng và cơ chế hoạt động

**Deletion curve:** Bắt đầu từ ảnh đầy đủ, lần lượt **xóa** các pixel theo thứ tự giảm dần của attribution (pixel quan trọng nhất bị xóa trước, thay bằng màu xám trung bình). Ghi lại class score sau mỗi bước. Nếu heatmap đúng, score phải giảm nhanh ngay khi xóa ít pixel đầu. **Metric: AUC** của đường cong Score-vs-%-pixels-deleted — **càng thấp càng tốt**.

**Insertion curve:** Bắt đầu từ ảnh blur nặng (baseline), lần lượt **thêm lại** pixel theo thứ tự attribution quan trọng nhất trước. Score phải tăng nhanh. **Metric: AUC** — **càng cao càng tốt**.

$$\text{Faithfulness score} = \text{AUC}_{\text{insertion}} - \text{AUC}_{\text{deletion}}$$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/insertion_deletion_curves.jpeg" style="width: 900px;"/>

#### Ví dụ

Trên ImageNet với ResNet-50:

| Phương pháp | Insertion AUC ↑ | Deletion AUC ↓ |
|---|---|---|
| Random baseline | ~0.33 | ~0.33 |
| Grad-CAM | ~0.53 | ~0.18 |
| RISE (perturbation) | ~0.55 | ~0.16 |
| Integrated Gradients | ~0.52 | ~0.19 |

Kết quả cho thấy perturbation-based methods (RISE) đôi khi có faithfulness tốt hơn gradient-based methods.

#### Ưu và nhược điểm

**Ưu điểm:** Đánh giá faithfulness trực tiếp qua ảnh hưởng thực tế lên output — không phụ thuộc vào bounding box annotation hay user study.

**Nhược điểm:** **Out-of-distribution problem** — ảnh với nhiều pixel bị xóa/thêm trở nên rất khác phân phối training, model có thể cho kết quả bất thường không phản ánh chất lượng heatmap thực sự.

### 3.3. Sanity Checks

**Sanity checks (Adebayo et al., 2018)** kiểm tra xem saliency method có thực sự phụ thuộc vào **tham số mô hình** và **nhãn training** không — hay chỉ là biến đổi của ảnh đầu vào bất kể mô hình học được gì.

#### Mô tả ý tưởng và cơ chế hoạt động

Hai bài test chính:

1. **Model parameter randomization:** Lần lượt ngẫu nhiên hoá trọng số của mô hình từ lớp cuối đến lớp đầu. Nếu heatmap không thay đổi khi model bị ngẫu nhiên hoá hoàn toàn → method đó thực ra chỉ detect edge/texture của ảnh, không phản ánh model thực sự.
2. **Data randomization:** Huấn luyện mô hình với nhãn hoàn toàn ngẫu nhiên (random labels). Nếu heatmap không thay đổi so với model trained đúng → explanation không phân biệt được model có ý nghĩa và model vô nghĩa.

**Kết quả đáng lo ngại từ bài báo:** Guided Backpropagation và Guided Grad-CAM **fail** cả hai sanity checks — heatmap gần như giống nhau dù model bị ngẫu nhiên hoá hoàn toàn. Điều này làm dấy lên nghi ngờ về faithfulness của các phương pháp cho output đẹp mắt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/sanity_checks.jpeg" style="width: 900px;"/>

#### Ví dụ

Bài báo của Adebayo et al. (2018) test trên MNIST và ImageNet:
- **Integrated Gradients, Grad-CAM:** Pass sanity checks — heatmap thay đổi rõ khi model bị ngẫu nhiên hoá.
- **Guided Backpropagation, Guided Grad-CAM:** Fail — heatmap nhìn như edge detector bất kể model có được trained hay không.

#### Ưu và nhược điểm

**Ưu điểm:** Không cần ground-truth annotation; là **điều kiện cần tối thiểu** mà mọi visualization method phải pass — loại bỏ các method chỉ detect edge.

**Nhược điểm:** Chỉ là điều kiện cần, không đủ — pass sanity check không đảm bảo explanation là faithful hay hữu ích trong thực tế.

## 4. Các thách thức của bài toán trực quan hoá mô hình CNN

Dù đã có hàng chục phương pháp được đề xuất, lĩnh vực XAI cho CNN vẫn đối mặt những thách thức cơ bản chưa có lời giải thỏa đáng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/12-cnn-visualization/challenges.jpeg" style="width: 1000px;"/>

1. **Faithfulness vs. Aesthetics:** Các phương pháp cho heatmap đẹp nhất (Guided Backprop, SmoothGrad) thường fail sanity checks — tức là trông giải thích tốt nhưng thực ra không phản ánh cơ chế bên trong mô hình. Ngược lại, Occlusion Sensitivity faithful hơn nhưng output thô và nhiễu. Trade-off này chưa được giải quyết triệt để.

2. **Không có ground-truth:** Không ai biết "heatmap đúng" trông như thế nào cho một quyết định cụ thể của model. Mọi metric đều là proxy — Pointing Game chỉ đo một điểm, Insertion/Deletion bị out-of-distribution problem, human evaluation tốn kém và chủ quan. Đây là thách thức nền tảng của toàn bộ lĩnh vực XAI.

3. **Adversarial vulnerability:** Heo et al. (2019) chứng minh có thể thay đổi hoàn toàn heatmap của Grad-CAM, Integrated Gradients trong khi giữ nguyên prediction — chỉ bằng cách thêm perturbation nhỏ không nhìn thấy được. Điều này rất nghiêm trọng nếu XAI được dùng cho audit pháp lý hay y tế.

4. **Giải thích mô hình lớn (ViT, Diffusion):** Attention map của Vision Transformer không tương đương saliency map — các nghiên cứu cho thấy attention không phải explanation (attention không phải explanation). Với mô hình 1B+ params, tính gradient backward tốn VRAM khổng lồ; cần các phương pháp xấp xỉ mới. Transformer-specific methods (Attention Rollout, Transformer Attribution, DINO) đang được phát triển nhưng chưa chuẩn hoá.

5. **Tin tưởng quá mức (Over-trust):** Nghiên cứu tâm lý học cho thấy người dùng có xu hướng tin tưởng quyết định của AI **nhiều hơn** khi được cung cấp heatmap — dù heatmap đó có thể sai. XAI có thể phản tác dụng nếu được thiết kế để tạo cảm giác tin tưởng thay vì truyền đạt uncertainty thực sự.

6. **Áp lực pháp lý và tiêu chuẩn hoá:** EU AI Act (2024) xếp các hệ thống AI rủi ro cao (y tế, tín dụng, tuyển dụng, tư pháp) vào nhóm bắt buộc giải thích. Tuy nhiên chưa có tiêu chuẩn kỹ thuật nào quy định cụ thể "explanation đủ tốt" nghĩa là gì về mặt định lượng — tạo ra khoảng trống pháp lý lớn mà cộng đồng nghiên cứu đang nỗ lực lấp đầy.
