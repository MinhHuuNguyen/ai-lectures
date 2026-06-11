---
time: 06/24/2022
title: Bài toán OCR (Optical Character Recognition)
description: OCR (Optical Character Recognition) là bài toán chuyển đổi chữ viết trong hình ảnh thành văn bản máy đọc được. Đây là một trong những bài toán kinh điển và có giá trị ứng dụng cao nhất của Computer Vision, là cầu nối giữa thế giới hình ảnh và thế giới văn bản số. Một hệ thống OCR hoàn chỉnh thường gồm hai giai đoạn cốt lõi là phát hiện vùng chữ (text detection) và nhận dạng nội dung chữ (text recognition), và ngày nay đang dần hợp nhất vào các mô hình thị giác - ngôn ngữ đa phương thức.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: false
---

## 1. Giới thiệu chung về OCR

Bài toán **OCR (Optical Character Recognition - Nhận dạng ký tự quang học)** là nhiệm vụ xây dựng các mô hình có khả năng "đọc" được chữ viết xuất hiện trong một hình ảnh và chuyển nó thành chuỗi văn bản mà máy tính có thể lưu trữ, tìm kiếm và xử lý.
Đầu vào có thể là ảnh chụp một trang sách, một tấm hóa đơn, một biển báo giao thông, một tờ chứng minh nhân dân, hay một khung hình video bất kỳ — đầu ra là nội dung văn bản tương ứng cùng (tuỳ bài toán) vị trí của từng vùng chữ.

Khác với việc con người đọc chữ một cách tự nhiên, máy tính phải vượt qua rất nhiều biến thể: font chữ đa dạng, kích thước khác nhau, chữ bị nghiêng, cong, mờ, thiếu sáng, nền phức tạp, hay chữ viết tay nguệch ngoạc.
Vì vậy OCR vừa là một bài toán đã có lịch sử hơn nửa thế kỷ, vừa là một lĩnh vực vẫn đang được nghiên cứu sôi nổi cho đến ngày nay.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/applications.jpeg" style="width: 1000px;"/>

OCR có ứng dụng thực tiễn vô cùng rộng khắp, là một trong những công nghệ AI được thương mại hóa sớm và thành công nhất:
- **Số hóa tài liệu và lưu trữ:** Chuyển sách giấy, văn bản hành chính, hồ sơ lịch sử thành dữ liệu số có thể tìm kiếm toàn văn, tiết kiệm không gian lưu trữ và bảo tồn tư liệu.
- **Tự động hóa nghiệp vụ (Document AI):** Đọc hóa đơn, chứng từ, hợp đồng, tờ khai để trích xuất các trường thông tin quan trọng (key-value) phục vụ kế toán, ngân hàng, bảo hiểm.
- **Định danh điện tử (eKYC):** Đọc thông tin trên căn cước công dân, hộ chiếu, bằng lái xe để xác thực danh tính khách hàng trực tuyến.
- **Giao thông thông minh:** Nhận dạng biển số xe (ALPR) phục vụ bãi đỗ xe, trạm thu phí, giám sát giao thông.
- **Dịch thuật qua camera:** Các ứng dụng như Google Lens dịch tức thời biển hiệu, thực đơn, tài liệu nước ngoài.
- **Hỗ trợ người khiếm thị:** Đọc to nội dung văn bản trong môi trường xung quanh thông qua camera điện thoại.

### Phân loại các bài toán con trong OCR

OCR không phải là một bài toán đơn lẻ mà là một họ các bài toán liên quan, có thể phân loại theo nhiều chiều khác nhau.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/ocr_subtasks.jpeg" style="width: 1000px;"/>

- **Theo loại chữ — chữ in vs chữ viết tay:** Nhận dạng chữ in (printed text) tương đối dễ vì font chuẩn hóa; nhận dạng chữ viết tay (**Handwritten Text Recognition - HTR**) khó hơn nhiều do mỗi người viết một kiểu.
- **Theo bối cảnh — tài liệu vs cảnh tự nhiên:** **Document OCR** xử lý ảnh quét/chụp tài liệu có nền sạch, bố cục rõ ràng; **Scene Text Recognition (STR)** đọc chữ "trong tự nhiên" như biển hiệu, bảng quảng cáo, nhãn sản phẩm — nền phức tạp, chữ nghiêng cong, ánh sáng thất thường.
- **Theo giai đoạn xử lý — phát hiện vs nhận dạng vs end-to-end:** Đây là cách phân chia quan trọng nhất về mặt kỹ thuật:
    - **Text Detection:** xác định *ở đâu* có chữ (trả về bounding box / đa giác bao quanh vùng chữ).
    - **Text Recognition:** đọc *nội dung* của một vùng chữ đã được cắt ra thành chuỗi ký tự.
    - **End-to-end Text Spotting:** gộp cả hai vào một mô hình duy nhất.
- **Theo mục tiêu — phiên âm thuần vs hiểu tài liệu:** Ngoài việc đọc đúng chữ, nhiều ứng dụng còn cần hiểu **cấu trúc và ngữ nghĩa** của tài liệu — đâu là tiêu đề, đâu là bảng, trường "Tổng tiền" ứng với giá trị nào. Đây là bài toán **Document Understanding / Key Information Extraction (KIE)**.

### Pipeline OCR kinh điển

Phần lớn các hệ thống OCR thực tế đi theo một pipeline hai bước rõ ràng: trước tiên **phát hiện** tất cả các vùng chứa chữ trong ảnh, sau đó cắt từng vùng ra và đưa qua bộ **nhận dạng** để đọc nội dung, cuối cùng là bước hậu xử lý (sắp xếp thứ tự đọc, sửa lỗi bằng từ điển, ghép thành đoạn).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/ocr_pipeline.jpeg" style="width: 1000px;"/>

Trong các phần tiếp theo, ta sẽ lần lượt đi qua các nhóm phương pháp theo đúng trình tự của pipeline này, từ những kỹ thuật truyền thống đến các mô hình thị giác - ngôn ngữ hiện đại nhất.

## 2. Nhóm các phương pháp giải bài toán OCR

Ta sẽ tổ chức các phương pháp theo **các giai đoạn của pipeline OCR**: bắt đầu từ phương pháp truyền thống, rồi đến hai trụ cột Text Detection và Text Recognition, tiếp theo là hướng gộp chung End-to-end Text Spotting, và cuối cùng là xu hướng hiện đại nhất — Document Understanding và OCR dựa trên mô hình thị giác - ngôn ngữ (VLM).

### 2.1. Phương pháp truyền thống (trước kỷ nguyên deep learning)

Trước khi deep learning bùng nổ, OCR đã là một công nghệ trưởng thành, hoạt động tốt trên tài liệu in ấn sạch sẽ. Các hệ thống kinh điển như **Tesseract** (khởi nguồn từ HP những năm 1980, sau được Google phát triển mã nguồn mở) là đại diện tiêu biểu cho nhóm này.

#### Mô tả ý tưởng và cơ chế hoạt động

Hãy hình dung cách một người thợ sắp chữ thủ công đọc một trang sách in: họ tách trang thành từng dòng, từng dòng thành từng chữ, từng chữ thành từng ký tự, rồi so từng ký tự với "bộ mẫu" trong trí nhớ.
OCR truyền thống mô phỏng đúng quy trình tuần tự, dựa trên luật này, gồm các bước:

1. **Tiền xử lý (Preprocessing):** chuyển ảnh xám, **nhị phân hóa (binarization)** để tách chữ đen khỏi nền trắng (ví dụ thuật toán Otsu), **khử nghiêng (deskew)** để xoay trang về phương ngang, khử nhiễu và chuẩn hóa độ tương phản.
2. **Phân tích bố cục và phân tách (Layout analysis & Segmentation):** xác định khối văn bản, dòng, từ, rồi tách thành từng ký tự riêng lẻ dựa trên khoảng trắng và biên (connected components).
3. **Trích đặc trưng và phân loại (Feature extraction & Classification):** với mỗi ký tự đã tách, trích các đặc trưng hình học thủ công (số nét, vòng kín, tỷ lệ...) rồi đưa qua bộ phân loại (kNN, SVM, hoặc đối sánh mẫu - **template matching**) để gán nhãn ký tự.
4. **Hậu xử lý:** dùng từ điển và mô hình ngôn ngữ n-gram để sửa các lỗi như nhầm "rn" thành "m".

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/traditional_pipeline.jpeg" style="width: 900px;"/>

Một nhánh quan trọng khác là **phát hiện chữ trong ảnh tự nhiên bằng đặc trưng thủ công**: **MSER (Maximally Stable Extremal Regions)** tìm các vùng có độ sáng ổn định (thường là nét chữ), còn **SWT (Stroke Width Transform)** khai thác đặc điểm chữ viết có độ dày nét gần như không đổi để phân biệt chữ với các vật thể khác.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/swt_stroke_width.jpeg" style="width: 700px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **Nhẹ và nhanh:** Không cần GPU, chạy được trên thiết bị cấu hình thấp, rất hiệu quả với tài liệu in chuẩn.
- **Dễ giải thích và kiểm soát:** Mỗi bước đều minh bạch, dễ debug và tinh chỉnh theo luật.
- **Trưởng thành và sẵn có:** Các thư viện như Tesseract đã được tối ưu nhiều năm, hỗ trợ hàng trăm ngôn ngữ.

**Nhược điểm:**
- **Kém bền với điều kiện khó:** Sai số tăng vọt khi gặp chữ nghiêng, cong, nền phức tạp, ánh sáng kém — tức là gần như mọi tình huống "scene text".
- **Phụ thuộc nặng vào phân tách:** Nếu bước segmentation tách sai ký tự (chữ dính nhau), toàn bộ kết quả phía sau sẽ sai theo.
- **Đặc trưng thủ công không tổng quát:** Phải thiết kế lại đặc trưng cho mỗi loại font/ngôn ngữ, khó mở rộng.

#### Một số mô hình tiêu biểu trong nhóm

- **Tesseract (Smith, 2007)** — [paper](https://ieeexplore.ieee.org/document/4376991) — engine OCR mã nguồn mở kinh điển nhất, mạnh với tài liệu in (các phiên bản mới đã tích hợp LSTM).
- **Template / Pattern Matching** — đối sánh trực tiếp ảnh ký tự với bộ mẫu, nền tảng của các máy OCR đời đầu.
- **MSER (Matas et al., 2002)** — [paper](https://www.bmva.org/bmvc/2002/papers/119/) — phát hiện vùng cực trị ổn định, được dùng rộng rãi để dò chữ trong ảnh tự nhiên.
- **SWT - Stroke Width Transform (Epshtein et al., 2010)** — [paper](https://ieeexplore.ieee.org/document/5540041) — khai thác độ dày nét gần như không đổi của chữ để phát hiện vùng text.

### 2.2. Text Detection (Phát hiện vùng chữ)

Khi chuyển sang ảnh tự nhiên, bài toán đầu tiên cần giải là: *chữ nằm ở đâu trong ảnh?* Text Detection trả về các hộp bao (bounding box), tứ giác (quadrilateral) hoặc đa giác (polygon) khoanh vùng từng dòng/từ chứa chữ. Đây thực chất là một biến thể đặc thù của bài toán **object detection** với "đối tượng" là chữ.

#### Mô tả ý tưởng và cơ chế hoạt động

Chữ có những đặc thù khiến nó khác với object detection thông thường: tỷ lệ khung hình rất đa dạng (một dòng chữ có thể dài và mảnh), hướng bất kỳ (nghiêng, dọc, cong), và mật độ dày đặc. Có **hai họ phương pháp chính**:

- **Regression-based (dựa trên hồi quy):** coi mỗi vùng chữ như một đối tượng, mô hình trực tiếp hồi quy ra tọa độ hộp/tứ giác bao quanh, tương tự các bộ phát hiện một giai đoạn (single-stage). Tiêu biểu là **EAST**, sinh thẳng từ ảnh ra bản đồ điểm số (score map) và hình học của tứ giác chữ, rất nhanh và gọn.
- **Segmentation-based (dựa trên phân vùng):** phân loại từng pixel thuộc "chữ" hay "nền" để tạo mặt nạ (segmentation map), rồi nhóm các pixel thành vùng chữ. Cách này linh hoạt với chữ **cong và hình dạng bất kỳ**, nhưng bước hậu xử lý nhóm pixel thường chậm.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/text_detection_reg_vs_seg.jpeg" style="width: 900px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/east_pipeline.jpeg" style="width: 800px;"/>

Một bước đột phá của nhóm segmentation là **DBNet (Differentiable Binarization)**. Vấn đề của các phương pháp phân vùng là cần một ngưỡng để nhị phân hóa bản đồ xác suất thành mặt nạ, nhưng phép nhị phân hóa cứng (hard threshold) không khả vi nên không học được. DBNet thay nó bằng **nhị phân hóa khả vi**:

$$\hat{B}_{i,j} = \frac{1}{1 + e^{-k(P_{i,j} - T_{i,j})}}$$

trong đó $P$ là bản đồ xác suất, $T$ là **bản đồ ngưỡng được học cùng mô hình**, và $k$ là hệ số khuếch đại (thường $k=50$). Nhờ hàm xấp xỉ sigmoid này, ngưỡng nhị phân hóa trở nên khả vi và được tối ưu end-to-end, giúp tách các dòng chữ sát nhau rất tốt mà vẫn nhanh.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/dbnet_arch.jpeg" style="width: 800px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **Regression nhanh, gọn:** Phù hợp ứng dụng thời gian thực với chữ thẳng (biển số, văn bản quét).
- **Segmentation linh hoạt:** Xử lý tốt chữ cong, đa hướng, hình dạng tùy ý nhờ làm việc ở mức pixel.
- **Tận dụng được tiến bộ của object detection:** Kế thừa backbone, FPN, anchor... từ các bộ phát hiện vật thể.

**Nhược điểm:**
- **Regression khó với chữ cong:** Hộp/tứ giác không bao khít được chữ uốn lượn.
- **Segmentation cần hậu xử lý:** Bước nhóm pixel thành vùng chữ có thể chậm và nhạy với ngưỡng (DBNet ra đời để giảm bớt nhược điểm này).
- **Chữ dày đặc và nhỏ:** Các dòng chữ sát nhau, kích thước nhỏ vẫn là thách thức gây dính vùng hoặc bỏ sót.

#### Một số mô hình tiêu biểu trong nhóm

- **CTPN (Tian et al., 2016)** — [paper](https://arxiv.org/abs/1609.03605) — phát hiện chữ ngang bằng chuỗi các "proposal" dọc hẹp kết nối bằng RNN, rất tốt cho văn bản tài liệu.
- **EAST (Zhou et al., 2017)** — [paper](https://arxiv.org/abs/1704.03155) — pipeline cực gọn, sinh thẳng score map và tứ giác xoay, là baseline regression kinh điển.
- **TextBoxes / TextBoxes++ (Liao et al., 2016–2018)** — [TextBoxes](https://arxiv.org/abs/1611.06779), [TextBoxes++](https://arxiv.org/abs/1801.02765) — cải tiến SSD với anchor "dài và dẹt" hợp với hình dạng dòng chữ.
- **SegLink (Shi et al., 2017)** — [paper](https://arxiv.org/abs/1703.06520) — phát hiện các đoạn chữ nhỏ (segment) rồi học cách liên kết (link) chúng thành dòng dài.
- **PSENet (Wang et al., 2019)** — [paper](https://arxiv.org/abs/1903.12473) — Progressive Scale Expansion, tách các dòng chữ sát nhau bằng cách "phình to" dần các kernel.
- **PAN (Wang et al., 2019)** — [paper](https://arxiv.org/abs/1908.05900) — Pixel Aggregation Network, nhanh và nhẹ hơn PSENet.
- **DBNet / DB++ (Liao et al., 2020–2022)** — [DBNet](https://arxiv.org/abs/1911.08947), [DB++](https://arxiv.org/abs/2202.10304) — nhị phân hóa khả vi, cân bằng tốc độ và độ chính xác hàng đầu, được dùng rộng rãi trong PaddleOCR.
- **CRAFT (Baek et al., 2019)** — [paper](https://arxiv.org/abs/1904.01941) — dự đoán xác suất tâm ký tự và mối liên kết giữa các ký tự (affinity), bám rất sát chữ cong.

### 2.3. Text Recognition (Nhận dạng nội dung chữ)

Sau khi đã cắt được một vùng chữ (thường là một dòng hoặc một từ), nhiệm vụ của Text Recognition là biến ảnh đó thành chuỗi ký tự. Đây là bài toán **chuỗi sang chuỗi** (sequence-to-sequence) đặc thù: đầu vào là chuỗi các cột đặc trưng của ảnh, đầu ra là chuỗi ký tự, và hai chuỗi này **không thẳng hàng (unaligned)** — ta không biết trước mỗi ký tự chiếm bao nhiêu cột pixel.

#### Mô tả ý tưởng và cơ chế hoạt động

Có **hai trường phái lớn** để giải bài toán căn chỉnh này: dựa trên **CTC** và dựa trên **Attention**.

**Trường phái CTC — kiến trúc CRNN.** Mô hình kinh điển **CRNN (Convolutional Recurrent Neural Network)** gồm ba khối: CNN trích đặc trưng thành một dãy "cột" theo chiều ngang → BiLSTM mô hình hóa ngữ cảnh chuỗi → tầng **CTC (Connectionist Temporal Classification)** giải mã ra chuỗi ký tự.

CTC giải quyết việc không thẳng hàng bằng cách cho phép mỗi bước thời gian dự đoán một ký tự hoặc một ký tự rỗng đặc biệt (blank), rồi gộp các ký tự lặp và bỏ blank. Xác suất của một nhãn $l$ là tổng xác suất của **mọi đường đi (path)** $\pi$ có thể rút gọn về $l$:

$$p(l \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(l)} \prod_{t=1}^{T} y^{t}_{\pi_t}$$

trong đó $y^{t}_{\pi_t}$ là xác suất xuất ký tự $\pi_t$ tại bước $t$, và $\mathcal{B}$ là phép gộp (gộp ký tự lặp, bỏ blank). Hàm mất mát là $\mathcal{L}_{CTC} = -\log p(l \mid x)$.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/crnn_ctc.jpeg" style="width: 900px;"/>

**Trường phái Attention — seq2seq.** Thay vì CTC, ta dùng một decoder tự hồi quy với cơ chế **attention**: ở mỗi bước, decoder "nhìn" có chọn lọc vào các vùng đặc trưng liên quan của ảnh để sinh ký tự tiếp theo, tương tự dịch máy. Cách này nắm bắt phụ thuộc ngôn ngữ tốt hơn, đặc biệt với từ dài.

Với **chữ bất quy tắc** (cong, nghiêng, phối cảnh), nhiều mô hình thêm bước **nắn chỉnh (rectification)** trước khi nhận dạng: **ASTER** dùng mạng biến đổi không gian **STN/TPS (Thin-Plate Spline)** để "duỗi thẳng" chữ cong về dạng ngang ngắn gọn, giúp bộ nhận dạng làm việc dễ hơn. Gần đây, các kiến trúc thuần **Transformer** (SVTR, PARSeq) hoặc kết hợp mô hình ngôn ngữ (ABINet) đạt độ chính xác cao nhất.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/aster_rectification.jpeg" style="width: 800px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/attention_vs_ctc.jpeg" style="width: 900px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **CTC nhanh, song song:** Không cần căn chỉnh thủ công, giải mã song song, rất phù hợp triển khai thực tế (CRNN vẫn là lựa chọn mặc định cho nhiều hệ thống).
- **Attention chính xác hơn:** Nắm bắt ngữ cảnh và phụ thuộc dài, đọc tốt từ khó, chữ dính.
- **Rectification giúp xử lý chữ cong:** Module nắn chỉnh giúp đọc chữ bất quy tắc mà không cần đổi bộ nhận dạng.

**Nhược điểm:**
- **CTC giả định đơn điệu (monotonic):** Giả định ký tự đi từ trái sang phải, khó với bố cục phức tạp; cũng yếu về mô hình ngôn ngữ.
- **Attention dễ "trôi" (attention drift):** Trên ảnh nhiễu/ dài, attention có thể căn sai vị trí gây mất hoặc lặp ký tự; lại giải mã tuần tự nên chậm hơn.
- **Phụ thuộc chất lượng cắt vùng:** Nếu Text Detection cắt thiếu/thừa, bộ nhận dạng sẽ đọc sai — nhược điểm cố hữu của pipeline tách rời.

#### Một số mô hình tiêu biểu trong nhóm

- **CTC (Graves et al., 2006)** — [paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — bài báo nền tảng giới thiệu Connectionist Temporal Classification.
- **CRNN (Shi et al., 2015)** — [paper](https://arxiv.org/abs/1507.05717) — CNN + BiLSTM + CTC, kiến trúc nhận dạng chữ kinh điển và phổ biến nhất.
- **ASTER (Shi et al., 2018)** — [paper](https://ieeexplore.ieee.org/document/8395027) — thêm rectification TPS + attention, mở đường cho nhận dạng chữ bất quy tắc.
- **SAR (Li et al., 2019)** — [paper](https://arxiv.org/abs/1811.00751) — Show, Attend and Read, attention 2D mạnh với chữ cong.
- **SRN (Yu et al., 2020)** — [paper](https://arxiv.org/abs/2003.12294) — Semantic Reasoning Network, lồng mô-đun suy luận ngữ nghĩa song song.
- **ABINet (Fang et al., 2021)** — [paper](https://arxiv.org/abs/2103.06495) — tách biệt mô hình thị giác và mô hình ngôn ngữ, lặp sửa lỗi (iterative correction).
- **MASTER (Lu et al., 2021)** — [paper](https://arxiv.org/abs/1910.02562) — dùng self-attention toàn cục thay RNN, giảm attention drift.
- **SVTR (Du et al., 2022)** — [paper](https://arxiv.org/abs/2205.00159) — kiến trúc thuần thị giác (vision transformer) cho nhận dạng, nhanh và mạnh.
- **PARSeq (Bautista & Atienza, 2022)** — [paper](https://arxiv.org/abs/2207.06966) — Permuted Autoregressive Sequence, SOTA trên nhiều benchmark scene text.

### 2.4. End-to-end Text Spotting (Phát hiện và nhận dạng hợp nhất)

Pipeline tách rời Detection → Recognition có một điểm yếu cố hữu: lỗi tích lũy. Nếu bộ phát hiện cắt vùng chữ lệch một chút, bộ nhận dạng dù tốt đến đâu cũng đọc sai. **End-to-end Text Spotting** giải quyết điều này bằng cách huấn luyện cả hai nhiệm vụ trong **một mô hình duy nhất**, chia sẻ đặc trưng và lan truyền gradient chung.

#### Mô tả ý tưởng và cơ chế hoạt động

Ý tưởng cốt lõi: một backbone trích đặc trưng dùng chung cho cả phát hiện và nhận dạng. Sau khi nhánh phát hiện đề xuất vùng chữ, một thao tác trích đặc trưng vùng (**RoI**) sẽ "cắt" đặc trưng tương ứng đưa thẳng sang nhánh nhận dạng — tất cả trong cùng một lượt forward, tối ưu cùng một lúc.

Thách thức lớn nhất là **cắt đặc trưng cho chữ nghiêng/cong**. Các mô hình giải quyết theo nhiều cách:
- **FOTS** đề xuất **RoIRotate** — xoay đặc trưng vùng chữ về phương ngang trước khi nhận dạng, xử lý chữ nghiêng mượt mà.
- **Mask TextSpotter** dùng mặt nạ phân vùng (mask) ở mức ký tự, đọc được cả chữ có hình dạng tùy ý.
- **ABCNet** biểu diễn đường biên chữ cong bằng **đường cong Bezier** và đề xuất **BezierAlign** để cắt đặc trưng dọc theo đường cong — rất hiệu quả với chữ uốn lượn và đạt tốc độ thời gian thực.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/end2end_spotting.jpeg" style="width: 900px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/abcnet_bezier.jpeg" style="width: 1000px;"/>

*Pipeline ABCNet: ảnh đầu vào → backbone + phát hiện đường cong Bezier → BezierAlign → đầu nhận dạng nhẹ. Nguồn: Liu et al., 2020, [ABCNet](https://arxiv.org/abs/2002.10200) — giấy phép CC BY-NC-SA 4.0.*

#### Ưu và nhược điểm

**Ưu điểm:**
- **Tối ưu toàn cục:** Hai nhiệm vụ "kéo" nhau cùng tiến bộ, giảm lỗi tích lũy của pipeline rời.
- **Nhanh hơn khi triển khai:** Một lượt forward duy nhất, chia sẻ tính toán backbone.
- **Đặc trưng chia sẻ giàu ngữ cảnh:** Nhánh nhận dạng được hưởng đặc trưng toàn ảnh, hữu ích khi vùng chữ mờ.

**Nhược điểm:**
- **Huấn luyện phức tạp:** Cân bằng nhiều hàm mất mát (phát hiện + nhận dạng) khó, cần kỹ thuật để hội tụ ổn định.
- **Đòi hỏi nhãn chi tiết:** Cần nhãn vừa là đa giác vị trí vừa là nội dung chữ, tốn công gán nhãn.
- **Khó "vá" từng phần:** Không thể dễ dàng thay riêng bộ nhận dạng như pipeline mô-đun.

#### Một số mô hình tiêu biểu trong nhóm

- **FOTS (Liu et al., 2018)** — [paper](https://arxiv.org/abs/1801.01671) — Fast Oriented Text Spotting với RoIRotate, end-to-end thời gian thực cho chữ nghiêng.
- **Mask TextSpotter v1–v3 (Lyu et al., 2018–2020)** — [v1](https://arxiv.org/abs/1807.02242), [v3](https://arxiv.org/abs/2007.09482) — dựa trên Mask R-CNN, đọc chữ hình dạng tùy ý.
- **ABCNet / ABCNetv2 (Liu et al., 2020–2021)** — [ABCNet](https://arxiv.org/abs/2002.10200), [ABCNetv2](https://arxiv.org/abs/2105.03620) — biểu diễn Bezier + BezierAlign, nhanh và mạnh với chữ cong.
- **TESTR (Zhang et al., 2022)** — [paper](https://arxiv.org/abs/2204.01918) — Text Spotting Transformer, kiến trúc dựa trên DETR, bỏ nhiều bước hậu xử lý.
- **SwinTextSpotter (Huang et al., 2022)** — [paper](https://arxiv.org/abs/2203.10209) — dùng backbone Swin Transformer, tăng tương tác giữa hai nhánh.
- **DeepSolo (Ye et al., 2023)** — [paper](https://arxiv.org/abs/2211.10772) — một decoder Transformer duy nhất giải đồng thời cả phát hiện và nhận dạng.

### 2.5. Document Understanding & OCR dựa trên mô hình thị giác - ngôn ngữ (VLM)

Nhóm phương pháp hiện đại nhất đẩy OCR vượt khỏi việc "đọc chữ thuần". Trong rất nhiều ứng dụng thực tế (đọc hóa đơn, biểu mẫu, hợp đồng), điều ta cần không chỉ là chuỗi ký tự mà là **hiểu cấu trúc và ngữ nghĩa tài liệu**: đâu là tiêu đề, đâu là bảng, trường "Tổng tiền" ứng với giá trị nào (**Key Information Extraction - KIE**). Đồng thời, xu hướng dùng **mô hình thị giác - ngôn ngữ (Vision-Language Model)** cho phép đọc thẳng từ ảnh sang văn bản mà không cần engine OCR riêng (**OCR-free**).

#### Mô tả ý tưởng và cơ chế hoạt động

Có thể chia nhóm này thành ba hướng tiếp cận chính:

1. **Kết hợp đa phương thức có sẵn bố cục (Layout-aware):** **LayoutLM** và các phiên bản sau nhúng đồng thời ba loại thông tin — **nội dung chữ (text), vị trí 2D (layout/bounding box), và đặc trưng hình ảnh** — vào một Transformer, để mô hình hiểu rằng hai trường gần nhau về không gian thường có quan hệ. Rất mạnh cho KIE trên biểu mẫu, hóa đơn. (Hướng này vẫn cần một engine OCR phía trước để lấy text + box.)

2. **OCR-free encoder-decoder:** **Donut** (Document Understanding Transformer) bỏ hẳn engine OCR: một encoder thị giác (Swin) đọc ảnh tài liệu, một decoder Transformer sinh thẳng ra chuỗi đầu ra (văn bản hoặc JSON có cấu trúc). **TrOCR** dùng encoder ViT + decoder ngôn ngữ pretrain. **Nougat** chuyển ảnh tài liệu khoa học thành markdown (kể cả công thức toán). Cách này tránh được lỗi tích lũy của engine OCR rời.

3. **VLM tổng quát theo prompt:** Các mô hình thị giác - ngôn ngữ lớn như **GPT-4V, Gemini, Qwen2-VL, InternVL** có thể đọc chữ trong ảnh và trả lời theo yêu cầu ngôn ngữ tự nhiên ("trích tất cả các trường trong hóa đơn này thành JSON"). **GOT-OCR2.0** là mô hình chuyên biệt thế hệ mới hợp nhất nhiều tác vụ OCR (văn bản, công thức, bảng, nhạc) trong một kiến trúc thống nhất.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/vlm_ocr.jpeg" style="width: 1000px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/donut_vs_pipeline.jpeg" style="width: 800px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/layoutlm_arch.jpeg" style="width: 800px;"/>

*Kiến trúc LayoutLMv3 nhúng đồng thời text + layout 2D + ảnh vào một Multimodal Transformer. Nguồn: Huang et al., 2022, [LayoutLMv3](https://arxiv.org/abs/2204.08387) — giấy phép CC BY-NC-SA 4.0.*

Chi tiết hơn về kiến trúc Transformer và cơ chế attention nền tảng cho nhóm mô hình này đã được trình bày trong [bài giảng về mô hình Transformer](/blog/mo-hinh-transformer).

#### Ưu và nhược điểm

**Ưu điểm:**
- **Hiểu ngữ cảnh và cấu trúc:** Không chỉ đọc chữ mà còn nắm quan hệ bố cục, trích xuất key-value mạnh mẽ — đúng nhu cầu Document AI.
- **OCR-free giảm lỗi tích lũy:** Mô hình một bước (Donut, TrOCR) tránh được lỗi do engine OCR phía trước.
- **Linh hoạt theo prompt:** VLM tổng quát giải nhiều tác vụ (đọc, hỏi đáp, trích xuất) chỉ bằng đổi câu lệnh, không cần huấn luyện lại.

**Nhược điểm:**
- **Tốn tài nguyên:** Các mô hình lớn cần GPU mạnh, độ trễ và chi phí inference cao, khó chạy trên thiết bị biên.
- **Nguy cơ "bịa" (hallucination):** VLM có thể tự sinh ra văn bản nghe hợp lý nhưng không có trong ảnh — rủi ro nghiêm trọng với tài liệu pháp lý/tài chính.
- **Khó kiểm soát và giải thích:** Đầu ra dạng "hộp đen", khó debug và đảm bảo độ chính xác tuyệt đối ở mức ký tự.

#### Một số mô hình tiêu biểu trong nhóm

- **LayoutLM / v2 / v3 (Xu et al., 2019–2022)** — [LayoutLM](https://arxiv.org/abs/1912.13318), [LayoutLMv3](https://arxiv.org/abs/2204.08387) — tiền huấn luyện đa phương thức text + layout + image cho hiểu tài liệu.
- **TrOCR (Li et al., 2021)** — [paper](https://arxiv.org/abs/2109.10282) — encoder ViT + decoder ngôn ngữ pretrain, nhận dạng chữ in và viết tay mạnh.
- **Donut (Kim et al., 2021)** — [paper](https://arxiv.org/abs/2111.15664) — OCR-free, ảnh tài liệu → chuỗi/JSON có cấu trúc trực tiếp.
- **Pix2Struct (Lee et al., 2022)** — [paper](https://arxiv.org/abs/2210.03347) — tiền huấn luyện ảnh-sang-văn-bản cho hiểu ảnh giàu chữ (screenshot, biểu đồ).
- **Nougat (Blecher et al., 2023)** — [paper](https://arxiv.org/abs/2308.13418) — chuyển PDF tài liệu khoa học thành markdown, đọc được cả công thức toán.
- **GOT-OCR2.0 (Wei et al., 2024)** — [paper](https://arxiv.org/abs/2409.01704) — mô hình OCR thế hệ mới hợp nhất (văn bản, công thức, bảng, sheet nhạc) trong một kiến trúc.
- **Qwen2-VL (Wang et al., 2024)** — [paper](https://arxiv.org/abs/2409.12191) — VLM mã nguồn mở có khả năng OCR đa ngôn ngữ rất mạnh.
- **GPT-4V (OpenAI, 2023)** & **Gemini (Google, 2023)** — VLM thương mại quy mô lớn, đọc và hiểu tài liệu theo prompt ngôn ngữ tự nhiên.

## 3. Các metrics trong OCR

Một hệ thống OCR cần được đánh giá ở nhiều mức khác nhau tùy bài toán: mức **ký tự** và **từ** cho nhận dạng, mức **vùng** cho phát hiện, và mức **trường thông tin** cho hiểu tài liệu. Khác với image generation (đánh giá chất lượng thị giác), OCR có **đáp án đúng (ground truth) rõ ràng**, nên phần lớn metric dựa trên việc so khớp chuỗi và đo khoảng cách chỉnh sửa.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/metrics.jpeg" style="width: 1000px;"/>

### 3.1. Character Error Rate (CER)

**Character Error Rate (CER)** là thước đo phổ biến nhất cho chất lượng nhận dạng, đo tỷ lệ ký tự bị lỗi giữa chuỗi dự đoán và chuỗi đúng. Nó dựa trên **khoảng cách chỉnh sửa Levenshtein** — số phép chèn, xóa, thay thế tối thiểu để biến chuỗi này thành chuỗi kia.

#### Mô tả ý tưởng và cơ chế hoạt động

CER được tính bằng tổng số lỗi chia cho số ký tự của chuỗi đúng:

$$CER = \frac{S + D + I}{N}$$

trong đó $S$ là số ký tự bị **thay thế (Substitution)**, $D$ là số ký tự bị **xóa (Deletion)**, $I$ là số ký tự bị **thêm thừa (Insertion)**, và $N$ là tổng số ký tự trong chuỗi đúng. Giá trị CER **càng thấp càng tốt**; CER = 0 nghĩa là đọc đúng hoàn toàn. Lưu ý CER có thể lớn hơn 1 nếu mô hình thêm quá nhiều ký tự thừa.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/edit_distance.jpeg" style="width: 700px;"/>

#### Ví dụ

Giả sử chuỗi đúng là `"HANOI"` ($N = 5$) và mô hình đọc ra `"HAN0I"` (nhầm chữ O thành số 0):
- Có 1 phép thay thế (O → 0): $S = 1$, $D = 0$, $I = 0$.
- $CER = \dfrac{1 + 0 + 0}{5} = 0.2$ (tức 20% ký tự sai).

Một ví dụ khác, chuỗi đúng `"OCR"` ($N=3$) đọc thành `"0 C R"` (thêm dấu cách thừa): $I = 1$, $S = 1$ (O→0) → $CER = \frac{1+0+1}{3} \approx 0.67$.

Giá trị CER tham chiếu điển hình:

| Bối cảnh | CER điển hình |
|---|---|
| Tài liệu in sạch, chất lượng cao | < 1% |
| Scene text (biển hiệu, ảnh tự nhiên) | 3% – 10% |
| Chữ viết tay | 5% – 20% |
| Tài liệu lịch sử, ảnh kém chất lượng | > 20% |

#### Ưu và nhược điểm

**Ưu điểm:**
- **Trực quan và chuẩn hóa:** Dễ hiểu, dễ so sánh giữa các mô hình và bộ dữ liệu.
- **Mức ký tự chi tiết:** Phản ánh chính xác từng lỗi nhỏ, phù hợp ngôn ngữ không tách từ rõ ràng.
- **Áp dụng rộng:** Dùng được cho mọi ngôn ngữ, mọi loại chữ.

**Nhược điểm:**
- **Không phân biệt mức nghiêm trọng:** Một lỗi vô hại (dấu phẩy) và một lỗi đổi nghĩa được tính như nhau.
- **Nhạy với chuẩn hóa:** Kết quả phụ thuộc cách xử lý chữ hoa/thường, dấu cách, dấu câu — cần thống nhất quy ước.
- **Bỏ qua bố cục:** Chỉ đo nội dung chuỗi, không phản ánh thứ tự đọc hay cấu trúc tài liệu.

### 3.2. Word Error Rate (WER) và Word Accuracy

**Word Error Rate (WER)** tương tự CER nhưng hoạt động ở **mức từ** thay vì ký tự, thường dùng khi ta quan tâm đến độ chính xác của cả từ (ví dụ tìm kiếm toàn văn). **Word Accuracy** là tỷ lệ từ được đọc đúng hoàn toàn.

#### Mô tả ý tưởng và cơ chế hoạt động

WER cũng dựa trên khoảng cách chỉnh sửa nhưng đơn vị là **từ**:

$$WER = \frac{S_w + D_w + I_w}{N_w}$$

với $S_w, D_w, I_w$ là số từ bị thay thế, xóa, thêm thừa, và $N_w$ là tổng số từ đúng. Trong các benchmark scene text (IIIT5K, SVT...), người ta thường báo cáo **Word Accuracy** — một từ chỉ được tính đúng khi **mọi ký tự đều đúng**.

#### Ví dụ

Câu đúng: `"NHA XUAT BAN GIAO DUC"` (5 từ, $N_w = 5$). Mô hình đọc: `"NHA XUAT BAN GIAU DUC"` (nhầm "GIAO" → "GIAU"):
- 1 từ bị thay thế: $WER = \dfrac{1}{5} = 0.2$ (20%).

Đáng chú ý, **chỉ một ký tự sai làm hỏng cả từ**: ở ví dụ trên CER chỉ là $\frac{1}{17} \approx 0.06$ (6%) nhưng WER lên tới 20% — cho thấy WER khắt khe hơn CER.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/cer_vs_wer.jpeg" style="width: 800px;"/>

| Metric | Công thức | Đơn vị | Tính chất |
|---|---|---|---|
| CER | $(S+D+I)/N$ | ký tự | Mịn, khoan dung với lỗi nhỏ |
| WER | $(S_w+D_w+I_w)/N_w$ | từ | Khắt khe, một ký tự sai hỏng cả từ |
| Word Accuracy | $\#\text{từ đúng} / \#\text{từ}$ | từ | Đánh giá theo từ trọn vẹn |

#### Ưu và nhược điểm

**Ưu điểm:**
- **Sát với trải nghiệm người dùng:** Nhiều ứng dụng (tìm kiếm, dịch) quan tâm từ đúng trọn vẹn hơn là từng ký tự.
- **Dễ diễn giải nghiệp vụ:** "Đọc đúng 95% số từ" dễ hiểu với người dùng cuối.

**Nhược điểm:**
- **Quá khắt khe với ngôn ngữ có dấu:** Một lỗi dấu nhỏ vẫn tính là sai cả từ — đặc biệt bất lợi cho tiếng Việt.
- **Phụ thuộc định nghĩa "từ":** Với ngôn ngữ không tách từ bằng dấu cách (tiếng Trung, Nhật), WER khó áp dụng — khi đó CER phù hợp hơn.

### 3.3. Metrics cho Text Detection (Precision / Recall / F1)

Phát hiện vùng chữ về bản chất là một bài toán detection, nên được đánh giá bằng đúng bộ công cụ của object detection: xác định **TP/FP/FN** qua ngưỡng **IoU**, rồi tính **Precision** và **Recall**.

> 📌 **Phần này dùng lại nguyên các khái niệm của bài toán object detection.** Định nghĩa **IoU**, cách xác định **TP/FP/FN** theo ngưỡng IoU, và công thức **Precision / Recall** đã được trình bày chi tiết trong [bài giảng Object Detection](/blog/object-detection) (mục *Các metrics trong object detection*) nên ở đây không nhắc lại. Dưới đây ta chỉ tập trung vào những điểm **khác biệt đặc thù của OCR**.

#### Mô tả ý tưởng và cơ chế hoạt động

Khác với object detection (vốn thường báo cáo **mAP**), text detection chủ yếu báo cáo **F1-score** (còn gọi là **H-mean**) — trung bình điều hòa của Precision và Recall tại một ngưỡng IoU cố định (thường $0.5$):

$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Ngoài ra, do một dòng chữ có thể bị tách thành nhiều hộp (hoặc nhiều dòng bị gộp làm một), các benchmark ICDAR dùng các giao thức ghép cặp chuẩn hóa riêng cho chữ: **IoU protocol** (ICDAR 2015) và **DetEval** (xử lý trường hợp một-nhiều, nhiều-một). Với chữ cong, người ta thay hộp chữ nhật bằng **đa giác (polygon)** và tính IoU trên đa giác.

#### Ví dụ

Một ảnh có 10 dòng chữ thật. Mô hình dự đoán 12 hộp, trong đó 9 hộp có IoU ≥ 0.5 với một dòng đúng:
- $TP = 9$, $FP = 12 - 9 = 3$, $FN = 10 - 9 = 1$.
- $\text{Precision} = \frac{9}{12} = 0.75$; $\text{Recall} = \frac{9}{10} = 0.90$.
- $F1 = \dfrac{2 \cdot 0.75 \cdot 0.90}{0.75 + 0.90} \approx 0.82$.

Giá trị F1 tham chiếu trên các benchmark detection phổ biến:

| Bộ dữ liệu | Đặc điểm | F1 của mô hình tốt |
|---|---|---|
| ICDAR 2013 | Chữ ngang, tài liệu/biển hiệu rõ | ~0.90+ |
| ICDAR 2015 | Chữ "ngẫu nhiên", nghiêng, mờ | ~0.85+ |
| Total-Text / CTW1500 | Chữ cong, hình dạng tùy ý | ~0.85 |

#### Ưu và nhược điểm

**Ưu điểm:**
- **Chuẩn mực và thống nhất:** Kế thừa hệ metric trưởng thành của object detection, dễ so sánh.
- **Cân bằng được hai loại lỗi:** F1 dung hòa giữa bỏ sót (recall) và báo nhầm (precision).

**Nhược điểm:**
- **Nhạy với ngưỡng IoU:** Đổi ngưỡng 0.5 → 0.7 có thể thay đổi đáng kể kết quả xếp hạng.
- **IoU không lý tưởng cho chữ cong:** Hộp chữ nhật bao chữ uốn lượn cho IoU thấp dù bám sát — cần đa giác và giao thức riêng.
- **Không phản ánh nội dung:** Phát hiện đúng vị trí không có nghĩa đọc đúng chữ — cần metric end-to-end.

### 3.4. Metrics End-to-end (F-score và 1 − N.E.D.)

Khi đánh giá toàn hệ thống (vừa phát hiện vừa nhận dạng), ta cần metric đo **đồng thời cả vị trí lẫn nội dung**. Một kết quả chỉ được tính đúng khi vùng chữ được định vị đúng **và** nội dung đọc ra khớp với ground truth.

#### Mô tả ý tưởng và cơ chế hoạt động

Có hai cách đánh giá end-to-end phổ biến:

- **End-to-end F-score:** giống F1 của detection, nhưng một dự đoán chỉ là TP khi IoU đạt ngưỡng **và** chuỗi đọc ra trùng khớp (thường có hai chế độ: *strong/weak/generic lexicon* — có hỗ trợ từ điển hay không).
- **1 − N.E.D. (Normalized Edit Distance):** thay vì yêu cầu khớp tuyệt đối, đo độ tương đồng chuỗi bằng khoảng cách chỉnh sửa chuẩn hóa. Với mỗi cặp dự đoán - đúng:

$$\text{N.E.D.} = \frac{\text{EditDistance}(\hat{s}, s)}{\max(|\hat{s}|, |s|)}, \qquad \text{Score} = 1 - \text{N.E.D.}$$

Điểm **càng cao càng tốt** (gần 1 là tốt). 1 − N.E.D. khoan dung hơn F-score: đọc gần đúng vẫn được điểm một phần, phù hợp với các cuộc thi ICDAR RRC.

#### Ví dụ

Mô hình phát hiện đúng vị trí một biển hiệu và đọc `"RESTAURANT"` trong khi đáp án là `"RESTAURENT"`:
- Theo **End-to-end F-score** (khớp tuyệt đối): tính là **sai** (False Positive) vì chuỗi không trùng khớp.
- Theo **1 − N.E.D.**: chỉ 1 ký tự sai trên 10 ký tự → N.E.D. $= \frac{1}{10} = 0.1$ → Score $= 0.9$ (vẫn được điểm cao).

Sự khác biệt này cho thấy 1 − N.E.D. phản ánh "mức độ đọc đúng" mượt mà hơn metric nhị phân đúng/sai.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/ned_example.jpeg" style="width: 800px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **Đánh giá đúng cái người dùng cần:** Đo trực tiếp chất lượng đầu ra cuối cùng của cả hệ thống.
- **1 − N.E.D. mượt và công bằng:** Không "trừng phạt" quá nặng một lỗi nhỏ, phân biệt được mô hình đọc gần đúng và đọc sai hẳn.

**Nhược điểm:**
- **Phức tạp khi tính:** Phải ghép cặp dự đoán - ground truth qua IoU rồi mới so chuỗi, nhiều bước.
- **Phụ thuộc từ điển:** Kết quả khác nhau lớn giữa chế độ có/không có lexicon, dễ gây hiểu nhầm khi so sánh.

### 3.5. Metrics cho KIE / Document Understanding

Với bài toán hiểu tài liệu (đọc hóa đơn, biểu mẫu), điều quan trọng không phải đọc đúng từng từ mà là **trích đúng các trường thông tin (field)**. Khi đó ta đánh giá ở mức **trường / thực thể (entity-level)**.

#### Mô tả ý tưởng và cơ chế hoạt động

Mỗi trường key-value (ví dụ `Tổng tiền = 150.000`, `Ngày = 01/06/2026`) được coi là một thực thể. Ta tính **Precision, Recall, F1 ở mức trường**: một trường được tính đúng khi cả tên trường lẫn giá trị được trích chính xác.

$$\text{F1}_{\text{field}} = \frac{2 \cdot P \cdot R}{P + R}, \quad P = \frac{\#\text{trường trích đúng}}{\#\text{trường mô hình trích}}, \quad R = \frac{\#\text{trường trích đúng}}{\#\text{trường thực tế}}$$

#### Ví dụ

Trên bộ **SROIE** (hóa đơn) với 4 trường cần trích (`company`, `date`, `address`, `total`), một hóa đơn mà mô hình trích đúng `company`, `date`, `total` nhưng sai `address`:
- $P = R = \frac{3}{4} = 0.75 \Rightarrow F1 = 0.75$.

Các bộ dữ liệu KIE tiêu biểu: **SROIE** (hóa đơn), **FUNSD** (biểu mẫu quét), **CORD** (hóa đơn nhà hàng).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/kie_extraction.jpeg" style="width: 800px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **Đúng mục tiêu nghiệp vụ:** Đo trực tiếp giá trị mà hệ thống Document AI mang lại (trích đúng trường).
- **Bỏ qua lỗi vô hại:** Sai chính tả ở phần không phải trường quan tâm không bị tính lỗi.

**Nhược điểm:**
- **Định nghĩa "đúng" mơ hồ:** Cần quy ước khớp chính xác hay khớp một phần (ví dụ `150000` vs `150.000`).
- **Phụ thuộc schema:** Mỗi loại tài liệu có bộ trường khác nhau, khó có benchmark tổng quát.

## 4. Các thách thức của bài toán OCR

Dù đã rất trưởng thành trên tài liệu in sạch, OCR vẫn đối mặt nhiều thách thức lớn trong điều kiện thực tế, đặc biệt với scene text, chữ viết tay và các ngôn ngữ giàu dấu như tiếng Việt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/challenges.jpeg" style="width: 1000px;"/>

- **Chữ bất quy tắc:** Chữ cong, nghiêng, dọc, đa hướng, font nghệ thuật cách điệu rất khó cả phát hiện lẫn nhận dạng. Đây là động lực ra đời của các phương pháp segmentation-based, rectification (TPS) và biểu diễn Bezier.

- **Đa ngôn ngữ và đặc thù tiếng Việt:** Đây là một trong những thách thức nổi bật nhất với người dùng Việt Nam. Tiếng Việt có **bộ ký tự lớn** do hệ thống **dấu phụ (diacritics)**: ngoài 12 nguyên âm có dấu mũ/móc (ă, â, ê, ô, ơ, ư...) còn chồng thêm **5 dấu thanh** (sắc, huyền, hỏi, ngã, nặng). Hậu quả:
    - Các ký tự chỉ khác nhau ở dấu (`a`, `à`, `á`, `ả`, `ã`, `ạ`, `â`, `ấ`...) rất **dễ nhầm lẫn**, đặc biệt khi ảnh mờ hoặc độ phân giải thấp — dấu nhỏ dễ bị mất hoặc đọc sai.
    - Dấu thanh nằm phía trên/dưới ký tự dễ bị cắt mất khi phát hiện dòng, hoặc bị nhầm với nhiễu.
    - Một lỗi dấu duy nhất làm sai nghĩa hoàn toàn (`ma` / `má` / `mà` / `mả` / `mã` / `mạ`) và làm hỏng cả từ theo metric WER.

    Giải pháp thực tế cho tiếng Việt gồm thư viện mã nguồn mở **VietOCR** (kết hợp backbone CNN với decoder Transformer/Attention, huấn luyện riêng cho tiếng Việt), và các bộ dữ liệu như **VinText** (scene text tiếng Việt), dữ liệu chữ viết tay **Cinnamon AI**, **BKAI**. Việc sinh dữ liệu tổng hợp (synthetic) có dấu cũng rất quan trọng để bổ sung mẫu huấn luyện.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/15-ocr/vietnamese_ocr.jpeg" style="width: 900px;"/>

- **Chất lượng ảnh kém:** Mờ, nhòe, độ phân giải thấp, thiếu sáng, bóng đổ, lóa sáng, nén JPEG mạnh — tất cả làm suy giảm nghiêm trọng độ chính xác, nhất là với chữ nhỏ và dấu phụ.

- **Chữ viết tay (HTR):** Nét chữ biến thiên vô hạn giữa người này với người khác, chữ dính liền nhau, viết ẩu — khó hơn nhiều so với chữ in và cần nhiều dữ liệu gán nhãn.

- **Bố cục phức tạp:** Bảng biểu, nhiều cột, văn bản đan xen hình ảnh, watermark, con dấu chồng chữ. Xác định **thứ tự đọc (reading order)** đúng là bài toán riêng không hề đơn giản.

- **Tài liệu chuyên ngành và lịch sử:** Đơn thuốc viết tay, văn bản Hán-Nôm, tài liệu cổ ố vàng, biểu mẫu chuyên ngành đòi hỏi dữ liệu và mô hình chuyên biệt.

- **Đánh giá và chuẩn hóa:** Thiếu một chuẩn đánh giá thống nhất giữa các bài toán con; việc so sánh end-to-end giữa các hệ thống vẫn khó khăn do khác biệt về giao thức, từ điển và cách chuẩn hóa văn bản.
