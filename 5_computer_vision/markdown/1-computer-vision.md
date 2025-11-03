---
time: 11/02/2022
title: Thị giác máy tính Computer Vision
description: Computer vision là tổ hợp các bài toán con trong lĩnh vực trí tuệ nhân tạo, nhằm giúp máy tính có thể hiểu và xử lý hình ảnh, video. Computer vision là một trong những lĩnh vực nghiên cứu có rất nhiều ứng dụng thực tiễn trong đời sống giúp nâng cao hiệu quả công việc của con người và tự động hoá nhiều quy trình.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về lĩnh vực Computer vision

Computer Vision (Thị giác máy tính) là một lĩnh vực nghiên cứu và ứng dụng của trí tuệ nhân tạo (AI) và xử lý ảnh, với mục tiêu cho phép máy tính “nhìn thấy” và hiểu được nội dung của hình ảnh hoặc video tương tự như con người.

Bằng cách sử dụng các kỹ thuật từ phân tích đặc trưng cục bộ (feature extraction) cho đến các mô hình học sâu (deep learning), hệ thống Computer Vision có thể thực hiện các nhiệm vụ phong phú như object recognition, face detection, image segmentation, depth estimation hay motion tracking ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/general.jpeg" style="width: 1000px;"/>

Nhờ tiềm năng phân tích hình ảnh với độ chính xác cao và khả năng xử lý tự động quy mô lớn, Computer Vision đã trở thành công nghệ then chốt trong nhiều ứng dụng thực tiễn: từ ô tô tự lái (autonomous driving) sử dụng camera để phát hiện chướng ngại vật, hệ thống giám sát an ninh (surveillance) tự động cảnh báo khi phát hiện hành vi bất thường, đến hỗ trợ chẩn đoán y khoa (medical imaging) giúp phát hiện sớm các bất thường trên X-quang hay MRI.

Các mô hình giải quyết bài toán Computer vision thường hoạt động trên dữ liệu dạng hình ảnh hoặc video.
Bên cạnh hình ảnh và video được chụp bằng máy ảnh thông thường, các mô hình computer vision còn có thể sử dụng những dạng hình ảnh đặc biệt như ảnh chụp x quang, ảnh sóng âm, ảnh sóng điện từ ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/data.gif" style="width: 1000px;"/>

Bên cạnh dữ liệu hình ảnh, các mô hình computer vision có thể còn sử dụng thêm các thông tin đến từ các dữ liệu loại khác để giúp gia tăng chất lượng của mô hình.

## 2. Các bài toán con của Computer vision

### 2.1. Phân lớp ảnh - Image classification:

Là bài toán phân lớp nhận đầu vào là ảnh và trả đầu ra là lớp tương ứng với ảnh đó.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/classification.jpeg" style="width: 1000px;"/>

### 2.2. Nhận diện đối tượng - Object detection:

Là bài toán kết hợp giữa hai bài toán object localization và object classification.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/object_detection.jpeg" style="width: 1000px;"/>

Trong đó, mô hình giải quyết bài toán object detection đầu tiên sẽ định vị vị trí có thể chứa đối tượng trong ảnh (object localization), sau đó mô hình sẽ thực hiện phân lớp đối tượng để nhận diện đối tượng đó là đối tượng nào (object classification)

### 2.3. Phân đoạn ảnh - Image segmentation:

Là bài toán phân lớp ảnh nhưng thay vì phân lớp toàn bộ ảnh, mô hình sẽ phân lớp từng pixel trong ảnh.
Bài toán này có thể chia thành hai dạng chính là semantic segmentation và instance segmentation.

**Semantic segmentation** là một dạng khác của bài toán classification, trong đó, mô hình, thay vì phân lớp trên cả ảnh, sẽ phân lớp từng pixel trên ảnh thuộc lớp nào

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/semantic_segmentation.jpeg" style="width: 1000px;"/>

**Instance segmentation:** là một phiên bản cao hơn của semantic segmentation, bên cạnh việc phân lớp các pixel trên ảnh thuộc lớp nào, đối với những pixel thuộc cùng một lớp, mô hình cần phải phân lớp rõ pixel đó thuộc đối tượng nào

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/instance_segmentation.jpeg" style="width: 1000px;"/>

### 2.4. Sinh ảnh - Image generation:

Là bài toán yêu cầu mô hình sinh ra dữ liệu ảnh mới từ một số điều kiện ban đầu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/image_synthesis_translation.jpeg" style="width: 1000px;"/>

Bên trong của bài toán Image generation gồm một số bài toán con như:

**Image synthesis:** sinh ảnh từ noise

**Image-to-image translation:** sinh ảnh mới từ ảnh ban đầu

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/text_to_image.jpeg" style="width: 500px;"/>

**Text-to-image translation:** sinh ảnh mới từ văn bản mô tả

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/inpainting_outpainting.jpeg" style="width: 500px;"/>

**Inpainting:** vẽ thêm hình ảnh vào khoảng trống trong ảnh ban đầu

**Outpainting:** mở rộng vẽ thêm hình ảnh từ một hình ảnh ban đầu

### 2.5. Đặt tiêu đề cho ảnh - Image captioning:

Là bài toán xây dựng một câu hoặc một đoạn văn mô tả một ảnh đầu vào.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/captioning.jpeg" style="width: 1000px;"/>

Bài toán này thường được giải quyết bằng cách kết hợp giữa mô hình xử lý ảnh và mô hình xử lý ngôn ngữ tự nhiên (NLP).
Image captioning là bài toán đòi hỏi mô hình vừa có khả năng hiểu và lưu giữ được thông tin hình ảnh và sinh ra đoạn văn bản mô tả đúng ngữ pháp và chân thực.

### 2.6. Xếp hạng ảnh - Image ranking:

Là bài toán sắp xếp các ảnh theo thứ tự ưu tiên dựa trên một số tiêu chí nhất định, có thể là độ liên quan, độ đẹp, độ phù hợp với một chủ đề nào đó ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/ranking.jpeg" style="width: 1000px;"/>

### 2.7. Nhóm các bài toán liên quan đến khuôn mặt - Face ecosystem:

Đối với một số dạng dữ liệu phổ biến và quan trọng như dữ liệu khuôn mặt, ta cũng có một số hệ sinh thái các bài toán như:
- **Face detection:** tương tự với object detection, tuy nhiên, bài toán face detection đòi hỏi mô hình đạt độ chính xác cao hơn và tốc độ chạy nhanh hơn, tiệm cận với real-time
- **Face attribute classification:** là bài toán phân tích các đặc tính trên khuôn mặt người như giới tính, cảm xúc, độ tuổi, chủng tộc ...
- **Face recognition:** là bài toán rất quan trọng và phổ biến hiện nay, face recognition là bài toán định danh khuôn mặt, đóng vai trò then chốt trong rất nhiều các hệ thống bảo mật hiện nay.
Face recognition gồm hai bài toán con là:
    - **Face indentification:** nhận đầu vào là một ảnh gương mặt người và trả đầu ra là định danh của gương mặt đó
    - **Face verification:** nhận đầu vào là hai ảnh gương mặt người và trả đầu ra là kết quả hai gương mặt có phải của cùng một người hay không.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/face_ecosystem.jpeg" style="width: 1000px;"/>

### 2.8. Phân cụm ảnh - Image clustering:

Là bài toán làm việc với hình ảnh không có nhãn, ta phân cụm các ảnh tương tự nhau thành một nhóm.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/clustering.jpeg" style="width: 1000px;"/>

### 2.9. Các bài toán liên quan đến video

Bên cạnh hình ảnh, video cũng là một loại dữ liệu quan trọng nhận được nhiều sự quan tâm trong computer vision.
Hầu như các bài toán trên dữ liêu hình ảnh đều có thể được áp dụng cho video như video classification, video object detection, video segmentation, video captioning ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/tracking.gif" style="width: 1000px;"/>

Nếu ta coi video là một chuỗi các hình ảnh được sắp xếp với nhau theo trình tự thời gian, ta hoàn toàn có thể sử dụng các mô hình xử lý dữ liệu ảnh để áp dụng cho video.
Tuy nhiên, rào cản và cũng là bài toán lớn nhất mà các mô hình xử lý video cần phải giải quyết là khối lượng tính toán lớn, dẫn đến thời gian xử lý lâu và chi phí vận hành hệ thống lớn.
Điều này là động lực để các nhà nghiên cứu phát triển những mô hình xử lý video với tốc độ cao và chất lượng được đảm bảo.

Một trong số những bài toán ứng phổ biến nhất đối với dữ liệu video là object tracking, theo dõi đối tượng trong suốt thời lượng của video.

## 3. Ảnh raster và ảnh vector trong máy tính

Ngày trước, hình ảnh sau khi được chụp sẽ được lưu giữ lại thông qua phim.
Ngày nay, với sự phát triển của máy tính, hình ảnh đã được lưu giữ lại thông qua các thiết bị số như thẻ nhớ, ổ cứng.

Trong lĩnh vực Computer Vision, hình ảnh đóng vai trò trung tâm.
Tuy nhiên, không phải tất cả “ảnh” đều giống nhau về bản chất và cách xử lý.
Hai dạng cơ bản thường gặp là ảnh raster (bitmap) và ảnh vector.

### 3.1. Ảnh vector

Ảnh vector mô tả hình học bằng các điểm, đường thẳng, đường cong và thuộc tính (màu sắc, đường viền).
Ảnh vector có thể phóng to vô hạn mà không mất độ nét.

Dữ liệu ảnh vector là tập hợp các thực thể đồ họa (đỉnh, cạnh, mặt, tham số đường cong).
Không dựa vào ma trận pixel nên không có khái niệm độ phân giải tuyệt đối.

Các định dạng phổ biến:
- **SVG** — định dạng XML cho web
- **EPS, PDF, AI** — thường dùng trong in ấn, thiết kế.

**Ưu điểm:**
- Vô hạn độ phóng to mà không vỡ nét.
- Kích thước tệp gọn gàng nếu đồ họa đơn giản.

**Nhược điểm:**
- Không phù hợp lưu trữ ảnh chụp với màu sắc phức tạp.
- Khó sử dụng trực tiếp trong các thuật toán pixel-based.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/vector_raster.jpeg" style="width: 500px;"/>

### 3.2. Ảnh raster (bitmap)

Ảnh raster (bitmap) là ma trận các điểm ảnh (pixel) sắp xếp theo hàng và cột.
Mỗi pixel lưu giá trị màu (grayscale hay RGB/RGBA) tại vị trí cố định.

Dữ liệu ảnh raster là một mảng hai chiều (W×H) với mỗi ô chứa 1–4 kênh màu.
Độ phân giải (resolution) xác định số pixel trên mỗi chiều.
Mỗi pixel là đơn vị nhỏ nhất, không thể phóng to mà không mất chất lượng.

Các định dạng ảnh raster phổ biến bao gồm:
- **PNG**: Hỗ trợ nén không mất dữ liệu, kênh alpha (trong suốt).
- **JPEG**: Nén mất dữ liệu, dung lượng nhỏ, phù hợp với ảnh chụp.
- **BMP**: Định dạng không nén, lưu trữ màu sắc đơn giản.
- **TIFF**: Hỗ trợ đa kênh, ít hoặc không nén, thường dùng trong in ấn và lưu trữ ảnh chất lượng cao.

**Ưu điểm:**
- Phù hợp với ảnh thực chụp, gradient, màu phức tạp.
- Dễ dùng trong hầu hết thư viện xử lý ảnh (OpenCV, PIL).

**Nhược điểm:**
- Không mở rộng vô hạn: phóng lớn gây vỡ hạt (pixelation).
- Kích thước tệp lớn ở độ phân giải cao.

## 4. Không gian màu Color Space

Không gian màu (color space) là cách thức biểu diễn màu sắc dưới dạng một hệ tọa độ đa chiều.
Mỗi điểm trong không gian màu tương ứng một màu duy nhất, xác định bởi bộ giá trị số (ví dụ trong RGB: R, G, B).

Vai trò của không gian màu trong computer vision rất quan trọng, vì nó ảnh hưởng đến cách mà máy tính hiểu và xử lý màu sắc trong hình ảnh.
- **Độc lập với chiếu sáng:**
Trong không gian RGB, giá trị màu thay đổi mạnh khi ánh sáng thay đổi.
Không gian HSV tách biệt kênh độ sáng (L, V) và kênh sắc độ (a/b hoặc H/S), giúp xử lý ổn định hơn.
- **Đơn giản hóa bài toán:**
Tìm miền màu đỏ bằng ngưỡng H trong HSV thay vì cùng lúc xét cả 3 kênh trong RGB.
- **Hiệu quả tính toán:**
Một số xử lý chỉ cần làm trên kênh sáng (grayscale) hoặc kênh sắc độ, giảm tải so với xử lý toàn bộ ba kênh RGB.

Hai không gian màu phổ biến nhất trong computer vision là RGB và HSV.
Một số không gian màu khác ít được sử dụng hơn như CIE LAB, CMYK ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/cymk.jpeg" style="width: 500px;"/>

### 4.1. Không gian màu RGB

RGB là không gian màu phổ biến dùng trong máy tính, máy ảnh, điện thoại và nhiều thiết bị kĩ thuật số khác.
Không gian màu này khá gần với cách mắt người tổng hợp màu sắc.

Nguyên lý cơ bản là sử dụng 3 màu sắc cơ bản R (red - đỏ), G (green - xanh lục) và B (blue - xanh lam) để biểu diễn tất cả các màu sắc.
Mỗi một màu sắc cơ bản sẽ gồm các giá trị từ 0 đến 255, do đó, số lượng màu tối đa thường được sử dụng là
$256 \times 256 \times 256 = 16,777,216$

Với việc mỗi điểm ảnh được cấu thành từ ba giá trị màu R - G - B, một ảnh bất kỳ sẽ được biểu diễn bởi một ma trận có kích thước $H \times W \times 3$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/rgb.jpeg" style="width: 500px;"/>

Trong một số thư viện như OpenCV, ảnh RGB thường được lưu trữ theo thứ tự BGR (Blue, Green, Red) thay vì RGB.

### 4.2. Không gian màu RGBA

RGBA là một biến thể của không gian màu RGB, trong đó thêm một kênh alpha (A) để biểu diễn độ trong suốt của từng vị trí trên ảnh (transparency).
Kênh alpha có giá trị từ 0 (hoàn toàn trong suốt) đến 255 (hoàn toàn không trong suốt).
Điều này cho phép ảnh có thể có các vùng trong suốt, giúp dễ dàng chồng ghép ảnh với nền khác.

### 4.4. Không gian màu HSV

Không gian màu HSV là một cách tự nhiên hơn để mô tả màu sắc, dựa trên 3 số liệu: H (Hue - Vùng màu), S (Saturation - Độ bão hòa) và V (Value - Độ sáng).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/hsv.jpeg" style="width: 500px;"/>

HSV thường được sử dụng khi ta cần phân tích kỹ hơn về vùng sáng vùng tối của ảnh hay vùng có độ tương phản cao và vùng có độ tương phản thấp.

## 4. Workflow dự án Computer vision

Tương tự như workflow nói chung trong quá trình xây dựng mô hình deep learning, computer vision workflow cũng trải qua một số bước như **chuẩn bị dữ liệu**, **tiền xử lý dữ liệu**, **xây dựng mô hình**, **huấn luyện** và **đánh giá mô hình**, **kiểm thử mô hình**, **triển khai mô hình**.

Tuy nhiên, đối với computer vision, bước tiền xử lý dữ liệu đòi hỏi những thao tác đặc thù với dữ liệu hình ảnh.
Những thao tác này được gọi chung là **Image processing**.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/workflow.jpeg" style="width: 1000px;"/>

Trong thời gian trước, các kỹ thuật Image processing đã phát triển đến mức có thể giúp chúng ta giải quyết một số bài toán với dữ liệu đơn giản mà không cần xây dựng các mô hình machine learning.
Ngày nay, với sự tiện dụng trong việc xây dựng các mô hình machine learning, việc trực tiếp sử dụng các kỹ thuật image processing trong việc giải các bài toán đã không còn quá phổ biến.

Tuy nhiên, các kỹ thuật image processing vẫn đóng vai trò rất quan trọng trong việc làm sạch dữ liệu ảnh, gia tăng dữ liệu ảnh và từ đó, cải thiện độ chính xác của các mô hình machine learning hay deep learning trên dữ liệu ảnh.
