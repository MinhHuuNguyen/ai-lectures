---
time: 02/16/2023
title: Âm thanh máy tính Computer Audition
description: Computer audition là tổ hợp các bài toán con trong lĩnh vực trí tuệ nhân tạo, nhằm giúp máy tính có thể hiểu và xử lý âm thanh. Computer audition là một trong những lĩnh vực nghiên cứu có rất nhiều ứng dụng thực tiễn trong đời sống giúp nâng cao hiệu quả công việc của con người và tự động hoá nhiều quy trình.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/7_computer_audition/images/1-computer-audition/banner.png
tags: [deep-learning, computer-audition]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về lĩnh vực Computer audition

Computer Audition (Âm thanh máy tính) là một lĩnh vực nghiên cứu và ứng dụng của trí tuệ nhân tạo (AI) và xử lý âm thanh, với mục tiêu cho phép máy tính “nghe thấy” và hiểu được nội dung của âm thanh tương tự như con người.

Âm thanh được tạo ra khi một vật nào đó rung lên, dẫn đến các phân tử không khí va chạm vào nhau.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/7_computer_audition/images/1-computer-audition/pressure.png" style="width: 800px;"/>

Sự va chạm này dẫn đến việc có những điểm trong không gian có áp suất lớn hơn so với những điểm khác, điều này tạo ra sóng âm cơ học trong không khí.
Sóng này truyền tải năng lượng từ điểm này đến điểm kia trong không gian.

Đây cũng là lý do tại sao trong chân không, chúng ta không nghe thấy âm thanh.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/7_computer_audition/images/1-computer-audition/vacuum.png" style="width: 800px;"/>

## 2. Các bài toán con của Computer audition

### 2.1. Phân lớp âm thanh - Audio classification

Là bài toán phân lớp nhận đầu vào là đoạn âm thanh và trả đầu ra là lớp tương ứng với đoạn âm thanh đó.

### 2.2. Phân tách âm thanh - Audio separation and segmentation

Là bài toán phân tách các đối tượng trong một đoạn âm thanh.

Bài toán này có ứng dụng trong việc tách lời nói của các nhân vật từ một đoạn hội thoại, tách lời và nhạc từ bài hát, tách nội dung và âm thanh background từ đoạn âm thanh ...

### 2.3. Sinh âm thanh - Audio generation

Là bài toán yêu cầu mô hình sinh ra dữ liệu âm thanh mới từ một số điều kiện ban đầu.

Bên trong của bài toán Audio generation gồm một số bài toán con như:
- **Audio synthesis:** sinh âm thanh từ noise.
- **Text to speech:** sinh âm thanh từ văn bản, là bài toán phổ biến nhất do nó có thể kế thừa kết quả của các mô hình xử lý ngôn ngữ tự nhiên NLP.
- **Music generation:** sinh âm thanh từ một số điều kiện ban đầu, ví dụ như sinh nhạc từ một đoạn nhạc đã có sẵn, sinh nhạc từ một chủ đề nhất định ...

### 2.4. Speech to text

Là bài toán quan trọng nhất trong xử lý âm thanh, mang lại một hướng tiếp cận mới cho hầu hết các bài toán xử lý âm thanh.

Thay vì ta giải quyết các bài toán trên dữ liệu audio, nếu ta xây dựng được một mô hình Speech to text tốt, ta hoàn toàn có thể giải quyết các bài toán đó thông qua các mô hình xử lý ngôn ngữ tự nhiên NLP.
Lấy ví dụ, thay vì phân loại một đoạn âm thanh, ta có thể chuyển đổi đoạn âm thanh đó thành văn bản và phân loại văn bản đó.

## 3. Tín hiệu analog và tín hiệu số

Tất cả âm thanh được thu lại dưới dạng tín hiệu analog.

Tín hiệu analog là một biểu đồ giữa thời gian và cường độ của âm thanh.
Trong đó, mỗi giá trị cường độ là giá trị số thực, và mỗi giá trị cường độ thu được trong một đơn vị thời gian vô cùng nhỏ.
Điều này dẫn đến việc lưu trữ tín hiệu analog là gần như bất khả thi do nó đòi hỏi bộ nhớ lưu trữ vô cùng lớn.

Do đó, ta cần một bước chuyển đổi tín hiệu analog thành tín hiệu số sao cho tối ưu khả năng lữu trữ nhưng vẫn đảm bảo được tính tái tạo của âm thanh.
Bước nay được gọi là Analog to Digital Conversion (ADC).

Có hai kỹ thuật thường được dùng trong ADC: Sampling và Quantization.

### 3.1. Sampling

Thay vì việc thu thập tất cả các giá trị trong tín hiệu analog liên tục, ý tưởng của Sampling là việc lấy ra các giá trị theo từng khoảng cố định và đều nhau.

Thông thường, ta lấy 44,100 giá trị mỗi giây của âm thanh, lúc này, âm thanh sẽ có tần số là 44.1 kHz.
Giá trị này giúp ta xây dựng âm thanh phù hợp nhất với ngưỡng nghe của con người.

### 3.2. Quantization

Trong khi kỹ thuật Sampling lấy các giá trị đều theo các khoảng thời gian, kỹ thuật Qauntization chia đều giá trị cường độ của âm thanh thành các khoảng bằng nhau, cụ thể là các giá trị nguyên.

Từ đó, khi ta thu thập cường độ của âm thanh tại một mốc thời gian bất kỳ (các mốc thời gian không nhất thiết phải cách đều nhau), ta sẽ lấy giá trị cường độ nguyên gần nhất với giá trị cường độ đúng tại mốc thời gian đó.

## 4. Các đặc trưng của âm thanh

### 4.1. Chu kỳ (Period) - Tần số (Frequency)

Chu kỳ của sóng cơ học là thời gian hoàn thành một bước sóng, được đo bằng đơn vị giây (s).

Tần số là nghịch đảo của chu kỳ, là số lượng bước sóng hoàn thành trong một đơn vị thời gian, được đo bằng đơn vị Hertz (Hz).

Tần số càng lớn thì chu kỳ càng nhỏ và ngược lại.

Tần số là một trong những đặc trưng quan trọng nhất của âm thanh.
Trong thực tế, ta cảm nhận được tần số thông qua cao độ của âm thanh, âm thanh có cao độ càng cao thì tần số càng lớn và ngược lại.

Ngưỡng nghe của con người là từ 20 Hz đến 20 kHz.
Tần số dưới 20 Hz được gọi là Infrasound, tần số trên 20 kHz được gọi là Ultrasound.
Con người không thể nghe được âm thanh ở hai ngưỡng Infrasound và Ultrasound, tuy nhiên, một số loài động vật có thể nghe được âm thanh ở hai ngưỡng này.

### 4.2. Cường độ (Intensity)

Cường độ cũng là một đại lượng quan trọng trong quá trình xử lý âm thanh.
Cường độ của âm thanh đo đạc năng lượng của âm thanh truyền đi, được đo bằng Watt/$m^2$

Trong thực tế, ta cảm nhận được cường độ thông qua độ to của âm thanh, âm thanh càng to thì cường độ càng lớn và ngược lại.

### 4.3. Âm sắc (Timbre)

Âm sắc là một khái niệm khá trừu tượng và khó đo đạc trong xử lý âm thanh.

Giả sử một chiếc kèn và một chiếc violin chơi một đoạn nhạc có cùng tần số, cường độ nhưng ta vẫn có thể phân biệt được đâu là âm thanh từ chiếc kèn và đâu là âm thanh từ chiếc violin.

Ta phân biệt được điều này thông qua âm sắc của từng âm thanh.

### 4.4. Spectrograms

#### Spectrum

Một âm thanh mà chúng ta vẫn nghe là kết quả của nhiều các tín hiệu âm thanh khác nhau tổng hợp lại.
Điều này có nghĩa là ta có thể phân rã sóng âm tổng hợp mà ta nghe được thành các sóng âm đơn có tần số cụ thể.
Ta có thể phân rã sóng âm tổng hợp thành các sóng âm đơn thành phần dựa vào Fourier Transforms.

Spectrum là tập hợp của các tần số mà kết hợp lại với nhau tạo thành một âm thanh nào đó.
Biểu đồ spectrum bao gồm giá trị tần số tương ứng với từng sóng đơn và biên độ của chúng.

Tần số nhỏ nhất trong spectrum được gọi là tần số nền tảng (fundamental frequency), các tần số là bội số của tần số nền tảng được gọi là harmonic frequency.

#### Time Domain và Frequency Domain

Hình ảnh biểu diễn sóng âm mà ta thường quan sát là mối quan hệ giữa biên độ và thời gian:
- Trục x biểu diễn khoảng thời gian
- Trục y biểu diễn biên độ theo từng thời điểm trên trục thời gian.
Cách biểu diễn này được gọi là Time Domain.

Hình ảnh của spectrum lại biểu diễn một góc nhìn khác của âm thanh, mô tả mối quan hệ giữa biên độ và tần số:
- Trục x biểu diễn tần số
- Trục y biểu diễn biên độ tương ứng với từng giá trị tần số
Cách biểu diễn này được gọi là Frequency Domain.
Khi ta quan sát được Frequency Domain, ta đang quan sát nó trong một khoảnh khắc trên trục thời gian.

#### Spectrograms

Đối với machine learning cổ điển, việc xử lý audio khá phức tạp và đòi hỏi ta cần hiểu khá sâu về chuyên môn của âm thanh để thực hiện feature extraction.

Tuy nhiên, với sự phát triển của deep learning và đặc biệt là computer vision, việc xử lý audio trở nên đơn giản hơn nếu ta có thể chuyển hoá được âm thanh về dạng hình ảnh và tận dụng sức mạnh của các mô hình CNN.

Ta hoàn toàn có thể chuyển hoá được audio thành hình ảnh thông qua Spectrograms và xử lý audio thông qua xử lý hình ảnh của Spectrograms.

Kết hợp thông tin từ Time Domain và Frequency Domain, ta thu được Spectrograms.
- Trục x là thời gian
- Trục y là tần số
- Màu sắc trên Spectrograms có thể được coi như là trục thứ 3 biểu diễn biên độ của âm thanh.
    - Ở những điểm màu sáng, ta có biên độ của sóng âm **LỚN** ở tần số đó và thời điểm đó.
    - Ở những điểm màu tối, ta có biên độ của sóng âm **NHỎ** ở tần số đó và thời điểm đó.

Rõ ràng, ta thấy với việc kết hợp thêm thông tin về tần số, ta thu được hình ảnh về Spectrograms chứa nhiều thông tin hơn so với hình ảnh biểu diễn sóng âm như thông thường (chỉ gồm thông tin về biên độ theo thời gian).

Với sự xuất hiện của Spectrograms, ta có thể dễ dàng sử dụng các mô hình machine learning xử lý ảnh hoặc deep learning để xử lý dữ liệu âm thanh và vẫn đạt hiệu quả cao.

### 4.6. Mel-spectrograms

### 4.7. Mel-frequency Cepstral Coefficients (MFCC)

## 5. Workflow dự án Computer audition








