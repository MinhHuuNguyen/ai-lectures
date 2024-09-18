---
slug: machine-learning
time: "11/09/2024"
title: "Machine learning"
description: "Machine Learning (ML) là một phần của trí tuệ nhân tạo (AI) mà chúng ta dùng để xây dựng các mô hình hoặc chương trình máy tính có khả năng tự học từ dữ liệu."
author: "Nguyễn Hữu Minh"
banner_url: "https://tenten.vn/tin-tuc/wp-content/uploads/2023/08/1cG6U1qstYDijh9bPL42e-Q.jpg"
tags:
---

# Audio processing trong Machine learning

## 1. Âm thanh dưới góc độ vật lý
Âm thanh được tạo ra khi một vật nào đó rung lên, dẫn đến các phân tử không khí va chạm vào nhau.
Sự va chạm này dẫn đến việc có những điểm trong không gian có áp suất lớn hơn so với những điểm khác, điều này tạo ra sóng âm cơ học trong không khí.
Sóng này truyền tải năng lượng từ điểm này đến điểm kia trong không gian.
Đây cũng là lý do tại sao trong chân không, chúng ta không nghe thấy âm thanh.

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121942&authkey=%21APYyLdlB_40Bpa0" style="width: 600px;"/>

### 1.1. Chu kỳ (Period) - Tần số (Frequency)

Chu kỳ của sóng cơ học là thời gian hoàn thành một bước sóng, được đo bằng giây. 
Tần số là nghịch đảo của chu kỳ, là số lượng bước sóng hoàn thành trong một đơn vị thời gian, được đo bằng Hz.

Trong thực tế, ta cảm nhận được tần số thông qua cao độ của âm thanh, âm thanh có cao độ càng cao thì tần số càng lớn và ngược lại.

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121943&authkey=%21AAF_Eb65tBewhzc" style="width: 600px;"/>

### 1.2. Cường độ (Intensity)

Cường độ cũng là một đại lượng quan trọng trong quá trình xử lý âm thanh.
Cường độ của âm thanh đo đạc năng lượng của âm thanh truyền đi, được đo bằng Watt/$m^2$

Trong thực tế, ta cảm nhận được cường độ thông qua độ to của âm thanh, âm thanh càng to thì cường độ càng lớn và ngược lại.

### 1.3. Âm sắc (Timbre)

Âm sắc là một khái niệm khá trừu tượng và khó đo đạc trong xử lý âm thanh.
Giả sử một chiếc kèn và một chiếc violin chơi một đoạn nhạc có cùng tần số, cường độ nhưng ta vẫn có thể phân biệt được đâu là âm thanh từ chiếc kèn và đâu là âm thanh từ chiếc violin.
Ta phân biệt được điều này thông qua âm sắc của từng âm thanh.

## 2. Từ tín hiệu analog đến tín hiệu số

Tất cả âm thanh thu được dưới tín hiệu analog.
Tín hiệu analog là một biểu đồ giữa thời gian và cường độ của âm thanh.
Trong đó, mỗi giá trị cường độ là giá trị số thực, và mỗi giá trị cường độ thu được trong một đơn vị thời gian vô cùng nhỏ.
Điều này dẫn đến việc lưu trữ tín hiệu analog là gần như bất khả thi do nó đòi hỏi bộ nhớ lưu trữ vô cùng lớn.

Do đó, ta cần một bước chuyển đổi tín hiệu analog thành tín hiệu số sao cho tối ưu khả năng lữu trữ nhưng vẫn đảm bảo được tính tái tạo của âm thanh.
Bước nay được gọi là Analog to Digital Conversion (ADC).
Có hai kỹ thuật thường được dùng trong ADC: sampling và quantization.

### 2.1. Sampling

Thay vì việc thu thập tất cả các giá trị trong tín hiệu analog liên tục, ý tưởng của Sampling là việc lấy ra các giá trị theo từng khoảng cố định và đều nhau.

Thông thường, ta lấy 44,100 giá trị mỗi giây của âm thanh, lúc này, âm thanh sẽ có tần số là 44.1 kHz.
Giá trị này giúp ta xây dựng âm thanh phù hợp nhất với ngưỡng nghe của con người.

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121944&authkey=%21AKPtJcqSeIFl-so" style="width: 700px;"/>

### 2.2. Quantization

Trong khi kỹ thuật Sampling chia đều các khoảng thời gian, kỹ thuật Qauntization chia đều giá trị cường độ của âm thanh thành các khoảng bằng nhau, cụ thể là các giá trị nguyên.
Từ đó, khi ta thu thập cường độ của âm thanh tại một mốc thời gian bất kỳ (các mốc thời gian không nhất thiết phải cách đều nhau), ta sẽ lấy giá trị cường độ nguyên gần nhất với giá trị cường độ đúng tại mốc thời gian đó.

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121945&authkey=%21ANeZGReiIrTGDWw" style="width: 700px;"/>

## 3. Spectrograms

Đối với machine learning cổ điển, việc xử lý audio khá phức tạp và đòi hỏi ta cần hiểu khá sâu về chuyên môn của âm thanh để thực hiện feature extraction.
Tuy nhiên, với sự phát triển của deep learning và đặc biệt là computer vision, việc xử lý audio trở nên đơn giản hơn nếu ta có thể chuyển hoá được âm thanh về dạng hình ảnh và tận dụng sức mạnh của các mô hình CNN.
Ta hoàn toàn có thể chuyển hoá được audio thành hình ảnh thông qua Spectrograms và xử lý audio thông qua xử lý hình ảnh của Spectrograms.

### 3.1. Spectrum

Một âm thanh mà chúng ta vẫn nghe là kết quả của nhiều các tín hiệu âm thanh khác nhau tổng hợp lại.
Điều này có nghĩa là ta có thể phân rã sóng âm tổng hợp mà ta nghe được thành các sóng âm đơn có tần số cụ thể.
Ta có thể phân rã sóng âm tổng hợp thành các sóng âm đơn thành phần dựa vào Fourier Transforms.

Spectrum là tập hợp của các tần số mà kết hợp lại với nhau tạo thành một âm thanh nào đó.
Biểu đồ spectrum bao gồm giá trị tần số tương ứng với từng sóng đơn và biên độ của chúng.

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121946&authkey=%21ABhdPYPhsu5he0w" style="width: 500px;"/>

Tần số nhỏ nhất trong spectrum được gọi là tần số nền tảng (fundamental frequency), các tần số là bội số của tần số nền tảng được gọi là harmonic frequency.

### 3.2. Time Domain vs Frequency Domain

Hình ảnh biểu diễn sóng âm mà ta thường quan sát là mối quan hệ giữa biên độ và thời gian (trục x biểu diễn khoảng thời gian, trục y biểu diễn biên độ theo từng thời điểm trên trục thời gian).
Cách biểu diễn này được gọi là Time Domain.

Hình ảnh của spectrum lại biểu diễn một góc nhìn khác của âm thanh, mô tả mối quan hệ giữa biên độ và tần số (trục x biểu diễn tần số, trục y biểu diễn biên độ tương ứng với từng giá trị tần số).
Cách biểu diễn này được gọi là Frequency Domain.
Khi ta quan sát được Frequency Domain, ta đang quan sát nó trong một khoảnh khắc trên trục thời gian.

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121947&authkey=%21APfQDuKO3zxumA4" style="width: 500px;"/>


### 3.3. Spectrograms

Kết hợp thông tin từ Time Domain và Frequency Domain, ta thu được Spectrograms.
Trên hình ảnh của Spectrograms, ta có trục x là thời gian, trục y là tần số, màu sắc trên Spectrograms có thể được coi như là trục thứ 3 biểu diễn biên độ của âm thanh.
Ở những điểm màu sáng, ta có biên độ của sóng ở tần số đó và thời điểm đó lớn, và ngược lại, ở những điểm màu tối, ta có biên độ của sóng ở tần số đó và thời điểm đó nhỏ.

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121948&authkey=%21AP4Xr3EZ9t6-5tg" style="width: 500px;"/>

Rõ ràng, ta thấy với việc kết hợp thêm thông tin về tần số, ta thu được hình ảnh về Spectrograms chứa nhiều thông tin hơn so với hình ảnh biểu diễn sóng âm như thông thường (chỉ gồm thông tin về biên độ theo thời gian).

Với sự xuất hiện của Spectrograms, ta có thể dễ dàng sử dụng các mô hình machine learning xử lý ảnh hoặc deep learning để xử lý dữ liệu âm thanh và vẫn đạt hiệu quả cao.

<!-- Bổ sung Mel Spectrograms -->

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121949&authkey=%21AF2UjWPyJwm-WxA" style="height: 600px;"/>

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121950&authkey=%21ABqdOXyQUp8nGFM" style="height: 600px;"/>

## 4. Các bài toán Audio processing

- Audio classification: là bài toán phân lớp các đoạn audio thành càng lớp khác nhau. Bài toán này có ứng dụng nhiều trong việc nhận diện và đánh giá cảm xúc của khách hàng thông qua lời feedback

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121951&authkey=%21AIXPGsHwE1MdRmI" style="width: 500px;"/>

- Audio separation and segmentation: là bài toán phân tách các đối tượng trong một đoạn âm thanh. Bài toán này có ứng dụng trong việc tách lời nói của các nhân vật từ cùng một đoạn audio, tách lời và nhạc từ bài hát, tách nội dung và âm thanh background từ đoạn âm thanh ...

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121952&authkey=%21AKZ3waFat9iz0IY" style="width: 500px;"/>

- Audio generation: là bài toán sinh ra các đoạn audio. Bài toán này có thể nhận đầu vào là các dạng dữ liệu khác nhau như đoạn văn bản (Text to speech), hình ảnh nốt nhạc, hoặc đơn giản là nhiễu. Trong đó, bài toán Text to speech là bài toán phổ biến nhất do nó có thể kế thừa kết quả của các mô hình xử lý ngôn ngữ tự nhiên NLP

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121953&authkey=%21ABx-ovyfRlzj3lI" style="width: 500px;"/>

- Speech to text: Đây là hai bài toán quan trọng nhất trong xử lý âm thanh. Bài toán speech to text mang lại một hướng tiếp cận mới cho hầu hết các bài toán xử lý âm thanh. Thay vì ta giải quyết các bài toán trên dữ liệu audio, nếu ta xây dựng được một mô hình Speech to text tốt, ta hoàn toàn có thể giải quyết các bài toán đó thông qua các mô hình NLP

<img src="https://onedrive.live.com/embed?resid=55F936846CC480BE%2121954&authkey=%21AMqjRWtH7utfIVA" style="width: 500px;"/>
