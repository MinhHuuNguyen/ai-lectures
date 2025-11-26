---
time: 01/03/2023
title: Thuật toán Maximum Likelihood Estimation (MLE) và Maximum A Posteriori (MAP)
description: Maximum Likelihood Estimation (MLE) và Maximum A Posteriori (MAP) là hai phương pháp thống kê quan trọng trong machine learning dùng để ước lượng tham số mô hình. MLE tìm giá trị tham số làm cực đại xác suất quan sát dữ liệu, tập trung hoàn toàn vào thông tin từ dữ liệu huấn luyện. Trong khi đó, MAP kết hợp cả dữ liệu và kiến thức tiên nghiệm (prior) thông qua định lý Bayes, cho phép ước lượng ổn định hơn khi dữ liệu hạn chế hoặc nhiễu. Cả hai phương pháp đều đóng vai trò cốt lõi trong các mô hình xác suất và suy luận Bayes hiện đại.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/banner.jpeg
tags: [math]
is_highlight: false
is_published: true
---

## 1. Data (dữ liệu), Model (mô hình) và Learning (học)

Trong một hệ thống machine learning, có ba thành phần cốt lõi: **Data (dữ liệu)**, **Model (mô hình)** và **Learning (học)**.
Câu hỏi trung tâm của machine learning là: "Thế nào là một mô hình tốt?".

Một trong những nguyên tắc quan trọng của machine learning là mô hình tốt phải duy trì hiệu năng cao trên dữ liệu chưa từng thấy.
Điều này đòi hỏi chúng ta phải xác định các chỉ số đánh giá phù hợp, chẳng hạn như độ chính xác hoặc sai lệch so với giá trị thực, đồng thời tìm ra phương pháp tối ưu hóa mô hình theo các chỉ số đó.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/6-mle-map/problem.jpeg" style="width: 600px;"/>

Ngoài ra, thuật ngữ “thuật toán machine learning” thường được sử dụng theo hai nghĩa:
- Thuật toán phục vụ giai đoạn huấn luyện, nơi mô hình được học từ dữ liệu.
- Thuật toán phục vụ giai đoạn dự đoán, nơi mô hình đã huấn luyện được sử dụng để đưa ra kết quả trên dữ liệu mới.

### 1.1. Dữ liệu dưới dạng vector

Giả định rằng dữ liệu có thể được máy tính đọc và được biểu diễn đầy đủ dưới dạng số.
Đối với các loại dữ liệu khác nhau như hình ảnh, âm thanh, video, văn bản, tín hiệu ..., dữ liệu sẽ được xử lý và mã hoá về dạng số theo các phương pháp khác nhau.
Tuy nhiên, để đơn giản hoá, dữ liệu được xem như dữ liệu bảng, trong đó mỗi hàng tương ứng với một mẫu cụ thể và mỗi cột đại diện cho một đặc trưng.

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf) mô tả bộ dữ liệu trước và sau khi được xử lý.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/6-mle-map/data_raw.jpeg" style="width: 800px;"/>

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/6-mle-map/data_number.jpeg" style="width: 800px;"/>

Từ đó, mỗi mẫu dữ liệu có thể được biểu diễn dưới dạng một vector trong không gian đa chiều.
Trong đó, mỗi chiều tương ứng với một đặc trưng của mẫu dữ liệu.

### 1.2. Mô hình dưới dạng hàm số

Khi dữ liệu đã được biểu diễn dưới dạng vector phù hợp, chúng ta có thể bắt đầu xây dựng một hàm dự đoán (gọi là predictor).
Ta có hai cách tiếp cận chính: xem **predictor như một hàm số** hoặc xem **predictor như một mô hình xác suất**.

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf) mô tả cách nhìn mô hình machine learning dưới dạng hàm số, cụ thể là đồ thị của hàm số dự đoán.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/6-mle-map/model_as_function.jpeg" style="width: 600px;"/>

Mình đã có bài viết mô tả khá chi tiết về cách nhìn và xây dựng một mô hình Machine Learning dưới dạng như hàm số, bạn có thể tham khảo bài viết [Mô hình hồi quy tuyến tính Linear Regression](/blog/mo-hinh-hoi-quy-tuyen-tinh-linear-regression).

### 1.3. Mô hình học là tìm tham số tối ưu cho mô hình

Mục tiêu của việc học (learning) là tìm được bộ tham số tối ưu cho mô hình, giúp mô hình có thể dự đoán chính xác nhất trên dữ liệu chưa từng thấy.

Quá trình học thường được chia thành ba giai đoạn chính:
- **Training (hoặc parameter estimation):** Sử dụng dữ liệu huấn luyện để tìm bộ tham số tối ưu cho mô hình, nói cách khác, là xây dựng được một predictor tốt nhất dựa trên dữ liệu đã biết.
- **Hyperparameter tuning (hoặc model selection):** Nhằm lựa chọn các siêu tham số (hyperparameters) của mô hình, những tham số này không được học trực tiếp từ dữ liệu huấn luyện mà được thiết lập trước.
Quá trình này thường sử dụng tập dữ liệu xác thực (validation set) để đánh giá hiệu năng của mô hình với các cấu hình siêu tham số khác nhau và chọn ra cấu hình tốt nhất.
- **Prediction (hoặc inference):** sử dụng một mô hình đã được huấn luyện để xử lý dữ liệu kiểm tra chưa từng thấy trước đó.
Nói cách khác, các tham số và lựa chọn mô hình đã được cố định, và predictor được áp dụng lên các vector mới đại diện cho những điểm dữ liệu đầu vào mới.

Training và Hyperparameter tuning thường được thực hiện nhiều lần trong quá trình phát triển mô hình và dành khá nhiều thời gian, nhằm tối ưu hoá hiệu năng của mô hình trước khi áp dụng vào giai đoạn Prediction.

Có hai cách đánh giá tham số thường được dùng là **Maximum Likelihood Estimation (MLE)** và **Maximum A Posteriori Estimation (MAP Estimation)**.
- Maximum Likelihood Estimation chỉ dựa trên dữ liệu đã biết trong tập dữ liệu huấn luyện (training data).
- Maximum A Posteriori Estimation không những dựa trên training data mà còn dựa trên những thông tin đã biết của các tham số.
Những thông tin này càng rõ ràng, càng hợp lý thì khả năng thu được bộ tham số tốt là càng cao.

## 2. Maximum likelihood estimation (MLE)

### 2.1. Ý tưởng chung

Với MLE, ta sẽ bắt đầu với một bộ dữ liệu huấn luyện gồm có N phần tử $X = {x_1, x_2, ..., x_N}$.
Ta giả sử rằng bộ dữ liệu này tuân theo một phân phối xác suất nào đó, và xây dựng được mô hình Machine Learning thống kê được đại diện bởi bộ tham số $\theta$.

Mục tiêu ở đây là tìm được mô hình thống kê hay cụ thể hơn là bộ tham số $\theta$ sao cho có thể mô tả được chính xác nhất bộ dữ liệu $X$.
Maximum Likelihood Estimation là việc đi tìm bộ tham số $\theta$ sao cho xác suất sau đây đạt giá trị lớn nhất, giá trị xác suất này được gọi là **likelihood**:

$$
\theta = \arg\max_{\theta} p(\mathbf{x}_1, ..., \mathbf{x}_N | \theta)
$$

- $p(\mathbf{x}_1| \theta)$ là xác suất mà điểm dữ liệu $x_1$ xuất hiện với điều kiện là mô hình sử dụng bộ tham số $\theta$
- $p(\mathbf{x}_1, ..., \mathbf{x}_N | \theta)$ là xác suất mà toàn bộ bộ dữ liệu $X$ cùng xuất hiện với điều kiện là mô hình sử dụng bộ tham số $\theta$.

Ta đi tìm tham số $\theta$ để cực đại hoá likelihood, đây là lý do tại sao phương pháp này được gọi là **Maximum Likelihood Estimation (MLE)**, chính là cách để ta tìm tham số $\theta$ sao cho tạo ra được mô hình xác suất phản ảnh đúng nhất bộ dữ liệu huấn luyện cho trước.

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf) mô tả mô hình machine learning được xây dựng dựa trên MLE.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/6-mle-map/mle.jpeg" style="width: 600px;"/>

### 2.2. Giải bài toán tối ưu của MLE

Việc trực tiếp giải bài toán tối ưu trên thường rất phức tạp do tính phụ thuộc của các điểm dữ liệu trong bộ dữ liệu.
Nói cách khác, xác suất xuất hiện của điểm dữ liệu này có thể phụ thuộc vào điểm dữ liệu khác dẫn đến việc tối ưu hoá xác suất đồng thời của các điểm dữ liệu gặp khó khăn.

Từ đó, để đơn giản hoá quá trình tối ưu, ta cần lập một giả sử các điểm dữ liệu trong bộ dữ liệu $X$ độc lập với nhau.
Khi các điểm dữ liệu được coi là độc lập với nhau, xác suất đồng thời của các điểm dữ liệu được tính bằng tích các xác suất của từng điểm dữ liệu.
Từ đó, ta có biểu thức:

$$
p(\mathbf{x}_1, ..., \mathbf{x}_N | \theta) \approx \prod_{n = 1}^N p(\mathbf{x}_n |\theta)
$$

Tuy nhiên, việc tối ưu một tích các giá trị xác suất thường khó khăn hơn việc tối ưu một tổng (do tích các xác suất có thể dẫn tới lỗi số học trong máy tính).
Do đó, ta sẽ biến đổi phép nhân thành phép cộng thông qua việc sử dụng **hàm logarit**:
- **log** của một tích bằng tổng của các log.
- **log** là một hàm đồng biến, một biểu thức sẽ là lớn nhất nếu log của nó là lớn nhất, và ngược lại.

$$
\theta = \arg\max_{\theta} \sum_{n=1}^N \log p(\mathbf{x}_n | \theta)
$$

Lúc này, ta gọi biểu thức trên là **log-likelihood** và ta sẽ đi tìm tham số $\theta$ sao cho log-likelihood đạt giá trị lớn nhất.

Maximum Likelihood Estimation (MLE) tối ưu hoá trực tiếp log-likelihood để tìm tham số $\theta$ thông qua các phương pháp như:
- Giải phương trình đạo hàm bằng 0.
- Sử dụng các thuật toán tối ưu hoá số như Gradient Descent, Newton's Method, Expectation-Maximization (EM) ...

Việc lựa chọn thuật toán dựa trên cấu trúc mô hình (ví dụ dựa trên phân phối Gaussian hay phân phối Bernoulli), kích thước dữ liệu và yêu cầu về độ chính xác/hiệu quả.

## 3. Maximum a posteriori estimation (MAP)

### 3.1. Ý tưởng chung

Với MLE, việc xây dựng mô hình và tìm tham số $\theta$ chỉ phụ thuộc vào bộ dữ liệu huấn luyện.
Khi bộ dữ liệu này có vấn đề, hiển nhiên tham số $\theta$ tìm được cũng sẽ không chính xác.

Ví dụ đối với bộ dữ liệu thống kê về kết quả thu được khi tung đồng xu.
Bộ dữ liệu ghi nhận trong 10.000 lần tung đồng xu, có 8.000 lần ra mặt ngửa và 2.000 lần ra mặt sấp.
Tỷ lệ ra mặt sấp lúc này là 1/5 = 20%.

Tuy nhiên, với kiến thức của chúng ta, tỷ lệ ra mặt sấp và ra mặt ngửa đối khi tung đồng xu ra tương đối cân bằng nhau, loanh quanh ngưỡng 50%.
Do đó, nếu ta xây dựng mô hình MLE trên bộ dữ liệu này thì có thể ta sẽ mắc phải hiện tượng overfitting.

Trong một số trường hợp cụ thể, bên cạnh việc xây dựng mô hình và lựa chọn tham số dựa trên bộ dữ liệu huấn luyện, ta còn cần phải định nghĩa trước một số kiến thức cho mô hình, và từ đó ngăn chặn việc mô hình quá phụ thuộc và bộ dữ liệu huấn luyện dẫn đến sai sót trong kết quả.

Ngược lại với MLE, đối với MAP, ta đi tìm bộ tham số $\theta$ sao cho xác suất sau đây đạt giá trị lớn nhất, giá trị xác suất này được gọi là **posterior**:

$$
\theta = \arg\max_{\theta} p(\theta | \mathbf{x}_1, ..., \mathbf{x}_N)
$$

Xác suất posterior là xác suất được điều chỉnh hoặc cập nhật của một biến cố xảy ra sau khi xem xét thêm các thông tin khác.

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf) mô tả mô hình machine learning được xây dựng dựa trên MAP.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/6-mle-map/map.jpeg" style="width: 600px;"/>

Sự khác biệt chính giữa MLE và MAP nằm ở chỗ:
- Likelihood trong MLE chỉ dựa trên bộ dữ liệu huấn luyện để tìm tham số $\theta$.
- Posterior trong MAP không những dựa trên bộ dữ liệu huấn luyện mà còn dựa trên những kiến thức đã biết trước về tham số $\theta$.

**Vậy, tại sao posterior lại bao gồm cả likelihood và những kiến thức đã biết trước về tham số $\theta$?**

### 3.2. Prior knowledge trong MAP

Ta sẽ ký hiệu gọn lại bộ dữ liệu huấn luyện gồm N phần tử là $\mathbf{X} = {\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_N}$.

$$
\theta = \arg\max_{\theta} p(\theta | \mathbf{x}_1, ..., \mathbf{x}_N) = \arg\max_{\theta} p(\theta | \mathbf{X})
$$

Áp dụng quy tắc Bayes

$$
\theta = \arg\max_{\theta} p(\theta | \mathbf{X}) =  \arg\max_{\theta} \left[ \frac{p(\mathbf{X} | \theta) p(\theta)}{p(\mathbf{X})} \right]
$$
- Trong đó, $p(\mathbf{X} | \theta)$ chính là likelihood đã được đề cập trong MLE.
- $p(\theta | \mathbf{X})$ chính là posterior mà ta đang cần tối ưu.
- $p(\theta)$ là prior, đại diện cho những kiến thức đã biết trước về tham số $\theta$ hay những kiến thức đã có trước của con người muốn định hướng cho mô hình.
- $p(\mathbf{X})$ là evidence, đại diện cho xác suất hiển nhiên xảy ra của bộ dữ liệu $\mathbf{X}$ độc lập với mô hình xác suất hay tham số $\theta$. 

$$
\theta = \arg\max_{\theta} p(\theta | \mathbf{X}) =  \arg\max_{\theta} \left[ \frac{\text{likelihood} \times \text{prior}}{\text{evidence}} \right] = \arg\max_{\theta} \left[ \text{likelihood} \times \text{prior} \right]
$$

Do evidence là độc lập với tham số $\theta$, ta có thể loại nó ra khỏi biểu thức tối ưu của $\theta$.
Đến đây, ta thấy điểm khác biệt giữa MAP và MLE nằm ở việc bổ sung thêm prior $p(\theta)$ vào trong biểu thức tối ưu.

**Vậy làm sao để xác định được prior $p(\theta)$?**

**Prior biểu diễn niềm tin trước khi thấy dữ liệu**.
Chọn prior là cách bạn đưa kiến thức (hoặc thiếu kiến thức) về tham số vào mô hình.

Trong thực tế, ta có thể đưa kiến thức về tham số $\theta$ vào prior trong các trường hợp sau:
- **Conjugate priors:** Sử dụng các phân phối xác suất có tính chất liên hợp (conjugate) với likelihood để đơn giản hoá việc tính toán posterior, giúp việc tối ưu trở nên dễ dàng hơn.
Ví dụ: nếu likelihood là phân phối Bernoulli, ta có thể chọn prior là phân phối Beta.
- **Noninformative / weakly informative priors:** Sử dụng các phân phối xác suất không mang nhiều thông tin hoặc mang thông tin yếu để biểu diễn sự thiếu kiến thức về tham số.
Ví dụ: nếu không có kiến thức gì về tham số, ta có thể sử dụng phân phối đều (uniform distribution) hoặc phân phối Gaussian với phương sai lớn để làm prior.
- **Informative priors:** Sử dụng các phân phối xác suất mang nhiều thông tin để biểu diễn kiến thức rõ ràng về tham số.
Ví dụ: nếu ta tin rằng tham số $\theta$ có giá trị trong khoảng xung quanh số 0, ta có thể sử dụng phân phối Gaussian với trung bình là 0 và phương sai nhỏ để làm prior.

MAP kết hợp cả dữ liệu huấn luyện và kiến thức tiên nghiệm (prior) để ước lượng tham số mô hình.
Do đó, vai trò của prior trong MAP là rất quan trọng, ảnh hưởng lớn đến kết quả ước lượng tham số và hiệu năng của mô hình trên dữ liệu chưa từng thấy.

Khi huấn luyện những mô hình machine learning được xây dựng dựa trên MAP, ta cần đánh giá và so sánh kết quả một cách cẩn thận để đảm bảo rằng prior được chọn phù hợp và không làm sai lệch quá mức kết quả học từ dữ liệu.
