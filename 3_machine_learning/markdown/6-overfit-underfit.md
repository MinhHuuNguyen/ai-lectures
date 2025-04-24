---
time: 08/02/2022
title: Hiện tượng Overfit và Underfit
description: Trong quá trình huấn luyện mô hình machine learning, ta thường gặp phải hiện tượng overfit và underfit. Hai hiện tượng này khiến cho việc huấn luyện mô hình gặp nhiều khó khăn và gây ra sự sai sót trong quá trình đánh giá mô hình.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/6-overfit-underfit/banner.png
tags: [machine-learning]
is_highlight: false
is_published: true
---

## 1. Tương quan giữa dữ liệu và mô hình trong quá trình huấn luyện

Khi xây dựng và huấn luyện một mô hình machine learning, ta luôn cần ít nhất 2 bộ dữ liệu, bộ dữ liệu huấn luyện (training data) và bộ dữ liệu kiểm thử (test data).
Trong đó, bộ dữ liệu train đại diện cho những dữ liệu mà mô hình được phép thấy và học, bộ dữ liệu test đại diện cho những dữ liệu dùng để đánh giá cuối cùng về mô hình.

Việc lựa chọn mô hình machine learning phù hợp để học bộ dữ liệu train và cho kết quả dự đoán tốt trên bộ dữ liệu test là điều quan trọng nhất trong quá trình huấn luyện nhưng ko dễ để thực hiện.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/6-overfit-underfit/model_vs_data_complexity.png" style="width: 500px;"/>

Trong quá trình huấn luyện mô hình machine learning, ta cần xem xét đến độ phức tạp của mô hình tương quan với độ phức tạp của bộ dữ liệu train.
- Độ phức tạp của mô hình thể hiện khả năng mà mô hình có thể học được những quy luật, xu hướng trong bộ dữ liệu train.
Độ phức tạp của mô hình được thể hiện qua số lượng tham số của mô hình, số lượng lớp trong mô hình, số lượng nơ-ron trong mỗi lớp, ...
- Độ phức tạp của bộ dữ liệu train thể hiện mức độ khó khăn mà mô hình gặp phải trong việc học được những quy luật, xu hướng trong bộ dữ liệu.
Độ phức tạp của bộ dữ liệu được thể hiện qua số lượng điểm dữ liệu trong bộ train, số lượng đặc trưng trong mỗi điểm dữ liệu, độ nhiễu (noise) trong bộ dữ liệu, ...
- Độ phức tạp của mô hình và bộ dữ liệu train có mối quan hệ tỷ lệ thuận với nhau, tức là nếu độ phức tạp của bộ dữ liệu train tăng lên thì độ phức tạp của mô hình cũng cần tăng lên để có thể học được những quy luật, xu hướng trong bộ dữ liệu.

Với những bộ dữ liệu đơn giản, ta có thể sử dụng một mô hình machine learning đơn giản để xử lý.
Với những bộ dữ liệu có mức độ phức tạp tăng dần, ta cần những mô hình machine learning phức tạp hơn tương ứng.

Tuy nhiên, câu hỏi đặt ra là như thế nào là một mô hình machine learning đủ phức tạp tương ứng với một bộ dữ liệu nào đó?
Nếu một mô hình không đủ phức tạp so với bộ dữ liệu, hoặc quá phức tạp so với bộ dữ liệu thì hiện tượng gì sẽ xảy ra?

## 2. Hiện tượng Underfitting

### 2.1. Hiện tượng underfitting là gì?

Underfitting là hiện tượng xảy ra khi mô hình không đủ sức để học ra được những quy luật, xu hướng trong bộ train và dẫn đến chất luợng của các dự đoán trên cả bộ train và bộ test đều thấp.

Ví dụ: Bài toán Định giá nhà
- Với bộ dữ liệu đơn giản, input là diện tích ngôi nhà và output là giá của ngôi nhà đó, ta chỉ cần một mô hình machine learning đơn giản như Linear Regression để xử lý.
- Với bộ dữ liệu phức tạp hơn, input là diện tích ngôi nhà, số phòng ngủ, số phòng tắm, vị trí ngôi nhà, ... và output là giá của ngôi nhà đó, Linear Regression vẫn có thể xử lý được nhưng không hiệu quả, độ chính xác không cao.

Ta gọi mô hình đang bị underfit.
Lúc này, ta cần dùng đến những mô hình có độ phức tạp cao hơn, khả năng học tốt hơn từ bộ dữ liệu.

### 2.2. Biểu hiện

Biểu hiện của hiện tượng underfit là giá trị loss của mô hình trên cả bộ train và bộ test đều thấp.

Ta chỉ cần quan sát thấy giá trị loss trên bộ train thấp là đã có thể nhận biết được mô hình đang bị underfit, giá trị loss trên bộ test thấp là điều hiển nhiên kéo theo.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/6-overfit-underfit/underfit_loss.png" style="width: 500px;"/>

Ngoài chỉ số loss, ta cũng có thể sử dụng các chỉ số khác như accuracy, precision, recall, ... đối với bài toán classification hay R2 score ... đối với bài toán regression để đánh giá hiện tượng underfit.

### 2.3. Giải pháp

Có hai nguyên nhân chính dẫn đến hiện tượng underfit:
- **Nguyên nhân 1:** Bộ dữ liệu train chứa quá nhiều nhiễu (noise) và các điểm dữ liệu ngoại lai (outlier), nói cách khác, trong bộ dữ liệu có tỷ lệ lớn các phần tử không tuân theo logic chung, điều này khiến cho mô hình gặp khó khăn để khái quát hoá được bộ dữ liệu.
- **Nguyên nhân 2:** Mô hình có độ phức tạp quá thấp (mô hình quá đơn giản) so với bộ dữ liệu.

Tương ứng, ta có hai giải pháp để giải quyết hiện tượng underfit:
- Đối với **Nguyên nhân 1**, ta cần phải làm sạch bộ dữ liệu train bằng cách loại bỏ các điểm dữ liệu không tuân theo logic chung.
- Đối với **Nguyên nhân 2**,, ta cần phải tăng độ phức tạp của mô hình bằng cách sử dụng một mô hình machine learning phức tạp hơn giúp mô hình học được bộ dữ liệu train dễ dàng hơn.

Trong thực tế, ta thường lựa chọn hoặc xây dựng những mô hình machine learning có độ phức tạp cao, dễ dàng học được bộ dữ liệu train (tức là dễ dàng vượt qua hiện tượng underfit).
Lúc này các mô hình đó sẽ có xu hướng gặp phải hiện tượng overfit.
Và ta sẽ sử dụng các kỹ thuật để giảm thiểu hiện tượng overfit.

## 3. Hiện tượng Overfitting

### 3.1. Hiện tương overfitting là gì?

Overfitting là hiện tượng xảy ra khi mô hình học thuộc phần lớn (hoặc thậm chí toàn bộ) bộ dữ liệu train.
Lúc này, mô hình không còn khả năng khái quát hoá bộ dữ liệu nữa mà chỉ ghi nhớ các điểm dữ liệu trong bộ train.

Nói cách khác, nếu đưa cho mô hình những điểm dữ liệu nằm trong bộ train, mô hình sẽ cho ra những dự đoán cực kỳ chính xác.
Nhưng chỉ cần khác một chút, đưa cho mô hình những điểm dữ liệu nằm ngoài bộ train, mô hình sẽ cho ra những dự đoán sai lệch rất lớn.

### 3.2. Biểu hiện

Biểu hiện của hiện tượng overfit là giá trị loss của mô hình trên cả bộ train và bộ test đều giảm dần trong giai đoạn đầu.
Tuy nhiên, sau một thời gian, giá trị loss trên bộ train tiếp tục giảm nhưng giá trị loss trên bộ test lại tăng lên.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/3_machine_learning/images/6-overfit-underfit/overfit_loss.png" style="width: 500px;"/>

Ngoài chỉ số loss, ta cũng có thể sử dụng các chỉ số khác như accuracy, precision, recall, ... đối với bài toán classification hay R2 score ... đối với bài toán regression trên cả bộ dữ liệu train và bộ dữ liệu test để đánh giá hiện tượng overfit.

### 3.3. Giải pháp

Nguyên nhân dẫn đến hiện tượng overfit là do mô hình có độ phức tạp quá cao (mô hình quá phức tạp) so với bộ dữ liệu train quá đơn giản.
Với bộ dữ liệu train đơn giản, có hai trường hợp xảy ra hiện tượng overfit:
- Trường hợp 1: Bộ dữ liệu train không đủ khái quát để mô tả được bộ dữ liệu test.
- Trường hợp 2: Bộ dữ liệu train đủ khái quát để mô tả được bộ dữ liệu test, nhưng mô hình lại quá phức tạp và thời gian huấn luyện quá lâu khiến cho mô hình ghi nhớ được các điểm dữ liệu trong bộ train.

Tương ứng, ta có hai giải pháp để giải quyết hiện tượng overfit:
- Đối với **Trường hợp 1**, ta cần phải tăng độ phức tạp của bộ dữ liệu train bằng cách:
    - **Thu thập thêm dữ liệu** cho bộ train thông qua quá trình thu thập và gán nhãn dữ liệu.
    - Sử dụng các kỹ thuật **tăng cường dữ liệu (data augmentation)** để tạo ra thêm dữ liệu cho bộ train.
    - Sử dụng bổ sung dữ liệu không có nhãn trong quá trình huấn luyện mô hình với các thuật toán semi-supervised learning và self-supervised learning.
- Đối với **Trường hợp 2**, ta cần phải giảm độ phức tạp của mô hình bằng cách:
    - Sử dụng một mô hình machine learning đơn giản hơn giúp mô hình học được bộ dữ liệu train dễ dàng hơn.
    - Đối với các mô hình dựa trên phép biến đổi tuyến tính (như Neural Network ...), ta có thể sử dụng các kỹ thuật **regularization** hay **dropout** để giảm độ phức tạp của mô hình.

## 4. Kỹ thuật Regularization

### 4.1. Regularization là gì?

Một cách đơn giản và hiệu quả để giảm độ phức tạp của mô hình là sử dụng các kỹ thuật regularization.
Regularization là một kỹ thuật được sử dụng để giảm thiểu hiện tượng overfitting bằng cách thêm một số ràng buộc liên quan đến độ phức tạp của mô hình vào hàm mất mát (loss function) của mô hình.

$$ L = L_{loss} + \lambda L_{reg} $$

trong đó:
- $L_{loss}$ là hàm mất mát (loss function) của mô hình.
- $L_{reg}$ là hàm ràng buộc (regularization function) của mô hình.
- $\lambda$ là tham số điều chỉnh độ mạnh của regularization.
- Nếu $\lambda$ = 0, mô hình sẽ không bị regularization.

Kỹ thuật regularization còn được gọi là weight decay hay weight regularization.

### 4.2. Các loại regularization

Hàm $L_{reg}$ là một hàm số dương, thể hiện độ phức tạp của mô hình.
Hàm $L_{reg}$ càng lớn thì độ phức tạp của mô hình càng cao.

Có nhiều loại hàm $L_{reg}$ khác nhau, nhưng phổ biến nhất là hai loại sau:
- **L1 regularization** (Lasso Regression): Hàm $L_{1}$ là tổng các giá trị tuyệt đối của các tham số trong mô hình.
$$ L_{1} = \sum_{i=1}^{n} |w_i| $$
- **L2 regularization** (Ridge Regression): Hàm $L_{2}$ là tổng các giá trị bình phương của các tham số trong mô hình.
$$ L_{2} = \sum_{i=1}^{n} w_i^2 $$
- **Elastic Net**: Hàm $L_{\text{elastic_net}}$ là tổng các giá trị tuyệt đối và bình phương của các tham số trong mô hình.
$$ L_{\text{elastic_net}} = \sum_{i=1}^{n} |w_i| + \sum_{i=1}^{n} w_i^2 $$

trong đó $w_i$ là các tham số trong mô hình.

Ngoài ra, ta có thể có công thức khái quát của L1 và L2 regularization như sau:
$$ L_{p} = \sum_{i=1}^{n} |w_i|^p $$
trong đó $p$ là một số dương.

Công thức trên được gọi là Norm p của vector $w$.
- Nếu $p$ = 1, ta có Norm 1 (L1 regularization).
- Nếu $p$ = 2, ta có Norm 2 (L2 regularization).
- ...

### 4.3. Tối ưu hóa hàm mất mát

Trong quá trình huấn luyện mô hình, ta sẽ tối ưu hóa hàm mất mát $L$ thay vì hàm mất mát $L_{loss}$.
Khi tối ưu hóa hàm mất mát $L$, mô hình sẽ cố gắng giảm thiểu cả hàm mất mát $L_{loss}$ và hàm ràng buộc $L_{reg}$.

Khi cực tiểu hoá hàm ràng buộc $L_{reg}$, sẽ có một lượng các tham số trong mô hình bị đẩy về 0, từ đó, giảm độ phức tạp của mô hình.
Tuy nhiên, khi giá trị hàm $L_{reg}$ quá nhỏ dẫn đến mô hình quá đơn giản, mô hình sẽ có thể không học được bộ dữ liệu train, kéo theo giá trị hàm $L_{loss}$ sẽ lớn.

Vì vậy, việc cực tiểu hoá hàm mất mát $L$ bản chất là tìm kiếm một sự cân bằng giữa việc cực tiểu hoá hàm mất mát $L_{loss}$ và hàm ràng buộc $L_{reg}$.

Việc thêm hàm ràng buộc $L_{reg}$ vào hàm mất mát $L$ sẽ giúp mô hình học ra được điểm cân bằng giữa độ phức tạp của mô hình và khả năng học trên bộ dữ liệu train.

## 5. Vai trò của bộ dữ liệu validation

Việc sử dụng các kỹ thuật để giảm thiểu hiện tượng overfit là cần thiết, tuy nhiên, điều này gây ra một hệ quả là mô hình có thể gặp phải hiện tượng overfit trên bộ test.

Với hai bộ dữ liệu train và test, ta cho phép mô hình học trên bộ train và kiểm tra trên bộ test.
Ta quan sát kết quả trên bộ test, và có thể rút ra những cải thiện cho mô hình để mô hình có thể cho ra những dự đoán tốt hơn trên bộ test.
Quá trình này lặp đi lặp lại cho đến khi mô hình cho ra những dự đoán tốt trên bộ test và vô hình chung, mô hình đã học thuộc bộ test thông qua bộ train.

Nói cách khác, ta đã sử dụng bộ test như một bộ dữ liệu train thứ hai.
Điều này dẫn đến việc mô hình có thể cho ra những dự đoán tốt trên bộ test nhưng lại không thể cho ra những dự đoán tốt trên những dữ liệu mới.

Vì vậy, trong thực tế, ta sẽ sử dụng bộ test như là bài kiểm tra cuối cùng và ta sẽ không sử dụng bộ test để phân tích và cải thiện mô hình.
Từ đó, ta cần một bộ dữ liệu nữa giúp đánh giá một cách khách quan và chính xác tình trạng overfit của mô hình và là căn cứ để giúp ta phần nào đó dự đoán được kết quả của mô hình trên bộ test cuối cùng.

Bộ dữ liệu này được gọi là bộ dữ liệu validation.
Bộ dữ liệu validation sẽ được sử dụng để đánh giá mô hình trong quá trình huấn luyện và giúp ta điều chỉnh các tham số của mô hình.
Ta sẽ phân tích và cải thiện mô hình dựa trên kết quả trên bộ dữ liệu validation.

Việc xuất hiện thêm bộ dữ liệu validation sẽ giúp ta có được một cái nhìn tổng quan hơn về mô hình.
Mô hình cần phải cho ra những dự đoán tốt trên cả ba bộ dữ liệu train, validation và test.

Tóm lại, với ba bộ dữ liệu train, validation và test:
- Nếu mô hình học kém trên bộ train, mô hình bị underfit.
- Nếu mô hình học tốt trên bộ train nhưng kém trên bộ validation, mô hình bị overfit trên bộ train.
- Nếu mô hình học tốt trên bộ train và bộ validation nhưng kém trên bộ test, mô hình bị overfit trên bộ validation.
- Nếu mô hình học tốt trên cả ba bộ dữ liệu, mô hình có thể cho ra những dự đoán tốt trên những dữ liệu mới.
