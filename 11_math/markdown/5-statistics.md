---
time: 12/26/2022
title: Thống kê dữ liệu
description: Thống kê dữ liệu là nền tảng quan trọng trong machine learning, giúp hiểu và mô tả đặc trưng của dữ liệu trước khi xây dựng mô hình. Thông qua các khái niệm như trung bình, phương sai, độ lệch chuẩn, phân phối xác suất và kiểm định giả thuyết, nhà nghiên cứu có thể đánh giá xu hướng, mức độ biến động và mối quan hệ giữa các biến. Việc phân tích thống kê giúp phát hiện dữ liệu ngoại lai, mất cân bằng hay nhiễu, từ đó hỗ trợ tiền xử lý và lựa chọn mô hình phù hợp. Nhờ đó, mô hình học máy đạt hiệu quả và độ chính xác cao hơn.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/banner.jpeg
tags: [math]
is_highlight: false
is_published: true
---

## 1. Tổng quan về dữ liệu

### 1.1. Kiểu dữ liệu trong thực tế

Dữ liệu trong thực tế được chia thành nhiều kiểu dữ liệu khác nhau.
Các kiểu dữ liệu này được nhóm thành hai nhóm chính: **dữ liệu định tính (qualitative data)** và **dữ liệu định lượng (Quantitative data)** (hoặc **dữ liệu số học (Numerical data)**).

#### Dữ liệu định tính

Là những kiểu dữ liệu không biểu thị bằng giá trị số, mà biểu thị bằng ngôn ngữ và các loại /nhóm (category).
- Ví dụ: màu tóc (đen, vàng, nâu), các loại trái cây (táo, cam, chuối), giới tính (nam, nữ), tên đội bóng (team A, team B, team C) ...
- Ta có thể mã hoá các kiểu dữ liệu này dưới dạng các con số, tuy nhiên, về bản chất, đây vẫn là kiểu dữ liệu định tính.
- Có hai cách để mã hoá loại dữ liệu này:
    - Mã hoá nhãn (Label encoding): Gán mỗi loại một số nguyên duy nhất, ví dụ: {táo: 0, cam: 1, chuối: 2}.
    Phương pháp này thường áp dụng với dữ liệu loại / nhóm và có thể được biểu diễn dưới dạng one-hot encoding, ví dụ: táo: [1,0,0], cam: [0,1,0], chuối: [0,0,1].
    - Mã hoá nhúng (Embedding encoding): Biểu diễn mỗi loại dưới dạng một vector trong không gian nhiều chiều.
    Phương pháp này thường áp dụng với dữ liệu phi cấu trúc như văn bản, hình ảnh và thường sử dụng trong các mô hình học sâu (deep learning) để biểu diễn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/1-linear-algebra/data_space.jpeg" style="width: 600px;"/>

#### Dữ liệu định lượng

Là dữ liệu được biểu thị ngay bằng giá trị số mà không cần mã hoá.
- Ví dụ: chiều cao (1.75m, 1.80m), cân nặng (70kg, 80kg), nhiệt độ (36.5°C, 37.0°C) ...
- Ta có thể thực hiện các phép toán, so sánh hoặc thống kê trực tiếp trên các giá trị này.
- Dữ liệu định lượng được chia làm hai nhóm con:
    - Dữ liệu rời rạc: Dữ liệu được biểu diễn dưới các giá trị là các số tự nhiên hoặc số nguyên.
    - Dữ liệu liên tục: Dữ liệu được biểu diễn dưới các giá trị là các số thực.
- Các dữ liệu định tính, sau khi được mã hoá, có thể được xử lý, phân tích, thống kê như dữ liệu định lượng.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/quantitative_data.jpeg" style="width: 800px;"/>

### 1.2. Hình dạng dữ liệu

Trong thực tế, dữ liệu có thể tồn tại dưới dạng có cấu trúc (structured data), bán cấu trúc (semi-structured data) và phi cấu trúc (unstructured data).

#### Dữ liệu có cấu trúc (Structured data)

Là dữ liệu được tổ chức theo một định dạng cụ thể, thường là bảng (table) trong cơ sở dữ liệu quan hệ (relational database).
Dữ liệu có cấu trúc dễ dàng lưu trữ, truy xuất và phân tích.

Ví dụ: dữ liệu khách hàng trong hệ thống quản lý bán hàng, dữ liệu nhân viên trong hệ thống quản lý nhân sự ...

#### Dữ liệu bán cấu trúc (Semi-structured data)

Là dữ liệu không hoàn toàn tuân theo định dạng bảng, nhưng vẫn có cấu trúc nhất định giúp dễ dàng phân tích.
Dữ liệu bán cấu trúc thường được lưu trữ dưới dạng tài liệu (document) như JSON - JavaScript Object Notation, XML - eXtensible Markup Language, YAML - YAML Ain't Markup Language ...

Ví dụ: dữ liệu các bài báo được lấy từ các trang web, dữ liệu log hệ thống ...

#### Dữ liệu phi cấu trúc (Unstructured data)

Là dữ liệu không có cấu trúc rõ ràng, không tuân theo định dạng bảng hay tài liệu cụ thể.
Dữ liệu phi cấu trúc thường bao gồm văn bản tự nhiên (natural text), hình ảnh, âm thanh, video ...

Ví dụ: dữ liệu email, dữ liệu hình ảnh từ camera giám sát, dữ liệu video từ các nền tảng chia sẻ video ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/structured_semi_structured_unstructured.jpeg" style="width: 800px;"/>

Bất kể dữ liệu có cấu trúc, bán cấu trúc hay phi cấu trúc, việc chuyển đổi chúng thành dạng có thể phân tích và thống kê là bước quan trọng trong quá trình xử lý dữ liệu.

Trong một bộ dữ liệu, ta cần xác định chính xác đối tượng dữ liệu để từ đó lựa chọn phương pháp xử lý, phân tích và thống kê phù hợp.

Ví dụ: Giả sử ta có bộ dữ liệu khách hàng của một cửa hàng bán lẻ.
- Nếu mục tiêu là phân tích hành vi mua sắm của khách hàng, thì đối tượng dữ liệu là từng khách hàng.
- Nếu mục tiêu là phân tích doanh số bán hàng theo từng sản phẩm, thì đối tượng dữ liệu là từng sản phẩm.

### 1.3. Mẫu thống kê và thống kê mô tả

Phần dữ liệu được thu thập và thống kê được gọi là **tập mẫu (sample set)** và nó có nguồn gốc từ một tập dữ liệu lớn hơn goi là **tập thế giới (population)**.
Tập sample sẽ mang thông tin nào đó về tập population, và các tập sample khác nhau có thể sẽ phản ánh những thông tin khác nhau của tập population.

Thống kê mô tả (descriptive statistics) là quá trình tóm tắt và mô tả các đặc trưng chính của một tập dữ liệu, thường là tập mẫu.
Mục tiêu của thống kê mô tả là cung cấp cái nhìn tổng quan về dữ liệu thông qua các giá trị tóm tắt và biểu đồ trực quan.

Để phân tích dữ liệu được chính xác nhất, ta phải làm việc với tập population, nhưng trong thực tế, tập population thường quá lớn và đòi hỏi chi phí cao để thu thập và xử lý toàn bộ dữ liệu.
Do đó, ta chỉ có thể làm việc với tập sample và kỳ vọng rằng tập sample có thể phản ảnh được hầu như toàn bộ bản chất của tập population.
Điều này dẫn đến bài toán về việc chọn mẫu thống kê sao cho đại diện nhất cho tập population.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/population_sample.jpeg" style="width: 700px;"/>

#### Phương pháp chọn mẫu ngẫu nhiên

Vì mỗi phần tử của tập population đã có xác suất được chọn được xác định từ trước, trước cả khi chọn mẫu, và nó cũng chính là một khía cạnh của bản chất của tập population, nên chọn mẫu ngẫu nhiên cho phép đánh giá khách quan hơn các đặc trưng của tập population.

Một số cách để chọn mẫu ngẫu nhiên như:
- **Chọn mẫu ngẫu nhiên đơn giản:**
    - Là phương pháp lựa chọn một cách hoàn toàn ngẫu nhiên, mọi phần tử của tập population có đồng khả năng lọt vào mẫu.
    - **Điểm mạnh:** Do tính ngẫu nhiên nên mẫu có tính đại diện cao và tin cậy.
    - **Điểm yếu:** Phương pháp đòi hỏi phải biết toàn bộ tập population và vì thế chi phí chọn mẫu khá lớn.
    Ngoài ra, nếu một số loại dữ liệu trong tập population có tỷ lệ xuất hiện thấp, thì mẫu ngẫu nhiên đơn giản có thể không bao gồm các loại dữ liệu này, dẫn đến mẫu không đại diện đầy đủ cho tập population.
- **Chọn mẫu phân nhóm:**
    - Đầu tiên ta chia tập population thành các nhóm tương đốì thuần nhất.
    - Từ mỗi nhóm trích ra một mẫu ngẫu nhiên và tập hợp tất cả các mẫu đó cho ta một mẫu ngẫu nhiên phân nhóm.
    - Để phương pháp này hiệu quả, ta phải có hiểu biết nhất định về cấu trúc tập population để phân chia nhóm hợp lý.
    Sau này, mỗi nhóm sẽ có vai trò khác nhau phụ thuộc vào độ quan trọng của chúng trong tập population.
    - **Điểm mạnh:** Giúp mẫu đại diện tốt hơn cho tập population, đặc biệt khi có các nhóm nhỏ với tỷ lệ xuất hiện thấp.
    - **Điểm yếu:** Đòi hỏi hiểu biết trước về cấu trúc của tập population và tập sample có thể bị thiên vị theo cách phân nhóm.

Ví dụ: Xét một tập dữ liệu population gồm toàn bộ sinh viên của một trường đại học, ta muốn chọn một mẫu ngẫu nhiên để khảo sát về thói quen học tập.
- Với phương pháp chọn mẫu ngẫu nhiên đơn giản, ta có thể sử dụng công cụ máy tính để chọn ngẫu nhiên một số sinh viên từ danh sách toàn trường.
- Với phương pháp chọn mẫu phân nhóm, ta có thể chia sinh viên thành các nhóm theo khoa (ví dụ: Khoa Toán, Khoa Văn, Khoa Khoa học Máy tính) và sau đó chọn ngẫu nhiên một số sinh viên từ mỗi khoa để đảm bảo mẫu đại diện cho toàn bộ trường.

#### Phương pháp chọn mẫu có suy luận

Phương pháp chọn mẫu này dựa trên ý kiến các chuyên gia về đối tượng nghiên cứu và điều này kéo theo hạn chế về tính chủ quan của mẫu và chất lượng của mẫu phụ thuộc nhiều vào trình độ và kinh nghiệm của chuyên gia.

Phương pháp này thường được sử dụng trong một số trường hợp rất cụ thể, còn trong hầu hết các trường hợp khác, ta nên sử dụng phương pháp chọn mẫu ngẫu nhiên.

## 2. Các đặc trưng của mẫu thống kê

Với một mẫu dữ liệu gồm có $k$ giá trị khác nhau $x_1, x_2, ..., x_k$, mỗi giá trị có tương ứng $n_1, n_2, ..., n_k$ phần tử và tổng số phần từ $n = n_1 + n_2 + ... + n_k$

### 2.1. Trung bình / Kỳ vọng (Mean / Expectation)

Trung bình của mẫu hay được gọi cách khác là kỳ vọng mẫu được ký hiệu là $\bar{X}$ và được tính bằng công thức sau:

$$
\bar{X} = \frac{1}{n} \sum_{i=1}^{k} x_i n_i
$$

Giá trị trung bình là một giá trị đặc trưng quan trọng của một mẫu dữ liệu.
Giá trị trung bình thường được dùng làm đại diện cho tất cả các phần từ trong mẫu dữ liệu.

Tuy nhiên, giá trị trung bình trong một số trường hợp không thể đại diện được cho một mẫu dữ liệu, ví dụ như:
- Mẫu dữ liệu có một số ít các phần tử ngoại lai (outlier) có giá trị lớn.
- Mẫu dữ liệu có phân bố không đều.
- ...

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf) minh họa các giá trị trung bình, mốt và trung vị trong một mẫu dữ liệu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/mean_modes_median.jpeg" style="width: 600px;"/>

Mốt (mode) là giá trị xuất hiện nhiều nhất trong mẫu dữ liệu và được ký hiệu là $M_o$.

### 2.2. Trung vị và Tứ phân vị (Median and Quantiles)

Trung vị của mẫu được ký hiệu là $M_e$ và được tính bằng cách sắp xếp mẫu thành dãy theo thứ tự tăng dần hoặc giảm dần:
- Đối với mẫu có số lượng phần tử là số lẻ, trung vị của mẫu là số nằm ở đúng vị trí chính giữa của dãy.
- Đối với mẫu có số lượng phần tử là số chẵn, trung vị của mẫu đuơc tính bằng trung bình cộng của hai số ở giữa của dãy.

Trung vị trong một số trường hợp cũng được sử dụng để làm đại diện cho tất cả các phần từ trong mẫu dữ liệu.
Đặc biệt, trong trường hợp giá trị trung bình không thể đại diện tốt được cho một mẫu dữ liệu như mẫu dữ liệu có một số ít các phần tử ngoại lai có giá trị lớn.

Tuy nhiên, trong trường hợp mẫu dữ liệu có nhiều các phần tử ngoại lại có giá trị nhỏ, trung vị không thể đại diện được tốt cho mẫu dữ liệu.

Đi kèm với trung vị, ta có thêm giá trị đặc trưng tứ phân vị (quantiles): tứ phân vị thứ nhất (giá trị ở vị trí 25%), tứ phân vị thứ hai (giá trị ở vị trí 50% hay là trung vị), tứ phân vị thứ ba (giá trị ở vị trí 75%).
Các giá trị này được sử dụng để xây dựng biểu đồ Box plot và Violin plot, được sử dụng nhiều trong quá trình trực quan hoá dữ liệu.

### 2.3. Phương sai - Độ lệch chuẩn (Variance - Standard deviation)

Phương sai của mẫu được ký hiệu là $S^2$ và được tính bằng công thức sau:

$$
S^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{X})^2 
$$

Và ta cũng có độ lệch chuẩn (standard deviation - std) được tính bằng căn bậc hai của phương sai của mẫu và được ký hiệu là $S$.

Phương sai hoặc độ lệch chuẩn là giá trị thể hiện độ phân tán của dữ liệu.
Đối với những bộ dữ liệu có phương sai nhỏ (bộ dữ liệu phân tán ít), ta có thể chỉ cần sử dụng một mẫu dữ liệu nhỏ để phân tích và đánh giá.
Ngược lại, với những bộ dữ liệu có phương sai lớn (bộ dữ liệu phân tán nhiều), ta có thể cần phải sử dụng một mẫu dữ liệu lớn.

### 2.4. Giá trị lớn nhất, Giá trị nhỏ nhất (Max, Min)

Giá trị lớn nhất và giá trị nhỏ nhất là hai giá trị đặc trưng cơ bản khi thống kê dữ liệu.
Hai giá trị này cung cấp thông tin về khoảng phân bố dữ liệu từ đó:
- Giúp ta có được cái nhìn sơ bộ về mẫu dữ liệu và đánh giá tổng quan về chất lượng của mẫu dữ liệu.
- Giúp lựa chọn tối ưu kiểu dữ liệu để lưu trữ mẫu dữ liệu.
- Kết hợp cùng các giá trị như trung bình, trung vị giúp đánh giá sâu hơn về phân bố của dữ liệu.
- ...

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/box_plot.jpeg" style="width: 600px;"/>

## 3. Các phân phối dữ liệu theo lý thuyết

### 3.1. Định lý giới hạn trung tâm (Central Limit Theorem)

Định lý giới hạn trung tâm (Central Limit Theorem - CLT) là một trong những định lý quan trọng nhất trong thống kê và xác suất.

**Phát biểu định lý:** Nếu bạn lấy nhiều mẫu ngẫu nhiên có cùng kích thước từ một tổng thể (population) bất kỳ có trung bình $\mu$ và phương sai $\sigma^2$ hữu hạn, thì phân phối của trung bình mẫu (sample mean) sẽ tiến gần đến phân phối chuẩn (normal distribution) khi kích thước mẫu đủ lớn, bất kể phân phối gốc của tổng thể là gì.

**Ý nghĩa thực tiễn:** Giúp dùng phân phối chuẩn để ước lượng hoặc kiểm định giả thuyết cho dữ liệu không chuẩn (miễn là kích thước mẫu đủ lớn, thường là $n \geq 30$). 
Điều này rất hữu ích trong thống kê và học máy, vì nhiều phương pháp dựa trên giả định phân phối chuẩn.

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf) minh họa về sự tương quan của các đặc trưng của dữ liệu.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/data_correlation.jpeg" style="width: 700px;"/>

### 3.2. Phân phối chuẩn (Normal distribution)

Phân phối chuẩn (hay Normal/Gaussian distribution) là phân phối liên tục cơ bản và rất phổ biến.
Phân phối chuẩn là phân phối mô tả biến ngẫu nhiên liên tục nhận giá trị thực $x \in (-\infty, \infty)$

#### Phân phối chuẩn một biến (Univariate Normal Distribution)

Phân phối chuẩn một biến được mô tả bởi hai tham số:
- Kỳ vọng (mean), ký hiệu là $\mu$
- Phương sai (variance), ký hiệu là $\sigma^2$
- Hoặc Độ lệch chuẩn (standard deviation) $\sigma$

Phân phối chuẩn được ký hiệu là $P(x) = \text{Norm}_x [\mu, \sigma^2]$.
trong đó:
- $\mu$ thể hiện vị trí đỉnh, nơi có xác suất cao nhất
- $\sigma$ thể hiện độ rộng của phân phối.
    - $\sigma$ lớn đồng nghĩa với phân phối có đầu ra biến đổi mạnh
    - $\sigma$ nhỏ đồng nghĩa với phân phối có đầu ra ổn định.

Hàm mật độ xác suất của phân phối chuẩn một biến là:

$$ P(x) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp \left( -\frac{(x - \mu)^2}{2\sigma^2}\right) $$

##### Tính toán tham số của phân phối chuẩn một biến

Ta có một bộ dữ liệu $X = \{x_1, x_2, ..., x_N\}$ với $N$ mẫu.
Giả sử rằng $X$ tuân theo phân phối chuẩn một biến với các tham số $\mu$ và $\sigma^2$.
Ta có thể ước lượng các tham số này như sau:
- Kỳ vọng $\mu$ được ước lượng bằng:
$$ \hat{\mu} = \frac{1}{N} \sum_{i=1}^N x_i $$
- Phương sai $\sigma^2$ được ước lượng bằng:
$$ \hat{\sigma}^2 = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \hat{\mu})^2 $$
- Độ lệch chuẩn $\sigma$ được ước lượng bằng:
$$ \hat{\sigma} = \sqrt{\hat{\sigma}^2} $$

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf) minh họa phân phối chuẩn một biến và phân phối chuẩn nhiều biến.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/uni_vs_multivariate.jpeg" style="width: 700px;"/>

#### Phân phối chuẩn nhiều biến (Multivariate normal distribution)

Phân phối chuẩn nhiều biến là dạng tổng quát của phân phối chuẩn một biến, được sử dụng để mô tả biến ngẫu nhiên liên tục nhiều chiều.

Ta xét biến ngẫu nhiên $D$ chiều, phân phối chuẩn nhiều biến được mô tả bởi hai tham số:
- Vector kỳ vọng (mean vector) $\mu \in R^D$.
- Ma trận hiệp phương sai (covariance matrix) $\Sigma \in \mathbb{S}_{++}^D$ là một ma trận đối xứng xác định dương.

Ma trận hiệp phương sai là một ma trận vuông, trong đó:
- Các phần tử nằm trên đường chéo chính lần lượt là phương sai của từng biến.
- Các phần từ còn lại (không nằm trên đường chéo) là các hiệp phương sai của đôi một hai biến ngẫu nhiên khác nhau trong tập hợp.

Hiệp phương sai là độ đo sự biến thiên cùng nhau của hai biến ngẫu nhiên (phân biệt với phương sai - đo mức độ biến thiên của một biến).
Giá trị hiệp phương sai nằm trong khoảng từ $(-\infty, \infty)$ trong đó:
- Giá trị dương biểu thị rằng cả hai biến chuyển động theo cùng một hướng
- Giá trị âm biểu thị rằng cả hai biến chuyển động ngược chiều nhau
- Giá trị bằng không biểu thị hai biến không có tương quan với nhau.

##### Vì sao ma trận hiệp phương sai lại là ma trận đối xứng xác định dương?
Ma trận hiệp phương sai **luôn đối xứng qua đường chéo chính** và các phần từ trên đường chéo chính luôn dương nên **các trị riêng chính của chúng luôn dương** và ma trận hiệp phương sai là xác định dương.

Hàm mật độ xác suất của phân phối chuẩn nhiều biến là:

$$ P(x) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp \left(\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right) $$

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf) minh họa phân phối chuẩn nhiều biến.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/multivariate_normal_distribution.jpeg" style="width: 600px;"/>

##### Tính toán tham số của phân phối chuẩn nhiều biến

Ta có một bộ dữ liệu $X = \{x_1, x_2, ..., x_N\}$ với $N$ mẫu.
Trong đó, mỗi mẫu $x_i$ là một vector chiều $x_i \in R^D$.
Giả sử rằng $X$ tuân theo phân phối chuẩn nhiều biến với các tham số $\mu$ và $\Sigma$.
Ta có thể ước lượng các tham số này như sau:
- Vector kỳ vọng $\mu$ được ước lượng bằng:
$$ \hat{\mu} = \frac{1}{N} \sum_{i=1}^N x_i $$
- Ma trận hiệp phương sai $\Sigma$ được ước lượng bằng:
$$ \hat{\Sigma} = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \hat{\mu})(x_i - \hat{\mu})^T $$
- Độ lệch chuẩn $\sigma$ được ước lượng bằng:
$$ \hat{\sigma} = \sqrt{\hat{\Sigma}} $$

### 3.3. Phân phối đều (Uniform distribution)

Trái ngược với phân phối chuẩn, phân phối biểu diễn biến ngẫu nhiên có những giá trị có xác suất xuất hiện cao hơn các giá trị khác, phân phối đều được sử dụng để mô tả biến ngẫu nhiên liên tục có xác suất nhận các giá trị trong một khoảng xác định là như nhau.

Phân phối đều được mô tả bởi hai tham số:
- Tham số $a$ là giá trị nhỏ nhất trong khoảng
- Tham số $b$ là giá trị lớn nhất trong khoảng

Phân phối đều được ký hiệu là $P(x) = \text{Unif}_x [a, b]$.

Hàm mật độ xác suất của phân phối đều là:
$$ P(x) = \begin{cases} \frac{1}{b - a} & \text{if } a \leq x \leq b \\ 0 & \text{otherwise} \end{cases} $$

Phân phối đều thường được sử dụng khi không có thông tin thiên lệch nào, tức tất cả giá trị trong khoảng $[a,b]$ đều khả dụng như nhau.
Nói cách khác, khi ta cần tạo sự ngẫu nhiên mà không ưu tiên giá trị nào, phân phối này là lựa chọn hợp lý.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/uniform.jpeg" style="width: 600px;"/>

Ví dụ: Ta có thể dùng phân phối đều để khởi tạo trọng số ban đầu (random initialization) trong mạng nơ-ron

Ví dụ: Ta có thể dùng phân phối đều để sinh mẫu dữ liệu ngẫu nhiên đều trong khoảng nhất định. 

### 3.4. Phân phối Bernoulli (Bernoulli distribution)

Phân phối Bernoulli là phân phối xác suất rời rạc cơ bản dành cho biến ngẫu nhiên nhị phân: nó mô tả trường hợp khi đầu ra chỉ nhận một trong hai giá trị $x ∈ {0, 1}$.

Phân phối Bernoulli được mô tả bằng một tham số $p \in [0, 1]$.
Nếu biến ngẫu nhiên $X$ tuân theo phân phối Bernoulli với tham số $p$ được ký hiệu là 
$X\sim \mathrm{Bernoulli}(p)$

có nghĩa là:
- $P(X = 1) = p$
- $P(X = 0) = 1 - P(X = 1) = 1 - p$

hoặc

$$ P(X = k) = \begin{cases} 1 - p & \text{if } k = 0 \\ p & \text{if } k = 1 \end{cases} $$

Hai đẳng thức này thường được viết gọn lại trở thành hàm khối xác suất:

$$ P(X = k) = p^k (1 - p)^{1 - k} $$
với $k \in \{0, 1\}$.

Ví dụ: Tung một đồng xu có xác suất ra mặt ngửa là $p$ là một phép thử Bernoulli.

Phân phối Bernoulli được ứng dụng nhiều trong machine learning khi mô hình hóa nhãn nhị phân (nói cách khác, được sử dụng trong bài toán phân lớp nhị phân).

Hình ảnh dưới đây được lấy từ cuốn sách [MATHEMATICS FOR MACHINE LEARNING](https://github.com/MinhHuuNguyen/ai-lectures/blob/master/books/mathematics_for_machine_learning_deisenroth.pdf).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/bernoulli.jpeg" style="width: 300px;"/>

Ví dụ: Trong bài toán binary classification của mô hình logistic regression, xác suất dự đoán nhãn positive được hiểu là $p$, và kết quả nhãn có thể xem như một biến Bernoulli với xác suất thành công đó.

## 4. Một số hình dạng của phân phối khi thống kê dữ liệu

### 4.1. Phân phối long tail - phân phối lệch (Long tail distribution - Skewed distribution)

Khi thống kê dữ liệu, ta có thể bắt gặp những trường hợp bộ dữ liệu có những giá trị lớn hoặc giá trị nhỏ xuất hiện rất ít lần.
Với những bộ dữ liệu như vậy, việc vẽ phân phối tần suất xuất hiện sẽ xuất hiện trạng thái được gọi là long tail.
Phân phối khi đó được gọi là phân phối long tail (long tail distribution).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/long_tail.jpeg" style="width: 600px;"/>

Nếu như đối với phân phối long tail, ta tập trung góc nhìn vào phần "đuôi" của phân phối, thì phân phối lệch là một góc nhìn khác khi ta tập trung vào phần đỉnh của phân phối.

Độ lệch của phân phối được sử dụng để so sánh với phân phối chuẩn và được tính bằng công thức:

$$
\text{skewness} = \frac{1}{n} \frac{\sum_{i=1}^{n} (X_i - \bar{X})^3}{s^3}
$$

Có ba loại độ lệch:
- skewness = 0, phân phối đối xứng và được xem như là phân phối chuẩn
- skewness < 0, phân phối lệch trái có độ lệch âm (negative skewness), khi đó, giá trị kỳ vọng < giá trị trung vị < giá trị mốt
- skewness > 0, phân phối lệch phải có độ lệch dương (positive skewness), khi đó, giá trị kỳ vọng > giá trị trung vị > giá trị mốt

Độ lệch được coi là đáng kể nếu độ lớn của giá trị tuyệt đối của nó lớn hơn 0.5.

### 4.2. Phân phối nhọn (Kurtosis distribution)

Độ nhọn của phân phối được sử dụng để so sánh với phân phối chuẩn và được tính bằng công thức:

$$
\text{kurtosis} = \frac{1}{n} \frac{\sum_{i=1}^{n} (X_i - \bar{X})^3}{s^3}
$$

Đối với phân phối chuẩn có độ nhọn = 3, ta tính toán chỉ số

$$
\text{excess kurtosis} = \text{kurtosis} - 3
$$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/kurtosis.jpeg" style="width: 600px;"/>

Có ba loại độ nhọn:
- excess kurtosis = 0 được gọi là mesokurtic và được xem như là phân phối chuẩn
- excess kurtosis > 0 được gọi là leptokurtic, khi đó phân phối có dạng nhọn
- excess kurtosis < 0 được gọi là platykurtic, khi đó phân phối có dạng rộng

Độ nhọn được coi là đáng kể nếu độ lớn của giá trị tuyệt đối của nó lớn hơn 1.

### 4.3. Phân phối đa thức (Multimodal distribution)

Phân phối đa thức (multimodal distribution) là phân phối dữ liệu có nhiều hơn một đỉnh (một mốt).
Phân phối có một đỉnh (một mốt) được gọi là unimodal.
Phân phối có hai đỉnh (hai mốt) được gọi là bimodal
Phân phối có nhiều đỉnh (nhiều mốt) được gọi là multimodal.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/multimodal.jpeg" style="width: 600px;"/>

## 5. Kullback-Leibler divergence (KL divergence)

KL divergence (viết đầy đủ là Kullback–Leibler divergence) là một khái niệm trong lý thuyết thông tin và xác suất, dùng để đo mức độ khác biệt giữa hai phân phối xác suất. 

KL divergence không phải là một khoảng cách thực sự (như khoảng cách Euclidean), vì nó không thỏa mãn tính đối xứng mà là một thước đo độ mất mát thông tin khi ta dùng phân phối $Q$ để xấp xỉ phân phối $P$.

$$ D_{KL}(P || Q) \neq D_{KL}(Q || P) $$

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/11_math/images/5-statistics/kl_divergence.jpeg" style="width: 800px;"/>

Giả sử $P(x)$ và $Q(x)$ là hai phân phối xác suất trên cùng một tập biến ngẫu nhiên rời rạc $x$, KL divergence từ $P$ đến $Q$ được định nghĩa là:

$$ D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} $$

hoặc với biến ngẫu nhiên liên tục:

$$ D_{KL}(P || Q) = \int_{-\infty}^{\infty} P(x) \log \frac{P(x)}{Q(x)} dx $$

trong đó:
- $P(x)$ là phân phối thực tế
- $Q(x)$ là phân phối xấp xỉ
- $D_{KL}(P || Q)$ là KL divergence từ $P$ đến $Q$
- $D_{KL}(P || Q) \geq 0$ với $D_{KL}(P || Q) = 0$ khi và chỉ khi $P = Q$.

KL divergence được sử dụng rộng rãi trong nhiều lĩnh vực, bao gồm học máy, thống kê và lý thuyết thông tin.
Trong học máy, KL divergence thường được sử dụng trong các thuật toán tối ưu hóa, như huấn luyện mô hình học sâu (deep learning) và học tăng cường (reinforcement learning), để đo sự khác biệt giữa phân phối dự đoán của mô hình và phân phối thực tế.
