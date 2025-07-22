---
time: 02/23/2023
title: Mạng nơ ron hồi quy Recurrent neural network
description: Mạng nơ ron hồi quy (RNN) là mô hình rất phổ biến trong thời gian trước đây với những kết quả đầy hứa hẹn trên các bài toán xử lý ngôn ngữ tự nhiên (NLP). Cho dù ở thời điểm hiện tại, với sự phát triển của cơ chế Attention và các mô hình Transformer đạt kết quả cao trên các bài toán xử lý ngôn ngữ, ý tưởng về cơ chế hoạt động của các mô hình RNN vẫn đáng chú ý và được áp dụng trong một số trường hợp cụ thể như lượng dữ liệu ít hoặc tài nguyên tính toán hạn chế.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/2-recurrent-neural-network/banner.png
tags: [deep-learning, natural-language-processing]
is_highlight: false
is_published: true
---

## 1. Ý tưởng "hồi quy" trong kiến trúc mạng nơ ron

Trong các mạng nơ ron truyền thống hay còn gọi là Feedforward Neural Network, các giá trị đầu vào và đầu ra là độc lập với nhau, tức là chúng không liên kết thành chuỗi với nhau, tuy nhiên, đối với bài toán xử lý dữ liệu theo chuỗi thời gian nói chung như xử lý dữ liệu âm thanh, xử lý dữ liệu video hay xử lý ngôn ngữ tự nhiên, đây là một ý tưởng rất tồi.

Ví dụ trong bài toán xử lý ngôn ngữ tự nhiên, nếu muốn đoán từ tiếp theo có thể xuất hiện trong một câu thì ta cũng cần biết các từ trước đó xuất hiện lần lượt thế nào.
Từ đó, ý tưởng chính của Recurrent Neural Network (RNN) là sử dụng các chuỗi thông tin có thứ tự làm đầu vào cho mạng.
RNN được gọi là hồi quy bởi lẽ chúng thực hiện cùng một tác vụ cho tất cả các phần tử của một chuỗi đầu vào.
RNN tính toán giá trị đầu ra phụ thuộc vào cả các phép tính trước đó.

Hình ảnh này được lấy từ bài báo [Recurrent Neural Networks: A Comprehensive Review of Architectures, Variants, and Applications](https://www.mdpi.com/2078-2489/15/9/517) giúp mô tả chi tiết kiến trúc bên trong của một cell trong mô hình LSTM.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/2-recurrent-neural-network/unfold_rnn.png" style="width: 600px;"/>

trong đó:
- $x_i$ là giá trị tại vị trí (thời điểm) thứ $i$ của chuỗi đầu vào
- $h_i$ là giá trị trạng thái của RNN tại vị trí (thời điểm) thứ $i$
- $y_i$ là giá trị tại vị trí (thời điểm) thứ $i$ của chuỗi đầu ra

Ví dụ: Xét bài toán dịch máy từ tiếng anh sang tiếng việt
- Với câu tiếng anh đầu vào là "I love you", ta sẽ có chuỗi đầu vào là $x_1 = I$, $x_2 = love$, $x_3 = you$.
- Với câu tiếng việt đầu ra là "Tôi yêu bạn", ta sẽ có chuỗi đầu ra là $y_1 = Tôi$, $y_2 = yêu$, $y_3 = bạn$.
- Khi đó, ta sẽ có các trạng thái như sau:
    - $h_1$ là trạng thái của RNN tại thời điểm đầu tiên (với input là Tôi)
    - $h_2$ là trạng thái của RNN tại thời điểm thứ hai (với input là yêu và thông tin ở thời điểm trước đó là Tôi)
    - $h_3$ là trạng thái của RNN tại thời điểm thứ ba (với input là bạn và thông tin ở thời điểm trước đó là Tôi yêu).

Hình ảnh này được lấy từ bài báo [Recurrent Neural Networks (RNNs): A gentle Introduction and Overview](https://arxiv.org/abs/1912.05911).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/2-recurrent-neural-network/feedforward_vs_recurrent.png" style="width: 900px;"/>

Một số thành phần mang lại sự khác biệt giữa ý tưởng của RNN và mạng nơ ron truyền thống (Feedforward Neural Network):
- Đầu tiên là $h_i$, $h_i$ được tính bằng $h_i = f(W_X x_i + W_H h_{i-1})$.
Với $i = 0$, tức là ở vị trí đầu tiên trọng mạng, lúc này ta chưa có giá trị $h_{-1}$.
Khi đó, ta thường khởi tạo ngẫu nhiên $h_{-1}$ hoặc khởi tạo $h_{-1}$ là vector 0.
- Tiếp theo là bộ trọng số $W_X, W_H, W_Y$.
Khác với việc mỗi lớp có bộ trọng số riêng trong mạng nơ ron truyền thông, trong RNN, $W_X, W_H, W_Y$ được sử dụng chung cho tất cả các lớp.
Do đó, kết quả đầu ra tại mỗi thời điểm của RNN phụ thuộc hoàn toàn vào giá trị đầu vào và giá trị trạng thái ở thời điểm đó.
- Cuối cùng là giá trị đầu ra của lớp cuối cùng.
Với việc số lượng các lớp của RNN không được xác định trước, điều này dẫn đến một câu hỏi về việc khi nào vòng lặp hồi quy sẽ kết thúc.
Trong xử lý ngôn ngữ tự nhiên, có một từ đặc biệt được gọi là "từ kết thúc" và khi mô hình RNN tính toán đầu ra là "từ kết thúc" thì vòng lặp hồi quy của mô hình RNN sẽ dừng lại.

Với ý tưởng trên, về lý thuyết, RNN có khả năng nhớ các thông tin được tính toán ở các bước trước đó.
Điều này có nghĩa là RNN có thể ghi nhớ và sử dụng được thông tin của một văn bản rất dài, tuy nhiên thực tế, nó chỉ có thể nhớ được thông tin của một vài bước trước đó.

## 2. Vấn đề phụ thuộc xa của RNN (long-term dependencies)

### 2.1. Ví dụ

Một điểm nổi bật của RNN chính là ý tưởng kết nối các thông tin phía trước để dự đoán cho hiện tại.
Nếu mà RNN có thể làm được việc đó thì chúng sẽ cực kì hữu dụng cho các bài toán xử lý ngôn ngữ, tuy nhiên, không phải lúc nào RNN cũng có thể làm được điều này.

Ví dụ, với câu “Các đám mây trên bầu trời”, ta chỉ cần đọc tới “Các đám mây trên bầu ...” là đủ biết được chữ tiếp theo là “trời” rồi.
Do đó, ta rút ra, với khoảng cách tới thông tin có được cần để dự đoán là nhỏ, nên RNN hoàn toàn có thể học được.

Ví dụ tiếp theo, với câu “Tôi được sinh ra và lớn lên ở Pháp. Món ăn ưa thích của tôi là bánh mì và tôi thường chơi bóng đá vào buổi chiều. Tôi có thể nói được ba ngoại ngữ, trong đó, tôi nói lưu loát nhất là tiếng Pháp.”.
Rõ ràng là các thông tin gần “tôi nói lưu loát nhất là” chỉ có phép ta biết được đằng sau nó sẽ là tên của một ngôn ngữ nào đó.
Do đó, ta cần phải có thêm ngữ cảnh “Tôi lớn lên ở Pháp.” thì mới có thể suy luận được.

### 2.2. Backpropagation Through Time (BPTT)

Backpropagation Through Time (BPTT) là một thuật toán được sử dụng để huấn luyện RNN.
BPTT là một biến thể của thuật toán lan truyền ngược (backpropagation), được thiết kế để xử lý các mạng nơ ron có cấu trúc hồi quy.

BPTT cho phép cập nhật trọng số của RNN bằng cách lan truyền ngược qua thời gian, từ đầu ra cuối cùng trở về đầu vào ban đầu.
BPTT hoạt động bằng cách "dérouler" (unroll) RNN theo thời gian, tức là biến RNN thành một mạng nơ ron truyền thống với nhiều lớp.
Điều này cho phép áp dụng thuật toán lan truyền ngược truyền thống để tính toán gradient và cập nhật trọng số.

Ví dụ: Xét một RNN với 3 bước thời gian $h_1, h_2, h_3$ và giá trị loss là $L$.

Ta có BPTT được tính như sau:

$$ \frac{\partial L}{\partial W_X} = \frac{\partial L}{\partial h_3} \cdot \frac{\partial h_3}{\partial W_X} + \frac{\partial L}{\partial h_2} \cdot \frac{\partial h_2}{\partial W_X} + \frac{\partial L}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_X} $$

trong đó:

$$ \frac{\partial L}{\partial h_2} = \frac{\partial L}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} $$

$$ \frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} $$

Với cách tính này, ta có thể coi RNN là một mạng nơ ron rất sâu với số lượng lớp bằng với số bước thời gian.
Tuy nhiên, với việc lan truyền ngược qua nhiều bước thời gian, BPTT có thể gặp phải vấn đề vanishing gradient và exploding gradient, ảnh hưởng đến khả năng học của RNN.

Đây chính là nguyên nhân chính dẫn đến việc RNN, trong thực tế, không thể học được các phụ thuộc xa trong chuỗi dữ liệu.
Từ đó, ta cần chỉnh sửa kiến trúc của RNN để có thể phần nào giải quyết được vấn đề này.

## 3. Kiến trúc mô hình Long short-term memory (LSTM)

LSTM là một dạng đặc biệt của RNN có khả năng học được các phụ thuộc xa tốt hơn so với RNN cơ bản, hoạt động cực kì hiệu quả trên nhiều bài toán xử lý ngôn ngữ khác nhau.

Hình ảnh này được lấy từ bài báo [Recurrent Neural Networks: A Comprehensive Review of Architectures, Variants, and Applications](https://www.mdpi.com/2078-2489/15/9/517) giúp mô tả chi tiết kiến trúc bên trong của một cell trong mô hình LSTM.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/2-recurrent-neural-network/lstm.png" style="width: 800px;"/>

Ý tưởng cốt lõi của LSTM là trạng thái tế bào (cell state) - đường chạy thông ngang phía trên của sơ đồ hình vẽ.
Cell state là một dạng giống như băng truyền. Nó chạy xuyên suốt tất cả các mắt xích (các nút mạng).
Vì vậy mà các thông tin có thể dễ dàng truyền đi thông suốt.

Hình ảnh này được lấy từ bài báo [Recurrent Neural Networks: A Comprehensive Review of Architectures, Variants, and Applications](https://www.mdpi.com/2078-2489/15/9/517), phần màu đỏ là cell state.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/2-recurrent-neural-network/lstm_cell_state.png" style="width: 800px;"/>

LSTM có khả năng bỏ đi hoặc thêm vào các thông tin cần thiết cho cell state, chúng được điều chỉnh cẩn thận bởi các cổng (gate).
Các cổng là nơi sàng lọc thông tin đi qua nó, chúng được kết hợp bởi một tầng mạng sigmoid và một phép nhân.

Một LSTM gồm có 3 cổng để duy trì và điều hành cell state là:
- **Forget Layer Gate**: Là nơi quyết định xem thông tin nào cần phải bỏ đi khỏi cell state
    - Thông tin được quyết định là sẽ được xoá đi được đưa ra bởi một lớp sigmoid.
- **Input Layer Gate**: Là nơi quyết định xem thông tin nào cần phải bổ sung thêm vào cell state và bổ sung thông tin đó vào cell state.
    - Đầu tiên, ta sử dụng một lớp sigmoid để quyết định giá trị nào ta sẽ cập nhật.
    - Tiếp theo, ta sử dụng một lớp tanh tạo ra một vector cho giá trị mới $C$ nhằm thêm vào cho cell state.
    - Cuối cùng, ta kết hợp hai giá trị đó lại để tạo ra một cập nhật cho cell state.
- **Output Layer Gate**: Là nơi quyết định xem ta muốn trả đầu ra ở bước này là gì và truyền thông tin gì cho bước tiếp theo. Output Layer Gate hoạt động khá giống với Input Layer Gate
    - Đầu tiên, ta sử dụng một lớp sigmoid để quyết định phần nào của giá trị đầu vào và giá trị của bước trước mà ta muốn trả đầu ra ở bước này.
    - Tiếp theo, ta sử dụng một lớp tanh để tạo ra một vector giá trị từ cell state.
    - Cuối cùng, ta kết hợp hai giá trị đó lại để tạo ra giá trị đầu ra ở bước này.

Hình ảnh này được lấy từ bài báo [Recurrent Neural Networks: A Comprehensive Review of Architectures, Variants, and Applications](https://www.mdpi.com/2078-2489/15/9/517), phần màu vàng là forget layer gate, màu xanh là input layer gate và màu tím là output layer gate.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/2-recurrent-neural-network/lstm_gates.png" style="width: 800px;"/>

Ví dụ về cách hoạt động của LSTM trong một mô hình ChatBot.

Xét đoạn hội thoại:

```
Minh: "Xin chào, tôi là Minh. "
Bot: "Chào Minh."
```

Ở đây, Bot cần phải ghi nhớ tên của người đang trò chuyện với nó là Minh.
Để làm được điều này, Bot sẽ ghi nhớ thông tin về tên của người đang trò chuyện với nó vào cell state bằng cách sử dụng Input Layer Gate.

```
Minh: "Xin chào, tôi là Minh. "
Bot: "Chào Minh."
Minh: "Tôi là một kỹ sư Trí tuệ nhân tạo. Sở thích của tôi là chơi đá bóng và đọc sách. Bạn có thể gợi ý cho tôi một vài cuốn sách hay được không?"
Bot: "Chắc chắn rồi, Minh. Bạn muốn đọc về thể thao hay công nghệ?"
Minh: "Tôi muốn đọc về công nghệ."
Bot: "Một số cuốn sách hay về công nghệ mà tôi biết là: 'Artificial Intelligence: A Modern Approach' của Stuart Russell và Peter Norvig, 'Deep Learning' của Ian Goodfellow, Yoshua Bengio và Aaron Courville."
```

Ở đây, Bot cần phải ghi nhớ sở thích của Minh và thông tin muốn đọc sách về công nghệ, cụ thể là về Trí tuệ nhân tạo.
Để làm được điều này, Bot sẽ ghi nhớ thông tin về sở thích của Minh và thông tin muốn đọc sách về công nghệ vào cell state bằng cách sử dụng Input Layer Gate và nó có thể quên đi thông tin về sở thích chơi đá bóng bằng cách sử dụng Forget Layer Gate.

## 4. Các biến thể của RNN nói chung và LSTM nói riêng

### 4.1. RNN hai chiều (Bidirectional RNN) và LSTM hai chiều (BiLSTM)

Với mô hình RNN hai chiều, đầu ra tại mỗi bước không những phụ thuộc vào các giá trị đầu vào và giá trị trạng thái phía trước mà còn phụ thuộc cả vào các giá trị phía sau.
Ví dụ, đối với bài toán điền từ còn thiếu trong câu, ta cần phải xem xét cả các giá trị phía trước trước và phía sau của từ cần điền.

Hình ảnh này được lấy từ bài báo [Recurrent Neural Networks (RNNs): A gentle Introduction and Overview](https://arxiv.org/abs/1912.05911).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/2-recurrent-neural-network/bidirectional_rnn.png" style="width: 600px;"/>

Từ đó, ta có thể coi mô hình Bidirectional RNN là việc chồng 2 mạng RNN ngược hướng lên nhau.
Lúc này đầu ra được tính toán dựa vào cả 2 trạng thái ẩn của 2 mạng RNN ngược hướng này.

Một phiên bản khác của Bidirectional RNN là Deep Bidirectional RNN - RNN hai chiều sâu.
Ở phiên bản này, ta không chỉ chồng 2 mạng RNN ngược hướng lên nhau mà ta chồng nhiều cặp mạng RNN ngược hướng lên nhau, từ đó, giúp tăng độ phức tạp của mô hình.

Hình ảnh này được lấy từ bài báo [Recurrent Neural Networks: A Comprehensive Review of Architectures, Variants, and Applications](https://www.mdpi.com/2078-2489/15/9/517).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/2-recurrent-neural-network/bilstm.png" style="width: 600px;"/>

Ngoài ra, ta cũng có thể áp dụng ý tưởng này cho LSTM, từ đó, ta có BiLSTM - LSTM hai chiều.

### 4.2. LSTM với peephole connections

LSTM với peephole connections được giới thiệu bởi bài báo [Recurrent Nets that Time and Count](https://sferics.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf).
Hình ảnh dưới đây được lấy từ bài báo này.

Đây là biến thể giúp cung cấp thêm thông tin về cell state trong các thời điểm cần đưa ra quyết định loại bỏ thông tin khỏi cell state (của Forget Layer Gate), bổ sung thông tin vào cell state (của Input Layer Gate) hay bổ sung thông tin vào kết quả đầu ra (của Output Layer Gate).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/2-recurrent-neural-network/peephole_connections.png" style="width: 1000px;"/>

Hình trên mô tả các đường được thêm vào mọi cổng, nhưng cũng có những nghiên cứu chỉ thêm cho một vài cổng mà thôi.

### 4.3. LSTM với coupled forget - input gates

Ở biến thể này, Forget Layer Gate và Input Layer Gate được kết hợp lại với nhau thành một cổng duy nhất giúp cân bằng giữa phần "quên" và phần bổ sung thêm.
Ta chỉ thêm thông tin mới vào cell state khi ta quên bớt thông tin gì đó hoặc ngược lại, ta chỉ quên bớt thông tin gì đó nếu ta bổ sung thêm thông tin mới vào cell state.

Hình ảnh này được lấy từ bài báo [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/2-recurrent-neural-network/coupled_input_forget_gate.png" style="width: 1000px;"/>

### 4.4. Gated Recurrent Unit (GRU)

GRU kết hợp Forget Layer Gate và Input Layer Gate thành Update Gate.
GRU cũng kết hợp cell state và hidden state lại với nhau để tạo ra một kiến trúc đơn giản hơn so với LSTM tiêu chuẩn.

Hình ảnh này được lấy từ bài báo [Recurrent Neural Networks: A Comprehensive Review of Architectures, Variants, and Applications](https://www.mdpi.com/2078-2489/15/9/517).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/6_natural_language_processing/images/2-recurrent-neural-network/gru.png" style="width: 600px;"/>
