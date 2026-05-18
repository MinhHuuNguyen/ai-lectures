---
time: 06/08/2022
title: Bài toán image generation
description: Các mô hình tạo sinh hình ảnh là một lĩnh vực đột phá, tập trung vào việc huấn luyện máy tính để tự tạo ra những hình ảnh mới, độc đáo và chân thực từ các dữ liệu đầu vào. Các mô hình này không chỉ đơn thuần "sao chép" hay "ghép" các phần của những bức ảnh có sẵn. Thay vào đó, chúng học các khái niệm, thuộc tính, phong cách và mối quan hệ giữa các đối tượng từ một tập dữ liệu khổng lồ. Từ đó, chúng có khả năng "tưởng tượng" và tổng hợp nên một hình ảnh hoàn toàn mới dựa trên yêu cầu của người dùng.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/banner.jpeg
tags: [deep-learning, computer-vision]
is_highlight: false
is_published: true
---

## 1. Giới thiệu chung về image generation

Bài toán image generation là nhiệm vụ xây dựng các mô hình có khả năng tạo ra hình ảnh mới sao cho chúng trông chân thực hoặc phù hợp với một mô tả đầu vào.
Đầu vào có thể là một vector bất kỳ, một hình ảnh khác, một đoạn văn bản ...
Đây là một hướng nghiên cứu quan trọng của trí tuệ nhân tạo, nơi máy học cách nắm bắt đặc trưng của dữ liệu ảnh để sinh ra ảnh chưa từng tồn tại trước đó.

Một thách thức lớn là ảnh sinh ra cần vừa đẹp về thị giác, vừa hợp ngữ nghĩa, đồng thời tránh lỗi như méo hình, thiếu chi tiết hoặc không đúng với yêu cầu đầu vào.
Tóm lại, sinh ảnh là bài toán giúp máy tính không chỉ "nhìn" và "hiểu" ảnh, mà còn có thể sáng tạo ra ảnh mới, mở ra rất nhiều ứng dụng thực tiễn trong đời sống và công nghệ.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/applications.jpeg" style="width: 1000px;"/>

Image generation có ứng dụng rộng khắp, từ giải trí đến sáng tạo nghệ thuật số, hỗ trợ y tế, quảng cáo, giáo dục, đến giám sát an ninh.
Ví dụ:
- **Sáng tạo nhân vật game, điện ảnh:** Hệ thống AI tự động hóa design, tạo phong cách đa dạng, phục vụ sản xuất nhanh chóng, hiệu quả.
- **Phục chế và nâng cấp ảnh cũ:** AI hồi phục ảnh cổ, tăng độ phân giải, đem lại giá trị lịch sử, nghệ thuật.
- **Media, quảng cáo:** Tạo ảnh/clip cá nhân hóa, truyền thông sáng tạo.
- **Giáo dục:** Hỗ trợ trực quan hóa, sinh dữ liệu giả phục vụ nghiên cứu, luyện tập.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/image_synthesis_translation.jpeg" style="width: 1000px;"/>

**Image synthesis** là bài toán tạo ra ảnh mới từ nhiễu ngẫu nhiên.
Mục tiêu của bài toán này là sinh ra những hình ảnh chân thực, đa dạng và phù hợp với nội dung mong muốn, chẳng hạn như tạo khuôn mặt người không có thật.

**Image-to-image translation** là bài toán biến đổi một ảnh đầu vào thành một ảnh đầu ra nhưng vẫn giữ lại một phần cấu trúc hoặc nội dung gốc.
Ví dụ:
- Chuyển ảnh phác thảo thành ảnh thật
- Chuyển ảnh đen trắng thành ảnh màu
- Chuyển ảnh chụp ban ngày thành ảnh ban đêm

Khác với image synthesis, bài toán này không tạo ảnh hoàn toàn từ đầu mà tập trung vào ánh xạ từ miền ảnh này sang miền ảnh khác.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/text_to_image.jpeg" style="width: 500px;"/>

**Text-to-image** nhằm tạo ra hình ảnh từ mô tả bằng văn bản.
Cụ thể, mô hình nhận đầu vào là một câu hoặc đoạn mô tả như "một chú chó đang chạy trên bãi biển lúc hoàng hôn", sau đó sinh ra một bức ảnh phù hợp với nội dung ngữ nghĩa, bối cảnh và chi tiết thị giác được nhắc đến trong văn bản.
Mục tiêu của bài toán này là tạo ảnh vừa chân thực, vừa đúng với mô tả, đồng thời vẫn đảm bảo tính đa dạng và thẩm mỹ trong kết quả sinh ra.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/1-computer-vision/inpainting_outpainting.jpeg" style="width: 500px;"/>

**Image inpainting** là bài toán khôi phục hoặc điền vào những vùng bị thiếu, bị che khuất, hoặc bị hư hỏng trong ảnh sao cho phần được sinh ra tự nhiên và phù hợp với ngữ cảnh xung quanh.
Ví dụ, mô hình có thể xóa một vật thể không mong muốn trong ảnh rồi tự động điền lại nền phía sau một cách hợp lý.

**Image outpainting** là bài toán mở rộng ảnh ra ngoài phạm vi ban đầu bằng cách sinh thêm nội dung mới ở các vùng biên, nhưng vẫn phải giữ được sự nhất quán về bố cục, màu sắc và ngữ nghĩa với ảnh gốc.
Ví dụ, từ một bức ảnh phong cảnh hẹp, mô hình có thể mở rộng thêm bầu trời, núi hoặc mặt đất để tạo thành một khung hình lớn hơn.

## 2. Nhóm các phương pháp giải bài toán image generation

### 2.1. Mô hình Variational Autoencoders

**VAE - Variational Autoencoder** được giới thiệu vào năm 2013 trong bài báo [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114) là một loại mô hình sinh (generative model) với mục tiêu không chỉ là nén và tái tạo lại dữ liệu (như autoencoder tiêu chuẩn) mà học một **không gian ẩn (latent space)** có cấu trúc và liên tục.
Từ latent space, mô hình có thể "sinh" ra những dữ liệu mới chưa từng tồn tại nhưng vẫn hợp lý và giống với dữ liệu gốc.

Ví dụ, hãy tưởng tượng bạn là một họa sĩ:
- **Học (Training):** Bạn xem hàng ngàn bức ảnh về khuôn mặt người.
Thay vì nhớ thuộc lòng từng chi tiết của mỗi bức ảnh, bạn học được những "đặc trưng cốt lõi" của một khuôn mặt: mắt, mũi, miệng trông như thế nào, tỷ lệ ra sao...
Toàn bộ kiến thức này được bạn nén vào trong não bộ.
- **Sáng tạo (Generation):** Khi có người yêu cầu bạn vẽ một khuôn mặt hoàn toàn mới, bạn không chép lại một bức ảnh nào đã xem.
Thay vào đó, bạn kết hợp các đặc trưng cốt lõi trong đầu mình để vẽ ra một người không có thật.

Trong ví dụ này:
- Bộ não của bạn chính là **Không gian ẩn (Latent Space)**.
- Quá trình bạn học từ ảnh thật chính là **Encoder (Bộ mã hóa)**.
- Quá trình bạn vẽ ra ảnh mới chính là **Decoder (Bộ giải mã)**.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/10-vae/banner.jpeg" style="width: 800px;"/>

Ưu điểm của VAEs là:
- **Quá trình huấn luyện ổn định:** VAEs có một hàm mục tiêu rõ ràng (Evidence Lower Bound - ELBO) để tối ưu hóa nên ổn định hơn nhiều so với GANs và chắc chắn sẽ hội tụ.
- **Latent Space có ý nghĩa:** Do được ràng buộc phải tuân theo một phân phối xác suất (VD: phân phối chuẩn), latent space của VAEs rất mượt mà và liên tục. 
Điều này giúp các tác vụ như nội suy (VD: biến đổi từ từ một khuôn mặt này sang một khuôn mặt khác).
- **Có thể ước lượng xác suất:** VAEs cung cấp một cách để ước lượng xác suất của một điểm dữ liệu, điều mà GANs không làm được.

Nhược điểm của VAEs là:
- **Chất lượng ảnh sinh ra bị mờ:** Đây là nhược điểm cố hữu của VAEs.
Do hàm mất mát có thành phần tái tạo (reconstruction loss) thường dùng sai số bình phương trung bình (MSE), mô hình có xu hướng tạo ra các ảnh "trung bình", an toàn, dẫn đến kết quả bị mờ và thiếu các chi tiết sắc nét so với GANs.
- **Vấn đề "Prior Hole":** Đôi khi, Decoder có thể học cách "lờ đi" thông tin từ không gian ẩn và chỉ tạo ra một kết quả trung bình cho mọi đầu vào, làm cho mô hình trở nên vô dụng.

### 2.2. Nhóm mô hình Generative Adversarial Networks - GANs

**GANs - Generative Adversarial Networks** được giới thiệu bởi Ian Goodfellow và các đồng nghiệp vào năm 2014 trong bài báo [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) với ý tưởng cốt lõi là xây dựng và huấn luyện một mô hình như một trò chơi giữa hai người chơi:
- **Generator - Kẻ làm giả:** Một họa sĩ chuyên làm giả các tác phẩm nghệ thuật. Mục tiêu của anh ta là tạo ra những bức tranh giả tinh vi đến mức không thể phân biệt được với tranh thật.
- **Discriminator - Chuyên gia nghệ thuật:** Một chuyên gia có con mắt tinh tường. Nhiệm vụ của ông là xem một bức tranh và xác định xem đó là tranh thật (từ một bộ sưu tập gốc) hay tranh giả (do kẻ làm giả tạo ra).

Trò chơi này diễn ra như sau:
- Ban đầu, kẻ làm giả còn non tay, các bức tranh giả rất dễ bị phát hiện. Chuyên gia dễ dàng chỉ ra đâu là giả.
- Kẻ làm giả nhận được phản hồi (biết mình đã bị phát hiện) và học hỏi từ những sai lầm để vẽ ra những bức tranh giả ngày càng tốt hơn.
Khi kẻ làm giả giỏi lên, chuyên gia cũng phải rèn luyện con mắt của mình để trở nên tinh tường hơn, tìm ra những chi tiết nhỏ nhất để phân biệt thật-giả.
Quá trình "đối đầu" này tiếp tục, cả hai cùng tiến bộ.
- Cuối cùng, khi trò chơi đạt đến trạng thái cân bằng, kẻ làm giả sẽ tạo ra những tác phẩm giả hoàn hảo đến mức chuyên gia chỉ có thể đoán bừa (với xác suất 50/50) xem nó là thật hay giả.
Khi đó, chúng ta đã có một Generator cực kỳ tài năng, có khả năng tạo ra những hình ảnh siêu thực.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/9-gans/banner.jpeg" style="width: 800px;"/>

Một số cái tên mô hình nổi bật nhất trong nhóm GANs bao gồm:
- **Progressive Growing of GANs**, **Style GAN** và **Style GAN 2**: 

Ưu điểm của GANs là:
- **Chất lượng ảnh sinh ra tốt:** Do tính chất đối nghịch, Generator bị "ép" phải tạo ra các ảnh cực kỳ sắc nét và chi tiết để có thể qua mặt được Discriminator.
- **Tốc độ sinh ảnh cực nhanh:** Sau khi đã được huấn luyện, Generator là một feed-forward model.
Do đó, chỉ cần đưa một vector nhiễu (noise vector) vào và nhận ngay kết quả đầu ra.

Nhược điểm của GANs là:
- **Quá trình huấn luyện không ổn định:** Việc cân bằng giữa Generator và Discriminator rất khó. Nếu một trong hai thành phần mạnh hơn quá nhiều, quá trình học sẽ sụp đổ và không hội tụ.
- **Mode Collapse:** Generator có thể overfit bằng cách chỉ học và tạo ra **một vài** ảnh trông rất thật mà lừa được Discriminator. Kết quả là mô hình chỉ sinh ra được một vài loại ảnh giống nhau, thiếu sự đa dạng, dù tập dữ liệu huấn luyện rất phong phú.

### 2.3. Nhóm mô hình dựa trên Transformer

Sự ra đời của Transformer cùng với cơ chế tự chú ý (self-attention) 

### 2.4. Nhóm mô hình Diffusion Models

**Diffusion Models** là một trong những kiến trúc mô hình đột phá và mạnh mẽ nhất trong lĩnh vực Generative AI nói chung và Image Generation nói riêng hiện nay.
Nền tảng lý thuyết của Diffusion Models được giới thiệu vào năm 2015 trong bài báo [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585) trong khi ứng dụng cụ thể được áp dụng và mang lại bước đột phá được giới thiệu vào năm 2020 trong bài báo [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239).

Ý tưởng Cốt lõi của Diffusion Model là điêu khắc từ một khối nhiễu không có nội dung cụ thể.
- Bắt đầu với một bức tượng hoàn hảo (một bức ảnh sạch, rõ nét).
- Từ từ phá hủy nó bằng cách thêm vào từng lớp "bụi" hoặc "nhiễu" (noise) cho đến khi nó trở thành một khối bụi/nhiễu không còn hình thù gì.
- Học cách đảo ngược quá trình này: Tức là học cách "thổi bụi" ra khỏi khối nhiễu để khôi phục lại bức tượng ban đầu.
- Sau khi đã học được kỹ năng "thổi bụi" này một cách thành thạo, mô hình có thể bắt đầu với một khối nhiễu hoàn toàn ngẫu nhiên và "điêu khắc" nó thành một bức ảnh mới toanh, chân thực và độc đáo.

Đó chính là triết lý của Diffusion Model, được chia thành hai quá trình chính:
- **Quá trình Thuận (Forward Process / Diffusion Process):** Thêm nhiễu vào ảnh.
- **Quá trình Nghịch (Reverse Process / Denoising Process):** Loại bỏ nhiễu để tạo ra ảnh.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/blog-sharing/refs/heads/master/1_pixta_seminar/12_from_diffusion_to_flow/diffusion_model_img/banner.jpeg" style="width: 800px;"/>

Ưu điểm của Diffusion Models là:
- **Chất lượng và độ đa dạng ảnh vượt trội:** Diffusion Models có khả năng tạo ra những hình ảnh cực kỳ chi tiết, chân thực và đa dạng, vượt qua GANs.
- **Quá trình huấn luyện ổn định:** Tương tự VAEs, Diffusion Models có một hàm mục tiêu được định nghĩa rõ ràng, giúp quá trình huấn luyện rất ổn định và dễ hội tụ.
- **Hạn chế Mode Collapse:** Chúng rất ít khi gặp phải vấn đề sụp đổ chế độ như GANs, có khả năng học và tái tạo lại toàn bộ sự đa dạng của dữ liệu huấn luyện.

Nhược điểm của Diffusion Models là:
- **Tốc độ sinh ảnh rất chậm:** Để tạo ra một ảnh, mô hình phải thực hiện quá trình khử nhiễu lặp đi lặp lại qua hàng chục, hàng trăm, thậm chí hàng nghìn bước.
Quá trình này chậm hơn rất nhiều so với một lần truyền thẳng của GAN.
Tuy nhiên, các nghiên cứu gần đây đang cải thiện tốc độ này một cách đáng kể.
- **Yêu cầu tính toán cao:** Cả quá trình huấn luyện và sinh ảnh đều đòi hỏi tài nguyên tính toán (GPU/TPU) rất lớn do bản chất lặp đi lặp lại của chúng.

Một số cái tên nổi bật trong nhóm Diffusion Models bao gồm:


## 3. Các metrics trong image generation

Một mô hình sinh ảnh tốt cần đáp ứng hai tiêu chí chính:
- **Chất lượng (Quality/Fidelity):** Các ảnh được sinh ra phải rõ nét, chân thực và có thể nhận biết được đối tượng trong đó.
Ví dụ, nếu mô hình sinh ra ảnh một con chó, nó phải trông giống một con chó thật, chứ không phải một vệt mờ.
- **Đa dạng (Diversity/Variety):** Mô hình phải có khả năng sinh ra nhiều loại ảnh khác nhau, bao phủ được sự đa dạng của dữ liệu thật. Nó không nên chỉ sinh đi sinh lại một vài kiểu ảnh đẹp (hiện tượng này gọi là "mode collapse").

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/metrics.jpeg" style="width: 800px;"/>

### 3.1. Inception Score (IS)

Inception Score được giới thiệu vào năm 2016 sử dụng một mô hình Inception-v3 pretrained trên bộ dữ liệu ImageNet.
Mục tiêu của IS là đo lường đồng thời cả chất lượng và sự đa dạng của các ảnh được sinh ra.
- **Để đo chất lượng:** IS đưa một ảnh qua mô hình Inception-v3 và lấy prediction của mô hình:
    - Nếu phân phối xác suất đầu ra $p(y|x)$ có entropy thấp (tức là có một đỉnh nhọn ở một lớp), nghĩa là mô hình tự tin dự đoán ảnh đó thuộc về một lớp cụ thể nào đó.
    Suy ra, **Đây là một ảnh đẹp và rõ ràng nội dung.**
    - Nếu phân phối xác suất đầu ra $p(y|x)$ có entropy cao (tức là phân phối xác suất khá đồng đều giữa các lớp), nghĩa là mô hình không tự tin dự đoán ảnh đó thuộc về một lớp cụ thể nào đó.
    Suy ra, **Đây là một ảnh xấu và không rõ ràng nội dung.**
- **Để đo sự đa dạng:** IS đưa một tập hợp ảnh qua mô hình Inception-v3 và lấy tổng hợp prediction của mô hình:
    - Nếu tổng hợp của các phân phối xác suất đầu ra $p(y|x)$ có entropy thấp (tức là có một đỉnh nhọn ở một lớp), nghĩa là mô hình sinh ra các ảnh tập trung ở một lớp cụ thể nào đó.
    Suy ra, **Mô hình đang thiếu sự đa dạng.**
    - Nếu tổng hợp của các phân phối xác suất đầu ra $p(y|x)$ có entropy cao (tức là phân phối xác suất khá đồng đều giữa các lớp), nghĩa là mô hình sinh ra các ảnh phân bố đều trên nhiều lớp khác nhau.
    Suy ra, **Mô hình đang có sự đa dạng.**

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/is.jpeg" style="width: 800px;"/>

#### Cách thức hoạt động

- Lấy một tập hợp ảnh được sinh ra (VD: 50,000 ảnh) từ mô hình của bạn.
- Đưa từng ảnh qua mô hình pre-trained Inception-v3 để nhận được vector xác suất 1000 chiều $p(y|x)$ (tương ứng với 1000 lớp của ImageNet).
- Tính toán phân phối xác suất trung bình (marginal distribution) bằng cách lấy trung bình của tất cả các vector $p(y|x)$ thu được $p(y) = E_x[p(y|x)]$.
- Tính toán Kullback-Leibler (KL) Divergence giữa $p(y|x)$ và $p(y)$ cho mỗi ảnh, sau đó lấy trung bình KL Divergence trên tất cả các ảnh.
Chúng ta muốn $p(y|x)$ (đặc trưng cho chất lượng - phải có entropy thấp) rất khác biệt so với $p(y)$ (đặc trưng cho sự đa dạng - phải có entropy cao).
- Công thức cuối cùng, hàm $exp$ được sử dụng để đưa kết quả về một thang đo dễ đọc hơn: $IS = exp(E_x[D_KL(p(y|x) || p(y))])$
- Kết luận: Điểm IS càng cao càng tốt.
Nó cho thấy các ảnh sinh ra vừa có chất lượng cao (dự đoán tự tin) vừa đa dạng (bao phủ nhiều lớp).

#### Ưu và nhược điểm

**Ưu điểm:**
- **Đơn giản để tính toán.**
- **Không cần sử dụng ảnh thật trong quá trình đánh giá.**

**Nhược điểm:**
- **Không so sánh với dữ liệu thật:** IS chỉ nhìn vào các ảnh được sinh ra mà không đối chiếu chúng với phân phối của ảnh thật.
- **Dễ bị "qua mặt":** Một mô hình có thể học cách tạo ra **một ảnh hoàn hảo cho mỗi lớp** trong 1000 lớp của ImageNet. Nó sẽ nhận được điểm IS rất cao, nhưng thực tế nó không có khả năng sinh ra các biến thể khác của đối tượng.
- **Phụ thuộc vào ImageNet và Inception-v3:** Nó hoạt động không tốt với các bộ dữ liệu không có các lớp tương tự ImageNet và bị giới hạn bởi khả năng của mô hình Inception-v3.

### 3.2. Fréchet Inception Distance (FID)

Fréchet Inception Distance (FID) được giới thiệu vào năm 2017 và đã khắc phục được một số nhược điểm của IS.
FID đo lường "khoảng cách" giữa phân phối của các ảnh được sinh ra và phân phối của các ảnh thật.
Nó không chỉ nhìn vào đầu ra của lớp phân loại mà còn xem xét các đặc trưng sâu hơn bên trong mạng nơ-ron.

FID cũng sử dụng mô hình Inception-v3, tương tự IS nhưng thay vì lấy lớp đầu ra (softmax), FID lấy các vector đặc trưng (feature vectors) từ một layer trong model.
Vector này được coi là một biểu diễn cô đọng về nội dung của ảnh và FID sẽ sử dụng vector này để đo đạc đánh giá.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/fid.jpeg" style="width: 800px;"/>

#### Cách thức hoạt động

- Chuẩn bị trước một tập hợp ảnh thật từ bộ dữ liệu của bạn (ví dụ: 10,000 ảnh).
- Lấy một tập hợp ảnh được sinh ra (VD: 50,000 ảnh) từ mô hình của bạn.
- Đưa từng ảnh của cả hai tập hợp ảnh qua mô hình Inception-v3 và thu thập các vector đặc trưng cho mỗi tập.
- Giả sử, các vector đặc trưng của mỗi ảnh trong mỗi tập tuân theo một phân phối Gaussian, ta tính toán vector giá trị kỳ vọng $\mu$ và ma trận hiệp phương sai $\Sigma$ cho cả hai tập: ảnh thật $(\mu_r, \Sigma_r)$ và ảnh sinh ra $(\mu_g, \Sigma_g)$.
- FID chính là khoảng cách Fréchet giữa hai phân phối Gaussian với công thức: $$FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$$
    - $||\mu_r - \mu_g||^2$ là khoảng cách bình phương giữa hai vector trung bình, đo lường sự khác biệt về nội dung trung bình của hai tập ảnh.
    - $Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$ là trace của ma trận, đo lường sự khác biệt về ma trận hiệp phương sai, tức là sự khác biệt về sự đa dạng và tương quan giữa các đặc trưng.
- Kết luận: Điểm FID càng thấp càng tốt.
FID bằng 0 có nghĩa là phân phối của ảnh sinh ra và ảnh thật giống hệt nhau (trong không gian đặc trưng của Inception).

#### Ưu và nhược điểm

**Ưu điểm:**
- **So sánh trực tiếp với dữ liệu thật:** Đây là cải tiến quan trọng nhất so với IS.
- **Nhạy cảm với Mode Collapse:** Nếu mô hình chỉ sinh ra một vài loại ảnh, ma trận hiệp phương sai Σ_g sẽ rất khác so với Σ_r, dẫn đến điểm FID cao.
- **Tương quan tốt hơn với đánh giá của con người:** Điểm FID thấp thường tương ứng với các ảnh có chất lượng và độ đa dạng cao theo cảm nhận của con người.

**Nhược điểm:**
- **Yêu cầu tính toán lớn:** Cần một lượng lớn mẫu (thường là 10,000 đến 50,000) để có được điểm số ổn định.
- **Phụ thuộc vào ImageNet và Inception-v3:** Nó hoạt động không tốt với các bộ dữ liệu không có các lớp tương tự ImageNet và bị giới hạn bởi khả năng của mô hình Inception-v3.

### 3.3. CLIP-MMD (CMMD)

CMMD (CLIP Maximum Mean Discrepancy) đã nổi lên như một thước đo (metric) tiêu chuẩn mới, được kỳ vọng sẽ khắc phục những hạn chế của các phương pháp trước đây.

CMMD là sự kết hợp giữa hai thành phần:
- **Mô hình CLIP (Contrastive Language-Image Pre-training):** Thay vì sử dụng mạng Inception-v3, CMMD dùng CLIP để trích xuất đặc trưng vì CLIP có khả năng hiểu hình ảnh gần với thị giác và ngữ nghĩa của con người hơn.
- **Maximum Mean Discrepancy (MMD):** Đây là một phương pháp thống kê dùng để đo lường khoảng cách giữa hai phân phối dữ liệu (giữa ảnh thật và ảnh do máy sinh ra), thay thế cho độ đo Fréchet.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/cmmd.jpeg" style="width: 800px;"/>

#### Khoảng cách Maximum Mean Discrepancy - MMD


### 3.4. Learned Perceptual Image Patch Similarity (LPIPS)

LPIPS (Learned Perceptual Image Patch Similarity), hay còn gọi là "Perceptual Loss", là một thước đo được thiết kế để đánh giá sự tương đồng giữa hai hình ảnh theo cách gần giống với cách con người cảm nhận.

Thay vì so sánh từng pixel một cách máy móc, LPIPS sử dụng một mạng nơ-ron sâu (Deep Neural Network) đã được huấn luyện trước để trích xuất các đặc trưng phức tạp và so sánh chúng.
Điểm LPIPS càng thấp thì hai hình ảnh càng giống nhau về mặt tri giác.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/lpips.jpeg" style="width: 800px;"/>

#### Các metrics trước LPIPS

Ta sử dụng các thước đo như sau để đánh giá mức độ tương đồng của hai hình ảnh.
Tuy nhiên, các thước đo này có một nhược điểm lớn: chúng không tương quan tốt với nhận thức của con người.
- **L1 Loss (Mean Absolute Error - MAE):** Tính trung bình của trị tuyệt đối của sự khác biệt giữa các pixel.
- **L2 Loss (Mean Squared Error - MSE):** Tính trung bình của bình phương sự khác biệt giữa các pixel.
- **PSNR (Peak Signal-to-Noise Ratio):** Dựa trên MSE.
- **SSIM (Structural Similarity Index):** Cố gắng đo lường sự thay đổi về cấu trúc, độ sáng và độ tương phản.

#### Ví dụ

Xét 3 ảnh:
- **Ảnh gốc (Original):** Một bức ảnh con mèo sắc nét.
- **Ảnh dịch chuyển (Shifted):** Cùng bức ảnh đó nhưng được dịch sang phải 1 pixel.
- **Ảnh mờ (Blurred):** Một phiên bản bị làm mờ của ảnh gốc.

Đánh giá 3 ảnh trên:
- **Theo mắt người:** Ảnh gốc (1) và ảnh dịch chuyển (2) gần như giống hệt nhau. Ảnh mờ (3) thì có chất lượng kém hơn hẳn.
- **Theo L2 Loss (MSE):** Điểm L2 giữa ảnh gốc (1) và ảnh dịch chuyển (2) sẽ rất cao (tức là rất khác nhau) vì mọi pixel đều bị lệch.
Ngược lại, điểm L2 giữa ảnh gốc (1) và ảnh mờ (3) có thể sẽ thấp hơn (tức là giống nhau hơn), điều này hoàn toàn trái ngược với cảm nhận của chúng ta.

Vấn đề này xảy ra vì L1/L2 chỉ quan tâm đến giá trị pixel tại đúng một vị trí, chúng không hiểu được "khái niệm" con mèo, kết cấu lông hay các đặc trưng bậc cao.

#### Cách thức hoạt động

- 
- Đưa từng ảnh qua mô hình pre-trained CNN nào đó để nhận được các vector đặc trưng ở nhiều lớp của mô hình.
LPIPS mong muốn nắm bắt các đặc trưng bậc thấp như cạnh, góc, màu sắc... ở những layer nông và nắm bắt các đặc trưng bậc cao, mang tính ngữ nghĩa hơn như các bộ phận của vật thể ở những layer sâu.
- Các vector đặc trưng ở mỗi lớp của hai ảnh sẽ được chuẩn hóa và tính khoảng cách L2.
Kết quả là một giá trị đo lường sự khác biệt về đặc trưng tại lớp đó.
- Tính tổng có trọng số các giá trị đo lường sự khác biệt ở các lớp để ra được điểm LPIPS cuối cùng.
    - Các nhà phát triển LPIPS đã huấn luyện một mạng tuyến tính nhỏ để học các trọng số cho mỗi lớp.
    - Các trọng số này được học từ một bộ dữ liệu lớn gồm các phán đoán của con người về sự tương đồng của các cặp ảnh.
    - Mục tiêu là để trọng số này phản ánh tầm quan trọng của từng loại đặc trưng đối với nhận thức của con người.
    - Điểm LPIPS cuối cùng là tổng của các khoảng cách ở mỗi lớp nhân với trọng số tương ứng đã học được.
- 

#### Ưu và nhược điểm

**Ưu điểm:**
- **Tương quan cao với tri giác con người:** Cách thức đánh giá của LPIPS gần với cách con người quan sát và đánh giá.
- **Bất biến với các thay đổi nhỏ:** LPIPS không quá nhạy cảm với các phép dịch chuyển, xoay nhẹ hoặc biến dạng nhỏ mà không làm thay đổi nội dung chính của ảnh.
- **Đánh giá được cả cấu trúc và phong cách:** Bằng cách sử dụng các lớp sâu, LPIPS có thể so sánh sự tương đồng về mặt ngữ nghĩa và phong cách, chứ không chỉ là kết cấu bề mặt.

**Nhược điểm:**
- **Chi phí tính toán cao:** LPIPS đòi hỏi phải thực hiện tính toán với mô hình CNN nên tốn nhiều tài nguyên và thời gian hơn.
- **Phụ thuộc vào mô hình CNN:** Kết quả của LPIPS phụ thuộc vào mạng CNN được sử dụng.
- **Không nắm bắt được mọi khía cạnh:** LPIPS có thể không đánh giá tốt các lỗi về mặt logic hoặc ngữ cảnh toàn cục trong một bức ảnh (VD: một người có ba tay).

### 3.5. Human evaluation

Đánh giá bởi con người (Human Evaluation) đóng một vai trò cực kỳ quan trọng và không thể thiếu trong lĩnh vực tạo sinh ảnh (Image Generation).
Mặc dù chúng ta có các chỉ số đánh giá tự động nhưng chúng vẫn còn nhiều hạn chế và không thể nắm bắt được toàn bộ chất lượng của một bức ảnh do AI tạo ra.

Đánh giá của con người được coi là thước đo cuối cùng và chính xác nhất về chất lượng của một mô hình tạo sinh ảnh.
Mọi chỉ số tự động đều được phát triển và kiểm chứng bằng cách so sánh xem chúng có tương quan tốt với nhận xét của con người hay không.
Nếu một chỉ số tự động cho điểm cao một mô hình mà con người lại đánh giá thấp, thì chỉ số đó được xem là chưa hiệu quả.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/human_evaluation.jpeg" style="width: 800px;"/>

#### Cách thức hoạt động

- **Đánh giá so sánh:** Người đánh giá được cho xem hai bức ảnh và được yêu cầu chọn ra bức ảnh tốt hơn dựa trên một tiêu chí cụ thể (VD: tính chân thật, độ thẩm mỹ ...)
- **Đánh giá theo thang đo:** Người đánh giá cho điểm một bức ảnh theo thang điểm (VD: từ 1 đến 5) cho các tiêu chí như chất lượng hình ảnh, tính thẩm mỹ, sự phù hợp với prompt.

#### Nhược điểm tồn đọng

- **Tốn kém và tốn thời gian:** Cần phải thuê nhiều người để đánh giá hàng ngàn, hàng triệu bức ảnh.
- **Tính chủ quan và thiên vị:** Đánh giá có thể khác nhau giữa những người khác nhau do sở thích, nền tảng văn hóa.
Do đó, ta cần đo đạc chỉ số độ đồng thuận giữa những người đánh giá.
- **Thiếu nhất quán:** Cùng một người có thể đưa ra những đánh giá khác nhau vào những thời điểm khác nhau.

### 3.6. Một số chỉ số đánh giá phụ khác

#### 3.6.1. CLIP Score

CLIP Score được xây dựng dựa trên mô hình CLIP là một metric định lượng mức độ tương đồng về mặt ngữ nghĩa giữa một hình ảnh và một đoạn mô tả văn bản, được sử dụng rất nhiều trong quá trình đánh giá mô hình Text-to-Image.
Điểm số càng cao, hình ảnh càng khớp với mô tả.

Chi tiết hơn về CLIP và một số biến thể nâng cấp đã được mình viết trong [bài viết này](/blog/transfer-learning-weakly-semi-un-va-self-supervised-learning).

#### Cách thức hoạt động

- Lấy một cặp: ảnh được sinh ra và prompt dùng để sinh ảnh đó.
- Dùng Text Encoder của CLIP để biến prompt thành text_embedding.
- Dùng Image Encoder của CLIP để biến ảnh thành image_embedding.
- Tính toán độ tương đồng cosine (cosine similarity) giữa hai vector text_embedding và image_embedding.
- Kết quả của phép tính này chính là CLIP Score:
    - **Điểm cao (gần 1):** Hai vector gần như chỉ về cùng một hướng trong không gian vector, nghĩa là hình ảnh và văn bản có sự tương đồng ngữ nghĩa cao.
    - **Điểm thấp (gần 0 hoặc âm):** Hai vector chỉ về các hướng khác nhau, cho thấy hình ảnh và văn bản không liên quan.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/clip_score.jpeg" style="width: 800px;"/>

#### Ưu và nhược điểm

**Ưu điểm:**
- **Hiểu ngữ nghĩa:** Nó không chỉ so khớp từ khóa, mà còn hiểu được các khái niệm, phong cách, và mối quan hệ phức tạp.

**Nhược điểm:**
- **Bias của CLIP:** CLIP được huấn luyện trên dữ liệu từ internet, nên nó cũng "học" cả những thiên kiến có sẵn trong dữ liệu đó (VD: thiên kiến về giới tính, chủng tộc).
-** Không phải là thước đo về thẩm mỹ:** Một ảnh có thể có CLIP Score rất cao (khớp hoàn hảo với prompt) nhưng trông lại không đẹp hoặc kỳ dị về mặt bố cục vì CLIP không phải là một nhà phê bình nghệ thuật.
- **Có thể bị "đánh lừa":** Đôi khi, các hình ảnh chứa văn bản (VD: ảnh có chữ "Apple" viết trên đó) có thể đạt điểm cao khi prompt là "quả táo", mặc dù đó không phải là thứ người dùng muốn.

#### 3.6.2. NIMA Score

NIMA là viết tắt của Neural Image Assessment là một mô hình được thiết kế để đánh giá chất lượng của một hình ảnh theo cách mà con người cảm nhận được giới thiệu trong bài báo [NIMA: Neural Image Assessment](https://arxiv.org/pdf/1709.05424).

Trong khi đó, NIMA cố gắng mô phỏng sự đánh giá của con người, không chỉ xem xét chất lượng kỹ thuật (technical quality) mà còn cả chất lượng thẩm mỹ (aesthetic quality).
Một bức ảnh có thể sắc nét về mặt kỹ thuật nhưng bố cục, màu sắc lại không hài hòa, NIMA có thể nhận ra điều này.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/nima_score.jpeg" style="width: 800px;"/>

#### Cách thức hoạt động

NIMA sử dụng một mô hình như Inception, VGG hay MobileNet và được tinh chỉnh trên một bộ dữ liệu lớn gồm các hình ảnh đã được con người chấm điểm về mặt thẩm mỹ như bộ dữ liệu AVA - Aesthetic Visual Analysis.

Output của NIMA không phải là một con số duy nhất, mà là một phân phối điểm số (từ 1 đến 10).
Từ phân phối này, chúng ta có thể tính ra điểm trung bình và độ lệch chuẩn.
Điểm trung bình càng cao, hình ảnh càng được đánh giá là có chất lượng và tính thẩm mỹ tốt.

Trong bài toán sinh ảnh, mục tiêu là tạo ra những hình ảnh không chỉ đúng với mô tả (prompt) mà còn phải đẹp, chân thực và hấp dẫn về mặt thị giác. Đây chính là lúc NIMA phát huy vai trò của mình.

NIMA có thể tự động chấm điểm cho tất cả các ảnh được tạo ra.
Và ta có thể chỉ giữ lại những ảnh có NIMA score cao, loại bỏ những ảnh bị lỗi, mờ, hoặc có bố cục xấu hoặc ta có thể xếp hạng kết quả cho người dùng, ưu tiên hiển thị những ảnh có điểm NIMA cao nhất trước.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Tương quan tốt với cảm nhận của con người:** Gần gũi với cách con người đánh giá một bức ảnh hơn các chỉ số kỹ thuật thuần túy.
- **Tự động và nhanh chóng:** Cho phép đánh giá hàng loạt mà không cần sự can thiệp của con người.
- **Linh hoạt:** Có thể được sử dụng trong quá trình huấn luyện và dự đoán của mô hình.

**Nhược điểm:**
- **Thiên kiến (Bias):** NIMA được huấn luyện trên một bộ dữ liệu cụ thể, do đó, nó có thể có "thiên kiến" và chấm điểm cao hơn cho những loại ảnh theo phong cách mà bộ dữ liệu này cho là đẹp, trong khi có thể chấm điểm thấp hơn cho các phong cách khác.

#### 3.6.3. ArcFace Score

ArcFace không được tạo ra cho bài toán sinh ảnh mà là một công nghệ đột phá trong lĩnh vực nhận dạng khuôn mặt (Face Recognition) được giới thiệu trong bài báo [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698).

Mục tiêu của ArcFace là tạo ra một vector đặc trưng (feature embedding) cho mỗi khuôn mặt.
Vector này có đặc điểm rất đặc biệt:
- Các khuôn mặt của cùng một người sẽ có vector đặc trưng rất gần nhau trong không gian vector.
- Các khuôn mặt của những người khác nhau sẽ có vector đặc trưng rất xa nhau.

Khi có hai vector đặc trưng của hai khuôn mặt, chúng ta có thể tính cosine similarity giữa chúng, nằm trong khoảng từ -1 đến 1 (1 khi hai vector hoàn toàn tương đồng, 0 khi hai vector hoàn toàn không tương đồng, -1 khi hai vector ngược chiều nhau), chính là ArcFace Score.
- ArcFace Score gần 1: Hai khuôn mặt gần như chắc chắn là của cùng một người.
- ArcFace Score gần 0 hoặc âm: Hai khuôn mặt là của hai người khác nhau.

Dựa vào ArcFace Score, ta có thể đo lường mức độ tương đồng về nhận dạng giữa hai khuôn mặt.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/arcface_score.jpeg" style="width: 800px;"/>

Chính vì khả năng "đo lường nhận dạng" xuất sắc này, các nhà nghiên cứu đã mượn ArcFace để giải quyết một trong những thách thức lớn nhất của việc sinh ảnh khuôn mặt: **Làm sao để ảnh được tạo ra trông giống hệt một người cụ thể?**

ArcFace score rất hữu ích trong việc sử dụng image generation trong các ứng dụng sinh ảnh mới từ một gương mặt cho trước.
ArcFace score cung cấp một chỉ số định lượng và khách quan, thay vì chỉ dựa vào cảm nhận của mắt người, chúng ta có thể tính ArcFace score giữa mỗi ảnh được tạo ra và ảnh gốc.
Ảnh nào có score cao nhất thì được xem là ảnh bảo toàn nhận dạng tốt nhất.
Điều này rất hữu ích trong việc tự động lọc và chọn lựa kết quả.

#### Cách thức hoạt động

- **Tạo vector tham chiếu:** Một hoặc một nhóm ảnh gốc của người A được đưa qua mô hình ArcFace đã được huấn luyện sẵn để trích xuất vector đặc trưng.
Nếu ta có một nhóm ảnh gốc của người A, ta có thể lấy trung bình các vector đặc trưng để tạo ra vector tham chiếu.
- **Bắt đầu sinh ảnh:** Sử dụng mô hình image generation để sinh ra loạt ảnh mới của người A đó.
- **Trích xuất vector và so sánh:** Các ảnh mới được sinh ra của người A sẽ được đưa qua mô hình ArcFace để trích xuất vector đặc trưng.
Ta lấy các vector đặc trưng này tính toán cosine similarity với vector tham chiếu ở Bước 1 để ra được ArcFace Score cho mỗi ảnh mới được sinh.
- **Lọc các ảnh không đạt yêu cầu:** Ta cần phân tích để chọn ra một ngưỡng ArcFace Score Threshold phù hợp để lọc ra những ảnh mới sinh "giống" nhất với những ảnh gốc của người A.

#### Ưu và nhược điểm

**Ưu điểm:**
- **Độ chính xác cao:** ArcFace là một trong những phương pháp nhận dạng khuôn mặt hàng đầu, do đó nó đảm bảo việc bảo toàn nhận dạng rất tốt.
- **Bền vững với thay đổi:** Nó có khả năng nhận ra một người ngay cả khi có sự thay đổi về góc mặt, ánh sáng, biểu cảm.
- **Định lượng được**: Cung cấp một con số cụ thể để đo lường, giúp tự động hóa việc đánh giá và tối ưu.

**Nhược điểm:**
- **Bộ ảnh tham chiếu rõ ràng:** Ta cần chuẩn bị bộ ảnh tham chiếu rõ ràng với từng người. Một bộ ảnh tham chiếu bị che khuất hay góc mặt quay sẽ khiến cho kết quả đánh giá ArcFace Score bị sai lệch rất đáng kể.
- **Không hiểu về ngữ cảnh:** ArcFace chỉ quan tâm đến nhận dạng và không hiểu các yêu cầu khác như "vẽ theo phong cách Picasso". Điều này đôi khi tạo ra sự xung đột giữa việc "giống người thật" và "giống phong cách nghệ thuật".
- **Nhạy cảm với ArcFace Score Threshold:** Để chọn ra được những tấm ảnh giống với chủ thể nhất, ta cần chọn ArcFace Score Threshold phù hợp với mỗi chủ thể của hình ảnh.

## 4. Các thách thức của bài toán image generation

Cũng giống như các bài toán khác trong lĩnh vực liên quan đến Generative AI nói chung, image generation cũng gặp phải nhiều thách thức và khó khăn trong quá trình triển khai hiện tại và tương lai.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/5_computer_vision/images/8-image-generation/challenges.jpeg" style="width: 800px;"/>

- **Thiên lệch (Bias):** Dữ liệu huấn luyện thường không đa dạng (VD: gương mặt đa số là người da trắng ở nhiều dataset).
Mô hình học theo và có thể tái tạo bias.
- **Nội dung độc hại (Safety):** Các mô hình sinh ảnh có thể tạo ra hình ảnh phản cảm, bạo lực hoặc giả mạo thông tin.
- **Giả mạo thông tin:** Mô hình image generation có thể tạo video, ảnh giả mạo gây hại.
- **Bản quyền:** Các mô hình học từ bộ dữ liệu lớn có thể vi phạm bản quyền ảnh nghệ sĩ.
- **Chi phí và tác động môi trường:** Huấn luyện và sử dụng mô hình image generation lớn tiêu tốn tài nguyên nhiều.
- **Khả năng giải thích:** Để kiểm tra, kiểm soát kết quả của mô hình image generation, cần nghiên cứu về cách giải thích logic, nguồn gốc từng thành phần.
- **Đánh giá chất lượng:** Cần thoát khỏi sự phụ thuộc vào đánh giá của con người, xây dựng các hệ thống tự động đánh giá.
