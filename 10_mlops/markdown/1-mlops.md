---
time: 02/19/2023
title:
description:
banner_url:
tags: [deep-learning, machine-learning, mlops]
is_highlight: false
is_published: false
---

# Machine learning operations (MLOps)

## 1. Một số khái niệm cơ bản

### 1.1. Khái niệm về Data Drift và Concept Drift

Data Drift (biến đổi dữ liệu) và Concept Drift (biến đổi khái niệm) là hai khái niệm quan trọng trong Machine Learning và quản lý mô hình.

Chúng liên quan đến sự thay đổi của dữ liệu và mô hình trong thời gian, và chúng đề cập đến những thách thức mà bạn cần phải đối mặt khi triển khai các hệ thống học máy trong môi trường thực tế.

<img src="" style="width: 1200px;"/>

#### 1.1.1. Data Drift (Biến đổi dữ liệu):

Data Drift là hiện tượng khi phân phối của dữ liệu đầu vào cho mô hình thay đổi theo thời gian, do nhiều nguyên nhân như:
- Sự thay đổi trong nguồn dữ liệu
- Quy trình thu thập dữ liệu
- Môi trường thay đổi
- Sự thay đổi của người dùng

Data Drift có thể gây ra việc mô hình trở nên kém hiệu suất vì nó đã được đào tạo trên một bộ dữ liệu cũ mà không còn phản ánh đúng thực tế hiện tại.

Ví dụ: Bài toán dự đoán giá bất động sản trong một khu vực nhất định

Mô hình sử dụng dữ liệu về diện tích, vị trí địa lý, số phòng, giá trị trung bình của khu vực, và dữ liệu lịch sử giá bất động sản trong khu vực đó.

Thời gian đầu, mô hình tốt và có độ chính xác cao, sau một thời gian, hiệu suất của mô hình giảm đi đáng kể và các dự đoán không còn chính xác.

Nguyên nhân của sự giảm hiệu suất này có thể là một ví dụ về Data Drift.
Dữ liệu mà bạn đã sử dụng để đào tạo mô hình ban đầu không còn phản ánh đúng thực tế hiện tại do:
- Thay đổi thị trường: Giá bất động sản thay đổi theo thời gian dựa trên tình hình kinh tế, chính trị, hoặc sự thay đổi trong cầu và cung. Nếu mô hình của bạn không được cập nhật với dữ liệu mới nhất, nó sẽ không thể dự đoán các biến động này.
- Thay đổi trong dữ liệu đầu vào: Nếu các nguồn dữ liệu mà bạn sử dụng thay đổi cách thu thập hoặc cung cấp dữ liệu, các biến số như diện tích, số phòng, hoặc vị trí có thể bị ảnh hưởng.
- Thay đổi trong môi trường: Nếu môi trường của bạn thay đổi, ví dụ: nếu bạn đang dự đoán giá bất động sản trong một khu vực nhất định, nhưng sau đó bạn mở rộng phạm vi của mình để dự đoán giá bất động sản trong nhiều khu vực hơn.
- Thay đổi trong người dùng: Nếu người dùng của bạn thay đổi, ví dụ: nếu bạn đang dự đoán giá bất động sản cho các nhà đầu tư, nhưng sau đó bạn mở rộng phạm vi của mình để dự đoán giá bất động sản cho người mua sử dụng.

Để kiểm soát Data Drift, bạn cần thường xuyên theo dõi và cập nhật mô hình của mình bằng cách sử dụng dữ liệu mới và đảm bảo rằng mô hình luôn phản ánh được sự thay đổi trong dữ liệu.

#### 1.1.2. Concept Drift (Biến đổi khái niệm):

Concept Drift là sự thay đổi trong mối quan hệ giữa đầu vào và đầu ra của mô hình theo thời gian.
Nó xảy ra khi mô hình ban đầu đã học một khái niệm hoặc quy tắc nào đó từ dữ liệu huấn luyện, nhưng sau đó, quy tắc này không còn cùng áp dụng trong thực tế.

Ví dụ, trong một hệ thống dự đoán giá cổ phiếu, mô hình có thể học cách dự đoán dựa trên dữ liệu lịch sử, nhưng khái niệm này có thể thay đổi khi có tin tức hoặc sự kiện đột ngột ảnh hưởng đến thị trường.
Để kiểm soát Concept Drift, bạn cần thường xuyên theo dõi hiệu suất của mô hình và cân nhắc cách điều chỉnh hoặc cập nhật mô hình để nó có thể thích nghi với sự thay đổi trong khái niệm hoặc quy tắc.
Cả Data Drift và Concept Drift đều đánh dấu sự thay đổi liên quan đến dữ liệu và mô hình, và quản lý chúng là một phần quan trọng của quy trình MLOps để đảm bảo tính nhất quán và hiệu suất của hệ thống học máy theo thời gian.

#### 1.1.3. Một số phương pháp giải quyết Data Drift và Concept Drift:

- Active Learning:
Active Learning là một phương pháp học máy trong đó mô hình được đào tạo để có khả năng "hỏi" cho người giám sát hoặc hệ thống về các điểm dữ liệu cụ thể để cải thiện hiệu suất học tập.
Thay vì đào tạo mô hình trên toàn bộ tập dữ liệu, Active Learning chọn ra những mẫu dữ liệu có giá trị thông tin cao để học và điều này có thể giảm đáng kể lượng dữ liệu cần thiết để đạt được hiệu suất tốt.
- Transfer Learning:
Transfer Learning là một phương pháp trong học máy mà một mô hình đã được đào tạo trước đó trên một tác vụ liên quan được sử dụng lại để cải thiện hiệu suất trên một tác vụ mới hoặc tương tự.
Thay vì bắt đầu từ mô hình rỗng, Transfer Learning sử dụng kiến thức đã học từ các tác vụ trước đó và điều chỉnh nó cho tác vụ hiện tại. Điều này có thể giúp giảm thời gian và tài nguyên đào tạo.
- Online Learning:
Online Learning là một phương pháp học máy mà mô hình được đào tạo liên tục khi có dữ liệu mới được cung cấp một cách tuần tự.
Thay vì đào tạo mô hình một lần duy nhất và sử dụng nó cho dự đoán, Online Learning cho phép mô hình cập nhật thông tin liên tục khi dữ liệu mới đến. Điều này thích hợp cho các tình huống mà dữ liệu thay đổi liên tục, chẳng hạn như theo dõi các dòng tin tức trực tuyến.
- Incremental Learning:
Incremental Learning là một phương pháp học máy mà mô hình được đào tạo một lần và sau đó tiếp tục học và cải tiến khi có dữ liệu mới được cung cấp.
Tương tự như Online Learning, Incremental Learning cho phép mô hình cập nhật khi có dữ liệu mới, nhưng nó có thể được sử dụng trong các tình huống mà dữ liệu mới không được cung cấp liên tục mà có thể đến từ những pha chu kỳ.


### 1.2. Khái niệm về Model-Centric và Data-Centric

Data-Centric và Model-Centric là hai phương pháp tiếp cận khác nhau trong lĩnh vực Machine Learning và Deep Learning, với sự tập trung vào các khía cạnh quan trọng khác nhau của quá trình phát triển mô hình.

<img src="https://images.viblo.asia/855cadf0-33bd-4e69-9eae-ffccaf38372a.jpeg" style="width: 1200px;"/>

#### 1.2.1. Data-Centric (Tập trung vào dữ liệu):

- Tập trung chủ yếu vào dữ liệu:
Tiếp cận data-centric đặt dữ liệu làm trung tâm của quá trình phát triển mô hình.
Nó coi dữ liệu là yếu tố quan trọng nhất để tạo ra một mô hình học máy hiệu quả.
- Dữ liệu chất lượng cao:
Data-centric giả định rằng dữ liệu phải được thu thập, xử lý, và làm sạch một cách cẩn thận trước khi áp dụng vào mô hình.
Điều này bao gồm việc loại bỏ nhiễu, điền giá trị thiếu, và chuẩn hóa dữ liệu.
- Đánh giá và kiểm tra dữ liệu:
Tiếp cận này thường đòi hỏi phải dành nhiều thời gian để kiểm tra và đánh giá dữ liệu, đảm bảo tính nhất quán và đáng tin cậy.
- Mô hình thụ động hơn:
Trong tiếp cận data-centric, mô hình thường được xây dựng dựa trên dữ liệu và có thể thay đổi linh hoạt theo thời gian tùy thuộc vào dữ liệu mới.

#### 1.2.2. Model-Centric (Tập trung vào mô hình):

- Tập trung chủ yếu vào kiến trúc mô hình:
Tiếp cận model-centric coi mô hình là yếu tố quan trọng nhất và tập trung vào việc thiết kế, tối ưu hóa và triển khai mô hình.
- Mô hình phức tạp:
Trong model-centric, người ta thường tạo ra các mô hình phức tạp và mạnh mẽ để giải quyết vấn đề..
- Thiết kế và tinh chỉnh mô hình:
Tiếp cận này tập trung vào việc lựa chọn và tinh chỉnh các tham số của mô hình để đạt được hiệu suất tốt nhất trên dữ liệu đầu vào.
- Dữ liệu đôi khi được xem như là đầu vào tiêu chuẩn:
Trong model-centric, dữ liệu có thể được xem như một phần của quá trình, và dữ liệu chất lượng thấp hoặc không hoàn hảo có thể được "sửa" bằng cách sử dụng các mô hình phức tạp để xử lý nhiễu hoặc dự đoán các giá trị bị thiếu.

## 2. Giới thiệu chung về MLOps

### 2.1. Lý do ra đời của MLOps

Việc triển khai các dự án Machine Learning không phải lúc nào cũng dễ dàng.
Các nhà phát triển và nhà nghiên cứu thường phải đối mặt với những thách thức sau:
- Quản lý dữ liệu:
Dữ liệu là yếu tố cốt lõi trong Machine Learning, và việc quản lý, tiền xử lý và tổ chức dữ liệu là một thách thức lớn.
- Đào tạo và đánh giá mô hình:
Đào tạo một mô hình thường đòi hỏi nhiều lần thử nghiệm và hiệu chỉnh.
Điều này đặt ra câu hỏi về việc theo dõi và quản lý các phiên bản mô hình.
- Triển khai và duy trì:
Đưa mô hình từ môi trường phát triển sang môi trường sản phẩm có thể gặp nhiều khó khăn.
Đồng thời, duy trì và cập nhật mô hình trong thời gian thực cũng đòi hỏi quy trình tự động hóa.

<img src="https://www.compact.nl/wordpress/wp-content/uploads/2022/10/C-2022-3-Maliutin-2t-groot.jpeg" style="width: 1200px;"/>

### 2.2. Khái niệm

MLOps viết tắt của Machine Learning Operations.
MLOps là việc tích hợp các mô hình học máy vào quy trình phát triển phần mềm của tổ chức.
MLOps sử dụng một tập hợp các bước để đảm bảo tính đáng tin cậy của mô hình học máy.

Nói cách khác, MLOps là một tập hợp các phương pháp, quy trình và công cụ để tự động hóa và quản lý toàn bộ chu trình phát triển, triển khai và duy trì mô hình Machine Learning.
MLOps kết hợp các phương pháp đã biết từ DevOps (quản lý phát triển và vận hành ứng dụng) và áp dụng chúng vào lĩnh vực Machine Learning.

MLOps đòi hỏi sự hợp tác giữa các nhà khoa học dữ liệu và các lập trình viên, người xây dựng và đào tạo mô hình, cùng với các chuyên gia IT, người xử lý hạ tầng và triển khai các mô hình.

<img src="https://www.ml4devs.com/images/illustrations/ml-lifecycle-fusing-model-and-software-development.webp" style="width: 1200px;"/>

### 2.3. Lợi ích chung mà MLOps mang lại

MLOps mang lại nhiều lợi ích quan trọng, bao gồm:

- Tăng tốc độ triển khai:
MLOps giúp tự động hóa quy trình triển khai mô hình, giúp tăng tốc độ ra thị trường và cung cấp giá trị cho doanh nghiệp nhanh hơn.
- Tăng tính nhất quán:
Bằng cách quản lý tất cả phiên bản mô hình và dữ liệu, MLOps giúp đảm bảo tính nhất quán và đáng tin cậy của hệ thống.
- Tối ưu hóa tài nguyên:
MLOps cung cấp khả năng theo dõi và tối ưu hóa việc sử dụng tài nguyên máy tính, giúp tiết kiệm chi phí.

### 2.4. Các nền tảng MLOps

Các nền tảng mã nguồn mở:
- Kubeflow:
Kubeflow là một nền tảng mã nguồn mở được xây dựng trên Kubernetes.
- MLflow:
MLflow là một nền tảng quản lý mô hình mã nguồn mở từ Databricks
- Apache Airflow:
Apache Airflow là một hệ thống lập lịch và quản lý luồng làm việc mã nguồn mở từ Apache.
- ClearML:
ClearML là một nền tảng mã nguồn mở từ Allegro AI.

Các nền tảng của các nhà cung cấp dịch vụ đám mây:
- Azure Machine Learning:
Đây là một nền tảng được cung cấp bởi Microsoft Azure.
- Amazon SageMaker:
Đây là một dịch vụ chuyên về Machine Learning của AWS.
- Google Cloud AI Platform:
Đây là một dịch vụ chuyên về Machine Learning của Google Cloud Platform.

<img src="https://learn.microsoft.com/en-us/azure/architecture/ai-ml/idea/_images/orchestrate-mlops-azure-databricks-01.jpeg" style="width: 1200px;"/>

## 3. Các thành phần trong kiến trúc của MLOps

### 3.1. Data Ingestion (Nhập dữ liệu):
Thành phần này liên quan đến việc thu thập, xử lý và chuẩn bị dữ liệu cho quy trình huấn luyện mô hình.

Nó bao gồm các công việc như:
- Thu thập dữ liệu (Data Collection):
Dữ liệu có thể được thu thập từ cơ sở dữ liệu, tệp tin, dịch vụ web, máy chủ, hoặc bất kỳ nguồn nào có sẵn. Việc này có thể thực hiện tự động hoặc thủ công tùy thuộc vào nguồn dữ liệu cụ thể.
- Trích xuất dữ liệu (Data Extraction):
Sau khi dữ liệu được thu thập, bạn cần trích xuất thông tin cần thiết từ nguồn dữ liệu gốc.
Trích xuất này có thể bao gồm việc lấy ra các cột hoặc thuộc tính cụ thể, lọc dữ liệu không cần thiết, và xử lý dữ liệu đầu vào.
- Biến đổi dữ liệu (Data Transformation):
Dữ liệu thường cần được biến đổi để phù hợp với quy trình đào tạo mô hình.
Các biến đổi này có thể bao gồm chuyển đổi dữ liệu số học, mã hóa dữ liệu phân loại, chuẩn hóa dữ liệu, và thậm chí là tạo ra các đặc trưng mới dựa trên dữ liệu hiện có.
- Làm sạch dữ liệu (Data Cleaning):
Dữ liệu thường chứa nhiễu hoặc giá trị thiếu.
Trong bước này, bạn cần làm sạch dữ liệu bằng cách xử lý giá trị thiếu (điền giá trị hoặc loại bỏ dòng dữ liệu), loại bỏ dữ liệu nhiễu, và kiểm tra tính nhất quán của dữ liệu.
- Lưu trữ dữ liệu (Data Storage):
Dữ liệu sau khi được làm sạch và biến đổi cần được lưu trữ ở một vị trí dễ truy cập.
Các hệ thống lưu trữ dữ liệu phổ biến bao gồm cơ sở dữ liệu SQL, NoSQL, hệ thống tệp tin, lưu trữ đám mây như Amazon S3 hoặc Google Cloud Storage.
- Quản lý phiên bản dữ liệu (Data Versioning):
Để theo dõi và quản lý sự thay đổi trong dữ liệu, quá trình nhập dữ liệu cần phải bao gồm quản lý phiên bản.
Điều này đảm bảo tính nhất quán và có thể tái tái sử dụng dữ liệu trong tương lai.
- Tự động hóa quy trình (Automation):
Việc nhập dữ liệu thường được tự động hóa để đảm bảo tính liên tục của quy trình.
Các công cụ và luồng làm việc tự động có thể được sử dụng để định thời gian và tự động hóa quá trình nhập dữ liệu.
- Kiểm tra và xác thực (Testing and Validation):
Trước khi sử dụng dữ liệu cho việc đào tạo mô hình, nó cần phải được kiểm tra và xác thực để đảm bảo tính chính xác và đáng tin cậy.
- Xác thực an toàn và quyền truy cập (Security and Access Control): Dữ liệu là tài sản quý giá và cần phải được bảo vệ. Hệ thống nhập dữ liệu cần xác thực an toàn và quản lý quyền truy cập để đảm bảo tính riêng tư và bảo mật của dữ liệu.

<img src="https://adataanalyst.com/wp-content/uploads/2021/05/Infra-Tooling3.jpeg" style="width: 1200px;"/>

### 3.2. Model Training (Đào tạo mô hình):

Phần này là nơi mô hình Machine Learning được xây dựng và đào tạo trên dữ liệu đã chuẩn bị. Nó liên quan đến việc lựa chọn thuật toán, tối ưu hóa mô hình và việc tạo ra phiên bản cuối cùng của mô hình.

Nó bao gồm các công việc như:
- Lựa chọn thuật toán (Algorithm Selection):
Thuật toán là một phần quan trọng trong việc xây dựng mô hình Machine Learning.
Thuật toán phù hợp với bài toán sẽ giúp mô hình đạt được hiệu suất tốt nhất.
- Tối ưu hóa mô hình (Model Optimization):
Một mô hình Machine Learning có thể được tối ưu hóa để đạt được hiệu suất tốt hơn.
Các kỹ thuật tối ưu hóa mô hình bao gồm tinh chỉnh tham số, tinh chỉnh kiến trúc, và tinh chỉnh siêu tham số.
- Đào tạo mô hình (Model Training):
Mô hình được đào tạo trên dữ liệu đã được chuẩn bị để tạo ra một phiên bản cuối cùng của mô hình.
- Theo dõi thử nghiệm (Experiment Tracking):
Thành phần này cho phép lưu trữ và theo dõi thông tin về các thử nghiệm MLOps, bao gồm các phiên bản mô hình, tham số, và kết quả kiểm tra.
- Xác thực mô hình (Model Validation):
Sau khi đào tạo, mô hình cần được kiểm tra và đánh giá để đảm bảo tính hiệu quả và đáng tin cậy.
Các kỹ thuật xác thực mô hình bao gồm kiểm định chéo và đánh giá hiệu suất mô hình.
- Lưu trữ mô hình (Model Storage):
Mô hình sau khi được đào tạo và xác thực cần được lưu trữ ở một vị trí dễ truy cập.
Các hệ thống lưu trữ mô hình phổ biến bao gồm cơ sở dữ liệu SQL, NoSQL, hệ thống tệp tin, lưu trữ đám mây như Amazon S3 hoặc Google Cloud Storage.
- Quản lý phiên bản mô hình (Model Versioning):
Để theo dõi và quản lý sự thay đổi trong mô hình, quá trình đào tạo mô hình cần phải bao gồm quản lý phiên bản.
Điều này đảm bảo tính nhất quán và có thể tái sử dụng mô hình trong tương lai.
- Tự động hóa quy trình (Automation):
Việc đào tạo mô hình thường được tự động hóa để đảm bảo tính liên tục của quy trình.
Các công cụ và luồng làm việc tự động có thể được sử dụng để định thời gian và tự động hóa quá trình đào tạo mô hình.
- Kiểm tra và xác thực (Testing and Validation):
Trước khi sử dụng mô hình cho việc triển khai, nó cần phải được kiểm tra và xác thực để đảm bảo tính chính xác và đáng tin cậy.
- Xác thực an toàn và quyền truy cập (Security and Access Control):
Mô hình là tài sản quý giá và cần phải được bảo vệ.
Hệ thống đào tạo mô hình cần xác thực an toàn và quản lý quyền truy cập để đảm bảo tính riêng tư và bảo mật của mô hình.

### 3.3. Model Deployment (Triển khai mô hình):

Mô hình đã được đào tạo và xác thực được triển khai trên một môi trường sản xuất để phục vụ các ứng dụng thực tế.
Điều này có thể bao gồm việc triển khai mô hình trên một môi trường điện toán đám mây hoặc hạ tầng on-premises.

Nó bao gồm các công việc như:
- Triển khai mô hình (Model Deployment):
Mô hình được triển khai trên một môi trường sản xuất để phục vụ các ứng dụng thực tế.
- Quản lý phiên bản mô hình (Model Versioning):
Để theo dõi và quản lý sự thay đổi trong mô hình, quá trình triển khai mô hình cần phải bao gồm quản lý phiên bản.
Điều này đảm bảo tính nhất quán và có thể tái sử dụng mô hình trong tương lai.
- Tự động hóa quy trình (Automation):
Việc triển khai mô hình thường được tự động hóa để đảm bảo tính liên tục của quy trình.
Các công cụ và luồng làm việc tự động có thể được sử dụng để định thời gian và tự động hóa quá trình triển khai mô hình.
Quá trình này có thể được gọi là Continuous Integration/Continuous Deployment (CI/CD).
CI/CD là một quá trình tự động hóa cho phép triển khai các phiên bản mới của mô hình một cách nhanh chóng và đáng tin cậy. Nó bao gồm việc tích hợp, kiểm tra và triển khai tự động.

### 3.4. Model Monitoring (Theo dõi mô hình):

Để đảm bảo tính ổn định và hiệu suất liên tục của mô hình sau khi triển khai, cần có một hệ thống để theo dõi mô hình trong thời gian thực. Thành phần này bao gồm việc xác định dấu hiệu của sự thay đổi và cảnh báo khi cần thiết.

Nó bao gồm các công việc như:
- Theo dõi mô hình (Model Monitoring):
Mô hình cần được theo dõi để đảm bảo tính ổn định và hiệu suất liên tục.
- Xác định dấu hiệu (Detecting Signals):
Để theo dõi mô hình, bạn cần xác định các dấu hiệu của sự thay đổi.
Các dấu hiệu này có thể bao gồm Data Drift, Concept Drift, và các lỗi khác.
- Cảnh báo (Alerting):
Khi các dấu hiệu của sự thay đổi được phát hiện, hệ thống cần cảnh báo người dùng để họ có thể thực hiện các biện pháp cần thiết.
- Tự động hóa quy trình (Automation):
Việc theo dõi mô hình thường được tự động hóa để đảm bảo tính liên tục của quy trình.
Các công cụ và luồng làm việc tự động có thể được sử dụng để định thời gian và tự động hóa quá trình theo dõi mô hình.
- Vòng phản hồi (Feedback Loop):
Thành phần này liên quan đến việc thu thập thông tin từ việc theo dõi hiệu suất và áp dụng các cải tiến vào quá trình đào tạo và triển khai mô hình tiếp theo.

<img src="https://cloud.google.com/static/architecture/images/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-4-ml-automation-ci-cd.svg" style="width: 1200px;"/>
