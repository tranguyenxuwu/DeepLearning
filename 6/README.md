# BÀI TẬP THỰC HÀNH CNN - MNIST

## Mục tiêu bài tập

Bài thực hành này dùng mạng tích chập CNN (Convolutional Neural Network) để nhận dạng chữ số viết tay từ bộ dữ liệu MNIST. 

## Dữ liệu sử dụng

- Bộ dữ liệu: MNIST
- Tập train: 60,000 ảnh chữ số viết tay
- Tập test: 10,000 ảnh
- Kích thước mỗi ảnh: 28x28 pixel, ảnh xám
- Số lớp phân loại: 10 lớp, tương ứng các chữ số từ 0 đến 9

Trong notebook, dữ liệu được nạp bằng `torchvision.datasets.MNIST`, chuyển sang tensor bằng `transforms.ToTensor()`, rồi đưa vào `DataLoader` với `batch_size=64`.

## Mô hình CNN gốc

Mô hình cơ sở gồm các thành phần sau:

1. Tầng tích chập `conv1`: đầu vào 1 kênh, đầu ra 16 kênh
2. Tầng tích chập `conv2`: đầu vào 16 kênh, đầu ra 32 kênh
3. Hai tầng `MaxPool2d` để giảm kích thước đặc trưng
4. Một tầng fully connected để đưa ra 10 giá trị dự đoán cuối cùng

Luồng xử lý của mô hình:

`Input image -> Conv1 + ReLU -> MaxPool -> Conv2 + ReLU -> MaxPool -> Flatten -> Fully Connected -> Output`

Mô hình dùng:

- Hàm mất mát: `CrossEntropyLoss`
- Bộ tối ưu: `SGD` với `momentum=0.9`
- Learning rate mặc định: `0.01`
- Số epoch mặc định trong các thí nghiệm cơ bản: `5`

### Câu 1: Thay đổi số lượng epoch

Yêu cầu của câu này là tăng số epoch từ 5 lên 10 và chạy lại mô hình CNN gốc.

Ý nghĩa:

- Epoch nhiều hơn giúp mô hình có thêm cơ hội học từ dữ liệu
- Loss thường giảm thêm trong các epoch đầu
- Accuracy có thể tăng, nhưng khi mô hình gần hội tụ thì mức cải thiện sẽ nhỏ dần
- Nếu train quá lâu, mô hình có thể gặp hiện tượng overfitting

Kỳ vọng khi quan sát kết quả:

- Đường loss giảm đều rồi chững lại
- Accuracy train tăng dần qua từng epoch
- Accuracy trên tập test tăng nhẹ hoặc gần bão hòa so với cấu hình 5 epoch

### Câu 2: Thêm một tầng tích chập

Ở câu này, mô hình được mở rộng bằng cách thêm tầng `conv3` với:

- Đầu vào: 32 kênh
- Đầu ra: 64 kênh

Ý nghĩa:

- Tầng tích chập sâu hơn giúp mô hình học được đặc trưng phức tạp hơn
- Các lớp đầu thường học cạnh, đường viền, độ tương phản
- Các lớp sâu hơn có xu hướng học các mẫu hình trừu tượng hơn

Trong bài MNIST, dữ liệu tương đối đơn giản nên việc thêm tầng chưa chắc cải thiện mạnh accuracy, nhưng giúp minh họa rõ vai trò của chiều sâu mô hình trong CNN.

### Câu 3: Thay đổi learning rate

Notebook thử hai giá trị learning rate khác nhau:

- `0.001`
- `0.1`

Ý nghĩa:

- Learning rate quá nhỏ làm mô hình học chậm, loss giảm chậm, cần nhiều epoch hơn để đạt kết quả tốt
- Learning rate quá lớn khiến bước cập nhật trọng số mạnh, dễ dao động hoặc vượt qua điểm tối ưu
- Learning rate phù hợp giúp mô hình hội tụ nhanh và ổn định

- Với `0.001`: đồ thị loss mượt hơn nhưng cải thiện chậm
- Với `0.1`: loss có thể dao động mạnh hơn, accuracy không ổn định
- Với `0.01`: thường là mức cân bằng tốt trong bài thực hành này

### Câu 4: Trực quan hóa feature map

Phần này dùng một ảnh trong tập test để xem các feature map sinh ra từ:

- Tầng tích chập thứ nhất `conv1`
- Tầng tích chập thứ hai `conv2`

Ý nghĩa:

- Feature map của `conv1` thường còn giữ nhiều thông tin hình dạng ban đầu
- Feature map của `conv2` trừu tượng hơn, tập trung vào các đặc trưng có ích cho phân loại
- Việc trực quan hóa giúp hiểu CNN không chỉ đưa ra dự đoán, mà còn học cách biểu diễn ảnh qua từng tầng

## Hàm huấn luyện dùng chung

- Huấn luyện mô hình theo số epoch chỉ định
- Ghi nhận `loss` và `accuracy` theo từng epoch
- Đánh giá trên tập test sau khi train xong
- Vẽ biểu đồ loss và accuracy để tiện so sánh

## Kết luận 

1. CNN phù hợp với dữ liệu ảnh hơn ANN vì khai thác được cấu trúc không gian của ảnh
2. Số epoch ảnh hưởng trực tiếp đến mức độ hội tụ của mô hình
3. Learning rate là siêu tham số quan trọng, quyết định tốc độ và độ ổn định khi học
4. Tăng số tầng tích chập giúp mô hình học đặc trưng sâu hơn, nhưng hiệu quả còn phụ thuộc độ phức tạp của dữ liệu
5. Feature map cho thấy CNN học các biểu diễn trung gian có ý nghĩa trước khi đưa ra dự đoán cuối cùng
