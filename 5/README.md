## BTVN 02 - Phần Mở Rộng: MNIST

**Nhận dạng chữ số viết tay (MNIST)**.

Quy trình hoạt động như sau:

1. **Chuẩn bị Data Preparation:**
    *  Tải bộ dữ liệu MNIST về (`torchvision.datasets.MNIST`). Bộ dữ liệu gồm 60,000 ảnh train và 10,000 ảnh test các chữ số từ 0-9.
    *  Chuyển ảnh về dạng Tensor và chuẩn hóa (`Normalize`) để giúp mô hình "dễ học" hơn (đưa giá trị pixel về khoảng [-1, 1] thay vì [0, 255]).
    *  Dùng `DataLoader` để chia nhỏ dữ liệu thành các (`batch`), giúp máy không bị quá tải khi train.

2. **Xây dựng mô hình:**
    *   Thiết kế một mạng ANN đơn giản.
    *   Tưởng tượng việc trải phẳng một bức ảnh 2 chiều (28x28) thành một sợi dây dài ngoằng (784 pixel). Đây là bước đầu vào bắt buộc cho các mạng Linear truyền thống.
    *   **Lớp ẩn:** Nơi mạng "suy ngẫm". Số lượng nút (neurons) ở đây càng nhiều thì mạng càng "thông minh" (nhưng coi chừng học vẹt - overfitting).
    *   **Đầu ra:** 10 nút, mỗi nút đại diện cho xác suất của một con số (0, 1, ..., 9).

3. **Training:**
    *   Sử dụng hàm mất mát `CrossEntropyLoss` (bài toán phân loại nhiều lớp) và bộ tối ưu `Adam` (thông minh hơn, tự điều chỉnh bước đi).
    *   **Vòng lặp:**
        *   Cho mô hình xem ảnh -> Mô hình đoán -> Tính lỗi sai (`Loss`).
        *   Dựa vào lỗi sai, điều chỉnh lại các trọng số (`Weights`) để lần sau đoán chuẩn hơn với  Backpropagation.
    *   *Kỳ vọng:* Loss sẽ giảm dần theo thời gian, giống như việc bạn càng luyện tập thì càng ít mắc lỗi.

4. **Testing:**
    *   Đem mô hình đi thi trên tập dữ liệu Test (không có trong quá trình train).
    *   **Mục tiêu:** Độ chính xác (Accuracy) càng cao càng tốt. Với mạng ANN đơn giản này, chúng ta có thể dễ dàng đạt trên 95% sau vài epoch.