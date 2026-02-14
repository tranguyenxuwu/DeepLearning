## BTVN 02

Đây là bài tập thí nghiệm với **Mạng Nơ-ron Nhân tạo (ANN)**. Mục tiêu là xoay vòng thay đổi cấu trúc mạng, hàm mất mát, và bộ tối ưu để quan sát ảnh hưởng đến hiệu suất.

Dữ liệu gốc từ lab ANN (`./4/ANN-LAB.ipynb`): 300 điểm chia thành vòng tròn (lớp 0) và vành đai (lớp 1), 80% train / 20% test.

### Phần 1: Thay đổi cấu trúc ANN

Giữ nguyên mọi thứ (Adam, BCELoss, 100 epochs), chỉ thay đổi kiến trúc mạng:

1. **Baseline (2-4-1):**
    * Đầu vào 2 nút → Ẩn 4 nút (ReLU) → Đầu ra 1 nút (Sigmoid).
    * Đây là mô hình gốc từ lab. Ghi lại loss và accuracy làm mốc so sánh.

2. **Tăng nút (2-8-1):**
    * Tăng lớp ẩn từ 4 → 8 nút. 
    * *Kỳ vọng:* Loss thấp hơn, accuracy cao hơn vì mạng có khả năng biểu diễn mạnh hơn.

3. **Thêm lớp (2-8-6-1):**
    * Thêm một lớp ẩn thứ 2 gồm 6 nút (ReLU). Mạng bây giờ sâu hơn.
    * *Kỳ vọng:* Học được đặc trưng phức tạp hơn, nhưng cẩn thận overfitting nếu dữ liệu ít.

### Phần 2: Thử nghiệm hàm mất mát & tối ưu hóa

Quay lại cấu trúc gốc 2-4-1, thay đổi lần lượt loss function và optimizer:

1. **BCEWithLogitsLoss thay BCELoss:**
    * `BCEWithLogitsLoss` = `Sigmoid` + `BCELoss` gộp lại trong 1 bước tính → ổn định số học hơn.
    * *Lưu ý:* Phải xóa Sigmoid khỏi lớp đầu ra vì loss đã tự xử lý.
    * *Kết quả:* Về lý thuyết ra giống BCELoss, nhưng tránh được lỗi tràn số.

2. **SGD thay Adam:**
    * SGD = "đi bộ đều đặn", Adam = "GPS tìm đường ngắn nhất".
    * SGD với `lr=0.01` cố định → hội tụ chậm hơn Adam, loss có thể dao động vì không tự điều chỉnh bước đi.
    * *Kết quả:* Cùng 100 epochs, SGD thường chưa tối ưu bằng Adam.

### Phần 3: Phân tích kết quả

Vẽ đồ thị loss theo epoch cho 3 trường hợp trên cùng 1 biểu đồ bằng `matplotlib`:

1. **Vẽ đồ thị:**
    * 3 đường: Baseline (2-4-1 Adam), Tăng nút (2-8-1 Adam), SGD (2-4-1 SGD).
    * Thêm `legend` để phân biệt.

2. **Nhận xét:**
    * Nhanh nhất: 2-8-1 + Adam → adaptive learning rate + nhiều tham số.
    * Chậm nhất: 2-4-1 + SGD → learning rate cố định, không có momentum.
    * Dao động: SGD có thể nhảy lên xuống vì bước cập nhật quá lớn ở một số thời điểm. Adam mượt hơn nhờ cơ chế moment.
