## BTVN 01

Đây là bài toán **Mô phỏng Logic game cờ caro 3x4**. Thay vì chỉ nhập xuất dữ liệu đơn thuần, ta xây dựng một trọng tài ảo để quản lý luật chơi.

Quy trình hoạt động như sau:

1. **Khởi tạo sân đấu:**
* Tạo ma trận 3x4 toàn số `99`. Tưởng tượng `99` là ô trống, chưa ai ngồi.
* `x` và `o` là hai phe tranh giành các ô này.

2. **Vòng lặp Game:**
* **Luân phiên:** Dùng biến đếm hoặc cờ để xác định lượt này là của `x` hay `o`.
* **Nhập liệu:** Nhận tọa độ `(hàng, cột)` từ người chơi.

3. **Trọng tài bắt lỗi:**
* Trước khi điền vào ma trận, máy kiểm tra 2 điều:
* *Có nằm trong sân không?* (Không thể đi ra ngoài ma trận).
* *Ghế có trống không?* (Nếu ô đó khác `99`, tức là đã có người ngồi  Bắt nhập lại).

4. **Quét tìm chiến thắng:**
* Sau mỗi nước đi, máy thực hiện quét ma trận theo 4 hướng: **Ngang, Dọc, Chéo chính, Chéo phụ**.
* Nếu phát hiện **3 phần tử liên tiếp** giống nhau của cùng một phe  Dừng game, tuyên bố thắng cuộc.

## BTVN 02

Bài này minh họa kỹ thuật **Truy xuất dữ liệu** trong mảng 2 chiều. Để cắt lấy phần dữ liệu cần thiết.

Quy trình hoạt động như sau:

1. **Lấy nguyên hàng:**
* Yêu cầu: Lấy `4, 5, 6`.
* Thực hiện: `y[1]`. Máy tính chỉ cần nhảy vào hàng có chỉ số 1 và bưng nguyên cả mảng đó ra.

2. **Lấy điểm lẻ:**
* Yêu cầu: Lấy `2, 5`.
* Thực hiện: Truy cập tọa độ cụ thể `y[0][1]` và `y[1][1]`. Đây là cách "nhặt hạt gạo", lấy từng phần tử rời rạc.

3. **Lấy ngược:**
* Yêu cầu: Lấy `9, 6, 3` (Cột cuối, từ dưới lên).
* Thực hiện:
* Cách thủ công: Chỉ định rõ tọa độ `(2,2)`, `(1,2)`, `(0,2)`.
* Cách tư duy mảng: Lấy cột cuối, sau đó đảo ngược thứ tự.

## BTVN 03

Bài này so sánh hai phong cách code trong Python: **Cổ điển (For + If)** vs **List Comprehension**.

Quy trình hoạt động như sau:

### 1. Vòng lặp For + If

* **Cơ chế:**
* Tạo một cái rổ rỗng.
* Cầm từng số lên, soi xem có chia hết cho 2 không (`if i % 2 == 0`).
* Nếu đúng, bỏ vào rổ (`append`).

### 2. List Comprehension

* **Code:** `[i for i in x if i % 2 == 0]`
* **Cơ chế:** Gom tất cả quy trình: Tạo rổ, lặp, kiểm tra, bỏ vào rổ... gói gọn trong đúng **1 dòng lệnh**.

## BTVN 04

Đây là quy trình chuẩn biến đổi dữ liệu thô thành dữ liệu sẵn sàng để đưa vào training.

Quy trình hoạt động như sau:

1. **Splitting Columns:**
* Tách dữ liệu thành 2 phần:
* **X (Features):** Chiều cao, cân nặng, lương... (Dữ liệu đầu vào để máy học).
* **Y (Target):** Điểm TBM (Đáp án máy cần dự đoán).

2. **Train/Test Split:**
* **Train (70%):** Dùng để dạy học. Máy tính sẽ nhìn vào đây để tìm quy luật.
* **Test (30%):** Dùng để đi thi. Dữ liệu này được giấu kín, chỉ lôi ra để chấm điểm xem máy học tốt đến đâu.
* *Lưu ý:* Không bao giờ được để máy nhìn thấy `Test` trong lúc học.


3. **Numpy to PyTorch:**
* Code: `torch.tensor(X_train)`
* **Lý do:** Numpy xử lý trên CPU rất tốt, nhưng PyTorch Tensor để chạy trên GPU và tính đạo hàm tự động.

4. **K-Fold/Batching:**
* Yêu cầu: 10 tập dữ liệu không chồng chéo.
* **Cơ chế:** Cắt cái bánh 150 phần tử thành 10 miếng, mỗi miếng 15 phần tử.
* **Mục đích:** Để thực hiện kỹ thuật Cross-Validation, đảm bảo mô hình không "học vẹt" trên một tập dữ liệu cố định.