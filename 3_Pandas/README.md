## BTVN 01

Đây là bài tập Tiền xử lý dữ liệu trong Pandas. Mục tiêu là biến dữ liệu thô thành dữ liệu sạch, sẵn sàng cho training

Quy trình hoạt động như sau:

1. **Khám phá dữ liệu:**
    *   **Nhìn tổng quan:** In vài dòng đầu/cuối (`head`, `tail`) để xem mặt mũi dữ liệu.
    *   **Kiểm tra sức khỏe:** Xem kích thước (`shape`), tên các cột (`columns`), và thống kê mô tả (`describe`) để biết dữ liệu phân bố ra sao.

2. **Dọn dẹp:**
    *   **Vứt rác:** Cột `Hepatitis B` và `Population` quá nhiều lỗ hổng (`NaN`) hoặc không cần thiết -> Xóa thẳng tay (`drop`)
    *   **Sửa tên:** Cột `thinness 1-19 years` tên hơi dài và dễ gây nhầm lẫn -> Đổi tên cho gọn.

3. **Chuẩn hóa:**
    *   Máy tính chỉ hiểu số, không hiểu chữ "Developing" hay "Developed".
    *   **Giải pháp:** Ánh xạ (`map`) sang dạng số `0` và `1`.

4. **Tách dữ liệu:**
    *   Tách thành 2 phần riêng biệt để chuẩn bị cho Machine Learning:
    *   **X (Features):** Tất cả các cột trừ cái cần dự đoán.
    *   **y (Target):** Cột `Life expectancy` (Tuổi thọ) - đây là đáp án mà mô hình cần học

## BTVN 02

Bài này đi sâu hơn vào việc **Xử lý dữ liệu khuyết** và **Phân tích nhóm**

Quy trình hoạt động như sau:

1. **Xử lý dữ liệu khuyết:**
    *   **Bắt bệnh:** Đếm xem mỗi cột bị thủng bao nhiêu lỗ (`isnull().sum()`).
    *   **Chữa bệnh:** Thay vì vứt bỏ dòng dữ liệu, ta lấp đầy các lỗ hổng bằng giá trị trung bình (`mean`).

2. **Phân tích nhóm:**
    *   **Gom theo Quốc gia:** Để tìm xem nước nào sống thọ nhất, nước nào thấp nhất.
    *   **Gom theo Status:** So sánh xem các nước phát triển có thực sự sống thọ hơn các nước đang phát triển không
    *   *Cơ chế:* Tưởng tượng như việc chia bộ bài thành các tệp nhỏ theo tiêu chí, rồi tính toán trên từng tệp đó.

3. **Gộp dữ liệu:**
    *   Tạo thêm một bảng dữ liệu giả (`Noise_level`) và ghép nó vào bảng chính.
    *   **Cơ chế:** Giống như hàm VLOOKUP trong Excel hoặc JOIN trong SQL. Dựa vào một chìa khóa chung (`ID` / `Country`) để nối hai bảng lại với nhau
