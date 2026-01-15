## GIẢI THÍCH CODE BÀI THỰC HÀNH TUẦN 1 [01_PyTorch.ipynb](./01_PyTorch.ipynb)

## BTVN 01
Đây là đoạn code minh họa tính năng cốt lõi của PyTorch: **Autograd (Tự động tính đạo hàm)**. Thay vì phải tự tay tính toán đạo hàm, máy tính sẽ làm thay

Quy trình xử lý như sau:

1. **Khởi tạo:**
* `x = torch.tensor([1.0], requires_grad=True)`: khai báo biến `x` và bật chế độ theo dõi. PyTorch bắt đầu ghi lại mọi thay đổi lên biến này.


2. **Xây dựng biểu đồ (Forward Pass):**
* `y = 5*x**5 ...`: Khi tính `y`, PyTorch âm thầm xây dựng một biểu đồ tính toán nối từ `x` đến `y`. Nó nhớ được mối quan hệ toán học giữa hai biến.


3. **Lội ngược dòng (Backward Pass):**
* `y.backward()`: Đây là lệnh kích hoạt. PyTorch sẽ đi ngược từ `y` về `x` dựa trên biểu đồ đã xây, áp dụng quy tắc đạo hàm (Chain rule) để tính độ dốc.


4. **Kết quả:**
* đạo hàm không nằm ở biến `y`, mà được lưu ngược lại vào thuộc tính `.grad` của biến đầu vào `x`.


## BTVN 02
Đây là ví dụ về thuật toán **Gradient Descent** để tìm giá trị sao cho hàm số đạt cực tiểu.

Quy trình hoạt động như sau:

1. **Khởi tạo:**
* Tạo biến tensor `x = 2.0`.
* Bật `requires_grad=True` để PyTorch biết cần theo dõi và tính đạo hàm cho biến này.


2. **Vòng lặp 50 lần:**
* **Bước 1 - Forward Pass:** Tính giá trị hàm số `y = x**3`.
* **Bước 2 - Backward Pass:** Gọi `y.backward()`. PyTorch tự động tính đạo hàm của `y` theo `x` và lưu kết quả vào `x.grad`.
* **Bước 3 - Cập nhật:**
* Điều chỉnh `x` ngược hướng đạo hàm để "đi xuống dốc" (giảm giá trị `y`):
* Dùng `torch.no_grad()` để việc trừ này không bị tính vào đồ thị đạo hàm.
* **Bước 4 - Reset:** Gọi `x.grad.zero_()` để xóa sạch đạo hàm vừa tính, tránh việc bị cộng dồn (accumulate) ở vòng lặp sau.


3. **Kết quả:** Sau 50 vòng lặp, giá trị `x` sẽ hội tụ về điểm mà tại đó hàm số thấp nhất 

Tuy nhiên đoạn code hiện tại đang bị hiện tượng **Bùng nổ Gradient**.

1. **Learning Rate quá lớn (0.1):** Đối với hàm số này, bước nhảy 0.1 là quá mạnh. Thay vì nhích dần xuống điểm cực tiểu, thuật toán đã đẩy giá trị `x` văng ra rất xa theo hướng ngược lại

2. **Vòng lặp khuếch đại:**
* Tại **Vòng 9**, `x` đã là **-119**.
* Tại **Vòng 12**, `x` bị đẩy vọt lên **-9,500 tỷ**.
* Vì đạo hàm chứa `x^2` (từ `x^3`), khi `x` lớn, đạo hàm càng lớn khủng khiếp.


3. **Tràn số:** Giá trị `x` và `grad` tăng nhanh đến mức vượt quá giới hạn lưu trữ của máy tính (Floating point). Máy tính trả về **NaN**

**Cách sửa:** Giảm `learning_rate` xuống nhỏ hơn (ví dụ: `0.01` hoặc `0.001`), với`lr=0.01`, bước nhảy đã nhỏ lại. Giá trị `x` thay đổi từ từ và có kiểm soát, không còn bị `NaN` hay số siêu to khổng lồ như trước.

## BTVN 03

Đây là ví dụ về bài toán **Hồi quy tuyến tính (Linear Regression)**: Dạy máy tính tìm ra quy luật của dữ liệu

Quy trình hoạt động như sau:

1. **Tạo Dữ liệu giả lập:**
* Dữ liệu được tạo theo quy luật `y = 3x + 5` .
* Thêm nhiễu (noise) để dữ liệu không thẳng tắp tuyệt đối.

2. **Khởi tạo:**
* Cho khởi tạo ban đầu là:  `w=3, b=5`.

3. **Học tập (Training Loop):**
* **So sánh:** Máy tính dùng `w` và `b` hiện tại để đoán, rồi so với kết quả thật. Độ chênh lệch được gọi là **Loss**
* **Sửa sai:** Dùng `backward()` để biết cần tăng/giảm `w` và `b` thế nào cho Loss bé đi.
* **Lặp lại:** Sau 100 lần sửa, Loss giảm dần, nghĩa là máy đoán ngày càng chuẩn.

### Nhận xét kết quả:

* **Quy luật gốc:** `y = 3x + 5`.
* **Máy học được:** `y = 3.02x + 5.21`.

**Tại sao không ra chính xác 3 và 5?**
Vì dữ liệu có **nhiễu**. Máy tính cố gắng tìm đường thẳng trung bình đi qua mớ dữ liệu hỗn loạn, nên nó không thể khớp chính xác 100% với công thức gốc, mà chỉ tìm ra xấp xỉ tốt nhất.

## BTVN 04
Bài này minh họa cạm bẫy kinh điển khi chuyển dữ liệu giữa **Numpy** và **PyTorch**: Vấn đề **Dùng chung bộ nhớ** vs **Sao chép**.

Quy trình hoạt động như sau:

### 1. Tự động phát hiện kiểu dữ liệu

* **Nguyên tắc:** PyTorch cố gắng bê nguyên xi kiểu dữ liệu của Numpy sang để đỡ phải convert.
* `arr` là `int64`  Tensor ra `LongTensor` (`int64`).
* `arr2` là `int32`  Tensor ra `IntTensor` (`int32`).

### 2. Dùng chung bộ nhớ với `from_numpy`

* **Code:** `x = torch.from_numpy(arr)`
* **Cơ chế:** "Chia sẻ vùng nhớ".
* PyTorch không tạo data mới, mà trỏ thẳng vào vùng nhớ của Numpy.

* **Hậu quả:** 
* Sửa `arr[0] = 99` thì `x` cũng thay đổi theo.
* *Lợi:* Tiết kiệm RAM.
* *Hại:* Dễ toang nếu lỡ tay sửa nhầm.

### 3. Sử dụng `torch.tensor()`

* **Code:** `tensor = torch.tensor(arr3)`
* **Cơ chế:** "Photo công chứng".
* PyTorch xin cấp phát bộ nhớ mới, copy dữ liệu từ Numpy sang, rồi đóng gói lại.


## BTVN 05

Bài này minh họa hai kỹ năng trong PyTorch: **Tạo Tensor** và **Reshape**.

Quy trình hoạt động như sau:

### 1. Khởi tạo Tensor

Tạo ra các ma trận với các giá trị khởi điểm khác nhau:

* **`empty` vs `zeros`:**
* `zeros`: Xin cấp phát bộ nhớ reset về số 0.
* `empty`: Xin cấp phát bộ nhớ xong dùng luôn.
* *Lưu ý:* Trong output `empty` tình cờ ra toàn 0, nhưng thực tế nó là rác còn sót lại trong RAM.

* **`rand`:** Sinh số ngẫu nhiên .

### 2. Reshape với `view`

* **Nguyên tắc cốt lõi:** Tổng số lượng phần tử phải **không đổi**.

* `view(2, -1)` thực hiện phép chia làm 2 hàng, còn mỗi hàng bao nhiêu cột thì **máy tự chia** cho".


* **`view_as`:**
* Thay vì phải soi xem `t_mau` kích thước bao nhiêu, thì thay đổi `t_goc` để giống `t_mau` kia".


