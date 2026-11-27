# Lab 2.1 — Cộng vector từ đầu đến cuối

Xem mục 2.3 đến 2.6 của sách (trang 26–39). Kernel nằm ở hình 2.10 (trang 35), hàm host ở hình 2.13 (trang 38), và mẫu kiểm tra lỗi ở ghi chú bên lề trang 32. Phiên bản Metal là phần bổ sung của khóa học.

Bạn sẽ viết toàn bộ chương trình của chương này. Mã kiểm thử đóng vai `main` trong hình 2.4. Nó giữ ba mảng host thông thường, gọi `vecAdd` của bạn và chỉ làm việc với các địa chỉ host.

## CUDA, HIP và trình giả lập

Sửa `work/kernel.cu`.

- `CHECK(call)` triển khai mẫu kiểm tra lỗi trang 32 dưới dạng macro dùng lại được. Đoạn mã in trong sách có lỗi biên dịch. Hãy tìm nguyên nhân trước khi dùng. Macro in chuỗi lỗi, tên tệp và số dòng rồi trả một giá trị khác 0 từ hàm bao quanh. Việc giải phóng nằm ngoài macro. Dùng `return` để bộ kiểm thử có thể gọi tiếp sau một lần lỗi.
- `vecAddKernel(A, B, C, n)` thực hiện một phép cộng theo cặp trên mỗi luồng.
- `vecAdd(A_h, B_h, C_h, n, threadsPerBlock)` cấp phát ba mảng trên thiết bị, chép A và B sang, tính số block từ kích thước block được truyền vào, rồi khởi chạy. Sau đó kiểm tra lần khởi chạy, chờ kernel, chép C về và giải phóng. Mọi lời gọi runtime đi qua `CHECK`. Hàm trả 0 nếu tất cả thành công, hoặc một giá trị khác 0 nếu có lỗi.

## Metal

Sửa `work/kernel.metal` và `work/launch.mm`.

- `vec_add` là kernel.
- `vecAdd(device, queue, pipeline, A_h, B_h, C_h, n)` tạo ba buffer, ghi lệnh, khởi chạy đúng `n` luồng, commit rồi chờ. Sau đó kiểm tra kết quả thực thi và chép đầu ra về. Với các hàm tạo đối tượng, kiểm tra giá trị `nil`. Sau khi chờ, kiểm tra trạng thái và lỗi của command buffer.
- Lab cung cấp sẵn `makeVecAddPipeline`. Nó chuyển mã `kernel.metal` thành một pipeline state. Mã kiểm thử gọi hàm này một lần và giữ kết quả.

Giữ nguyên tên hàm và chữ ký. Mã điều khiển kiểm thử nằm trong `main.cpp`; phần bạn sửa nằm trong `work/`.

Từ thư mục gốc của dự án, chạy kiểm thử với trình giả lập trước. Sau đó có thể bỏ `--backend emu` để thử trên mọi backend khả dụng.

```sh
python3 labs/pmpp.py run 2.1 --backend emu
```

Đo hiệu năng trên thiết bị thật bằng lệnh sau.

```sh
python3 labs/pmpp.py run 2.1 --bench
```

## Bộ kiểm thử tìm gì?

Đầu vào nằm trong khoảng [0.5, 1.5] nên tổng đúng luôn cách xa 0. Ban đầu, `C_h` chứa NaN. Các độ dài kiểm thử gồm 1, 31, 32, 33, 255, 256, 257, 1000 và 4097, nằm ở hai phía của các mốc kích thước block. Thiết bị thật có thêm độ dài 2^20 + 3. CUDA, HIP và trình giả lập chạy mỗi độ dài với 32 và 256 luồng mỗi block. Một trường hợp dùng số luồng tối đa mỗi block theo thông tin thiết bị.

- Nếu vẫn còn NaN trong `C_h`, những phần tử đó chưa được chép về.
- Nếu phần cuối sai, hãy kiểm tra số block. Vùng nhớ thiết bị chưa được luồng nào ghi chứa mẫu đánh dấu trên trình giả lập, nội dung cũ trên CUDA và số 0 trên Metal.
- Nếu AddressSanitizer báo lỗi trên trình giả lập, xem dòng mã được chỉ ra. Nguyên nhân có thể là thiếu kiểm tra chỉ số, truyền số phần tử vào đối số cần số byte hoặc giải phóng trước khi chép về. CUDA không tài liệu hóa việc kiểm tra biên của mảng thiết bị, nên hãy chạy trình giả lập trước.
- Riêng CUDA, HIP và trình giả lập, có một lời gọi yêu cầu block lớn gấp đôi giới hạn thiết bị. Lời gọi đó phải trả một giá trị khác 0 và chương trình tiếp tục. Lời gọi hợp lệ ngay sau đó phải trả 0 và cho đúng các tổng.

## Trước khi đo hiệu năng

Hãy ghi lại một dự đoán. Ở kích thước lớn nhất, vòng lặp CPU mất thời gian gấp bao nhiêu lần lời gọi `vecAdd` hoàn chỉnh? Tỷ số lớn hơn 1 nghĩa là bản GPU nhanh hơn.

Bộ đo tính thời gian vòng lặp CPU trong hình 2.4 và thời gian một lời gọi `vecAdd` từ đầu đến cuối. Nó thử ba độ dài 2^16, 2^20 và 2^24 phần tử, hoặc ít hơn nếu thiết bị báo thiếu bộ nhớ. Mỗi trường hợp có một lượt làm nóng không tính giờ để bỏ chi phí khởi động runtime, rồi lấy kết quả nhanh nhất trong vài lượt đo và in tỷ số. Lab 2.2 phân tích thời gian đi vào từng bước.

## Trạng thái lời giải tham khảo

Trạng thái ghi trong bản gốc tại thời điểm biên soạn gồm đã chạy kiểm thử trên Metal, đã kiểm tra trên trình giả lập và đã qua kiểm tra cú pháp CUDA bằng front end. Khi đó bản gốc ghi chưa chạy trên thiết bị CUDA hay HIP. Các báo cáo chạy sau đó nằm trong [thư mục reports](../../../reports).
