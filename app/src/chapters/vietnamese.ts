import type { Section, Tier } from './types';

type VietnamesePlanRow = { title: string; why: string; milestone?: string };

export const vietnamesePlan: Record<string, VietnamesePlanRow> = {
  ch01: { title: 'Giới thiệu', why: 'Động lực học lập trình song song, định luật Amdahl và bốn thách thức.' },
  ch02: { title: 'Tính toán song song dữ liệu trên hệ thống không đồng nhất', why: 'Kernel đầu tiên, host và thiết bị, cú pháp khởi chạy.', milestone: 'M0 · Phép toán vector' },
  ch03: { title: 'Lưới và dữ liệu đa chiều', why: 'Ánh xạ luồng vào dữ liệu, nhân ma trận theo cách trực tiếp.', milestone: 'M1 · GEMM cơ bản' },
  ch04: { title: 'Kiến trúc tính toán và lập lịch', why: 'Mô hình thực thi qua warp, phân kỳ và mức chiếm dụng (occupancy).', milestone: 'M1 · Cấu hình chạy từ thông số thiết bị' },
  ch05: { title: 'Kiến trúc bộ nhớ và tính cục bộ của dữ liệu', why: 'FLOP trên mỗi byte, mô hình roofline, các loại bộ nhớ và kỹ thuật chia ô.', milestone: 'M2 · GEMM chia ô' },
  ch06: { title: 'Các yếu tố ảnh hưởng đến hiệu năng', why: 'Gộp truy cập bộ nhớ, xung đột bank và các bước tối ưu.', milestone: 'M2 · Tinh chỉnh hiệu năng' },
  ch07: { title: 'Tích chập', why: 'Bộ nhớ hằng và kỹ thuật chia ô có vùng biên (halo).' },
  ch08: { title: 'Tính toán theo mẫu láng giềng (stencil)', why: 'Giao thêm việc cho mỗi luồng và chia ô bằng thanh ghi.' },
  ch09: { title: 'Biểu đồ tần suất (histogram)', why: 'Phép toán nguyên tử, tranh chấp tài nguyên và dùng bản sao riêng.', milestone: 'M5 · Đếm token' },
  ch10: { title: 'Phép rút gọn (reduction)', why: 'Hệ số chuẩn hóa softmax, max và argmax.', milestone: 'M4 · Softmax và argmax' },
  ch11: { title: 'Quét tiền tố (scan)', why: 'Xác suất tích lũy cho top-p.', milestone: 'M5 · Top-p' },
  ch12: { title: 'Lọc', why: 'Gom các ứng viên còn lại sau khi lọc để lấy mẫu.', milestone: 'M5 · Top-k' },
  ch13: { title: 'Trộn dãy đã sắp xếp', why: 'Tập trung vào co-rank, cách xác định vị trí tương ứng trong hai dãy.' },
  ch14: { title: 'Sắp xếp', why: 'Radix sort và cách sắp xếp logits.', milestone: 'M5 · Sắp xếp ứng viên' },
  ch15: { title: 'Tối ưu nâng cao cho phép nhân ma trận', why: 'Từ GEMM chia ô đến kernel có thể dùng trong thư viện.', milestone: 'M3 · GEMM nhanh' },
  ch16: { title: 'Quy hoạch động và song song theo mặt sóng', why: 'Đồng bộ dựa trên quan hệ phụ thuộc.' },
  ch17: { title: 'Tính toán trên ma trận thưa', why: 'Đánh đổi khi chọn định dạng lưu trữ.' },
  ch18: { title: 'Duyệt đồ thị', why: 'Tập đỉnh đang xét (frontier) và hàng đợi riêng.' },
  ch19: { title: 'Mạng nơ-ron tích chập', why: 'Đưa tích chập về GEMM và cách framework dùng thư viện.' },
  ch20: { title: 'Mô hình ngôn ngữ lớn', why: 'Kernel attention, KV cache và FlashAttention.', milestone: 'M6 · Attention, M7 · KV cache' },
  ch21: { title: 'Bản đồ thế tĩnh điện', why: 'Hai cách tổ chức truy cập dữ liệu, scatter và gather.' },
  ch22: { title: 'Chọn thuật toán và mô hình hóa bài toán', why: 'Định luật Amdahl và đánh đổi giữa độ trễ với thông lượng khi gom lô.', milestone: 'M8 · Khảo sát kích thước lô' },
  ch23: { title: 'Lập trình đa GPU', why: 'Stream, sao chép bất đồng bộ và bộ nhớ ghim (pinned memory).' },
  ch24: { title: 'Kết luận và triển vọng', why: 'Tổng kết và định hướng.' },
  appA: { title: 'Các vấn đề về tính toán số', why: 'Thứ tự cộng và các định dạng có độ chính xác thấp.' },
  B: { title: 'Kiến thức cơ bản về học sâu', why: 'Phần kiến thức đầu vào của khóa học.' },
  appC: { title: 'Bộ nhớ CUDA, không gian địa chỉ và con trỏ', why: 'Các mô hình bộ nhớ và sự khác biệt với bộ nhớ hợp nhất.' },
  x1: { title: 'Kernel lấy mẫu', why: 'Greedy, top-k và top-p trên GPU.', milestone: 'M5' },
  x2: { title: 'Kernel lượng tử hóa', why: 'GEMV với trọng số int8 và 4 bit, cùng bước giải mã lượng tử hóa.', milestone: 'M9' },
  x3: { title: 'Đọc mã GPU trong các dự án thực tế', why: 'llama.cpp, MLX, PyTorch và quy trình chạy mô hình kiểu Llama từ đầu đến cuối.', milestone: 'M10' },
};

export const vietnameseTier: Record<Tier, string> = {
  Deep: 'Chuyên sâu',
  Standard: 'Tiêu chuẩn',
  Condensed: 'Rút gọn',
  Skipped: 'Bỏ qua',
};

export const vietnameseSections: Record<string, Pick<Section, 'title' | 'lede'>> = {
  'ch01/start': {
    title: 'Bắt đầu từ hai thiết kế có cùng ngân sách',
    lede: 'Giải thích vì sao GPU thắng CPU khi có nhiều phép toán độc lập nhưng thua khi chỉ có một chuỗi phụ thuộc. Dự đoán thiết kế nào sẽ hoàn thành trước với một khối lượng công việc cụ thể.',
  },
  'ch01/1.3': {
    title: 'Mức tăng tốc và giới hạn của nó',
    lede: 'Ước tính mức tăng tốc tối đa của cả chương trình trước khi viết kernel tối ưu, rồi chỉ ra bốn khó khăn mà cuốn sách sẽ nhiều lần quay lại.',
  },
  'ch01/1.7': {
    title: 'Cuốn sách, ấn bản này và lộ trình của bạn',
    lede: 'Lên thứ tự đọc phù hợp với mục tiêu của bạn và nhận ra khi một tài liệu trực tuyến đề cập đến đúng ấn bản này.',
  },
  'ch01/wrap': {
    title: 'Tóm tắt và tự kiểm tra',
    lede: 'Những điều cần nhớ trước khi sang chương 2, câu hỏi ôn tập cả chương, và những chỗ nội dung đã cũ hoặc bản PDF có lỗi.',
  },
};
