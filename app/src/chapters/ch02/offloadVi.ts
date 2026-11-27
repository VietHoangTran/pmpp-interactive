import type { Defect, Env, Group, Outcome, OutcomeClass } from '../../sim/offload';

export const groupVi: Record<Group, string> = {
  allocate: 'cấp phát',
  'copy in': 'chép sang',
  fill: 'điền qua contents',
  launch: 'khởi chạy',
  wait: 'chờ',
  'copy back': 'chép về',
  read: 'đọc qua contents',
  free: 'giải phóng',
};

export const defectVi: Record<Defect, string> = {
  'bytes-as-elements': 'Kích thước của lần chép về là n thay vì n * sizeof(float).',
  'alloc-in-elements': 'Ba lần cấp phát chỉ xin n byte, nhưng các lượt sao chép chuyển n * sizeof(float) byte.',
  'missing-copy-in': 'B chưa được chép sang thiết bị.',
  'copy-back-early': 'Lời gọi chép về đứng trước lệnh khởi chạy.',
  'free-before-copy-back': 'C_d bị giải phóng trước khi chép về.',
  'swapped-pointers': 'Hai đối số đầu của lượt chép về bị đổi chỗ, còn kiểu sao chép giữ nguyên.',
  'host-dereference': 'Mã host đọc trực tiếp C_d[0].',
};

export const outcomeClassVi: Record<OutcomeClass, string> = {
  'no-counterpart': 'không có thao tác tương ứng',
  passes: 'kết quả đúng, không báo lỗi',
  'wrong-values': 'giá trị sai',
  'out-of-bounds': 'truy cập ngoài vùng cấp phát',
  'use-after-free': 'dùng sau khi giải phóng',
  'undefined-per-docs': 'hành vi không xác định theo tài liệu',
  'not-documented': 'không được phép, tài liệu không nêu hậu quả',
  'slow-zero-copy': 'truy cập zero copy chậm theo tài liệu AMD',
};

const summaries: Record<Defect, { cuda: string; emu: string } & Partial<Record<Env, string>>> = {
  'bytes-as-elements': {
    cuda: 'Lời gọi chép n byte, tức n/4 số float. Phần còn lại của C_h giữ giá trị bên gọi đã đặt. Mọi lời gọi đều báo thành công.',
    emu: 'Chạy đến hết vì thao tác sao chép nằm trong biên nhưng quá ngắn. n/4 phần tử đầu đúng, phần còn lại giữ giá trị đánh dấu.',
  },
  'alloc-in-elements': {
    cuda: 'Mỗi vùng cấp phát chỉ bằng một phần tư dung lượng mà sao chép và kernel dùng, nên chúng ghi ra ngoài. Tài liệu không bảo đảm có kiểm tra biên và không xác định kết quả. Kết quả thậm chí có thể trông đúng.',
    emu: 'AddressSanitizer dừng ở lượt chép sang đầu tiên vì ghi 4n byte vào vùng n byte.',
  },
  'missing-copy-in': {
    cuda: 'Kernel cộng A với nội dung cũ trong B_d vì cudaMalloc không xóa vùng nhớ. Mọi phần tử đều sai và không lời gọi nào báo lỗi.',
    hip: 'Kernel cộng A với nội dung sẵn có trong B_d. Tệp tiêu đề AMD không nêu hipMalloc có xóa bộ nhớ hay không, nên hãy coi vùng đó chưa khởi tạo. Không lời gọi nào báo lỗi.',
    metal: 'Buffer B được điền số 0, nên kết quả là A + 0 = A. Kết quả sai dù trông hợp lý.',
    emu: 'Chạy đến hết. Trình giả lập điền vùng nhớ mới bằng byte 0xA5, tương ứng float khoảng -2.9e-16. Khi cộng vào A, giá trị nhỏ này bị làm tròn mất, nên kết quả bằng A.',
  },
  'copy-back-early': {
    cuda: 'C_h nhận nội dung chưa khởi tạo của C_d trước khi kernel ghi. Kernel chạy sau đó và kết quả không được chép về.',
    metal: 'Bộ nhớ dùng chung nên đọc được. Giá trị đọc được là các số 0 điền lúc tạo buffer.',
    emu: 'Chạy đến hết. Mọi phần tử của C_h giữ mẫu điền 0xA5 của trình giả lập.',
  },
  'free-before-copy-back': {
    cuda: 'Lượt chép đọc vùng nhớ đã trả về kho cấp phát. Tài liệu không quy định mã lỗi hay giá trị thu được. Giá trị có thể vẫn trông đúng.',
    emu: 'AddressSanitizer dừng tại lượt chép về vì đọc vùng nhớ đã giải phóng.',
  },
  'swapped-pointers': {
    cuda: 'Con trỏ không khớp kiểu sao chép dẫn đến hành vi không xác định. Mã cudaErrorInvalidMemcpyDirection tồn tại, nhưng tài liệu không bảo đảm sẽ nhận mã đó.',
    hip: 'Tài liệu AMD quy định hành vi không xác định nếu con trỏ không khớp kiểu sao chép, giống CUDA. Tài liệu không bảo đảm mã lỗi.',
    emu: 'Chạy đến hết. Trình giả lập chép theo hai con trỏ, từ host sang thiết bị. C_h không được ghi và giữ giá trị đánh dấu.',
  },
  'host-dereference': {
    cuda: 'Kết quả đúng, rồi hàm bao thực hiện một thao tác CUDA không cho phép. NVIDIA không mô tả hậu quả. Sách nói có thể xảy ra ngoại lệ hoặc lỗi lúc chạy.',
    hip: 'Kết quả đúng. AMD mô tả host có thể truy cập bộ nhớ hipMalloc qua zero copy với giao diện chậm, tùy cấu hình. Trường hợp này chưa được kiểm tra ở đây.',
    metal: 'Sau khi host chờ GPU hoàn tất, đọc buffer dùng chung qua contents là cách thông thường.',
    emu: 'Mọi kiểm thử đều qua. Host và thiết bị dùng chung không gian địa chỉ trong trình giả lập, nên không có báo lỗi khi đọc.',
  },
};

export function outcomeSummaryVi(outcome: Outcome) {
  const summary = summaries[outcome.defect];
  return summary[outcome.env] ?? (outcome.env === 'hip'
    ? summary.cuda
    : 'Khi dữ liệu ở ngay trong các buffer dùng chung, không có lời gọi tương ứng để mắc lỗi này. Bản này không truyền số byte hay kiểu sao chép và không gọi giải phóng riêng.');
}
