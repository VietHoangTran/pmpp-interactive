# Lời thuyết minh tiếng Việt

Các tệp `*.vi.m4a` được tạo bằng Piper và mô hình `vi_VN-vais1000-medium`. Văn bản để tạo lời đọc là các phụ đề tiếng Việt trong mã nguồn bài học. Mô hình dùng tập dữ liệu VAIS 1000, ghi công theo CC BY 4.0. Phiên bản mô hình dùng cho các tệp này là `v1.0.0`. [Thông tin mô hình](https://huggingface.co/rhasspy/piper-voices/blob/v1.0.0/vi/vi_VN/vais1000/medium/MODEL_CARD) · [Piper](https://github.com/OHF-Voice/piper1-gpl) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Để tạo lại audio trên macOS, cài `piper-tts` vào môi trường Python, tải tệp ONNX và JSON cùng phiên bản từ trang mô hình, rồi chạy từ thư mục `app`:

```sh
PMPP_PIPER_PYTHON=/path/to/venv/bin/python node localization/narrate.mjs /path/to/vi_VN-vais1000-medium.onnx ch01 ch02
```

Ứng dụng phát các tệp đã tạo mà không cần cài Piper. Tệp `manifest.vi.json` lưu thời lượng từng đoạn để hoạt ảnh chạy theo tiến độ của giọng tiếng Việt.
