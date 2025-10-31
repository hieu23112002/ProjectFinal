# Hướng dẫn chạy OCR Project

## 1. Thiết lập môi trường

### Bước 1: Kích hoạt virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### Bước 2: Cài đặt dependencies (nếu chưa có)
```powershell
pip install tensorflow==2.10.1 flask tornado opencv-python numpy matplotlib
```

## 2. Chạy server

### Cách 1: Chạy từ thư mục root
```powershell
python app/app.py
```

### Cách 2: Chạy từ thư mục app
```powershell
cd app
python app.py
```

## 3. Truy cập web interface

Mở trình duyệt và truy cập: **http://localhost:5000**

## 4. Sử dụng

- Upload ảnh chứa text cần nhận dạng
- Hệ thống sẽ xử lý và hiển thị kết quả OCR từ 2 model:
  - Word model: Nhận dạng từng từ
  - Line model: Nhận dạng cả dòng

## Lưu ý

- Server sẽ chạy trên port 5000 (mặc định)
- Model sẽ được load khi khởi động server (có thể mất vài giây)
- Ảnh upload sẽ được lưu trong thư mục `app/images/`

