import cv2 as cv
import numpy as np

def _sauvola(src, win=25, k=0.2, R=128):
    src = src.astype(np.float32)
    mean = cv.boxFilter(src, ddepth=-1, ksize=(win, win), normalize=True)
    sqmean = cv.boxFilter(src * src, ddepth=-1, ksize=(win, win), normalize=True)
    var = np.maximum(sqmean - mean * mean, 0)
    std = np.sqrt(var)
    thresh = mean * (1 + k * (std / R - 1))
    return ((src > thresh).astype(np.uint8) * 255)

def _remove_small_components(bin_img, min_area=25):
    num, lbl, stats, _ = cv.connectedComponentsWithStats(bin_img, connectivity=8)
    out = np.zeros_like(bin_img)
    for i in range(1, num):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            out[lbl == i] = 255
    return out

def preprocess(path):
    # --- 1️⃣ Đọc và chuyển xám ---
    bgr = cv.imread(path)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    # --- 2️⃣ Khử nền (dùng blur mạnh và normalize) ---
    bg = cv.medianBlur(gray, 51)
    norm = cv.normalize((gray.astype(np.float32) / (bg.astype(np.float32) + 1e-3)) * 128.0,
                        None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # --- 3️⃣ Làm mịn giảm nhiễu ---
    norm = cv.bilateralFilter(norm, d=9, sigmaColor=50, sigmaSpace=50)

    # --- 4️⃣ Tăng tương phản cục bộ ---
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)

    # --- 5️⃣ Binarize (Sauvola) ---
    bin_img = _sauvola(norm, win=25, k=0.2, R=128)

    # # --- 6️⃣ Xóa dòng kẻ ngang ---
    # bin_img = _remove_horizontal_lines(bin_img, min_len_ratio=0.2, thickness=3)

    # --- 7️⃣ Lọc nhiễu nhỏ ---
    bin_img = _remove_small_components(bin_img, min_area=50)

    # --- 8️⃣ Làm mịn nhẹ để giảm chấm nhỏ ---
    bin_img = cv.medianBlur(bin_img, 3)

    # 🟩 --- 9️⃣ Làm đậm và sắc nét chữ ---
    # Bước 1: Closing để nối chữ, lấp khoảng trống nhỏ
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    bin_img = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel_close, iterations=1)

    # Bước 3: Làm mượt lại bằng Gaussian Blur nhẹ
    bin_img = cv.GaussianBlur(bin_img, (3, 3), 0)

    return bin_img


if __name__ == "__main__":
    path_in = "../image/d943294e-57e5-4c90-91a6-647619bcf55e.jpg"
    path_out = "../image/output_clean1.jpg"
    img = preprocess(path_in)
    cv.imwrite(path_out, img)
    print("✅ Ảnh đã lưu:", path_out)
