import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

"Dùng trong việc tách từ , dòng trong ảnh"
def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):

    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c)  # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = img[y:y + h, x:x + w]
        res.append((currBox, currImg))

    # return list of words, sorted by x-coordinate
    return sorted(res, key=lambda entry: entry[0][0])


def prepareImg(img, height):
    """convert given image to grayscale image (if needed) and resize to desired height"""
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
    """create anisotropic filter kernel according to given parameters"""
    assert kernelSize % 2  # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x ** 2 / (2 * sigmaX) - y ** 2 / (2 * sigmaY))
            xTerm = (x ** 2 - sigmaX ** 2) / (2 * math.pi * sigmaX ** 5 * sigmaY)
            yTerm = (y ** 2 - sigmaY ** 2) / (2 * math.pi * sigmaY ** 5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

if __name__ == '__main__':
    input_path = '../image/output_clean1.jpg'   # ảnh đã qua preprocess
    output_path = '../image/w1.jpg'     # nơi lưu ảnh kết quả

    # 1️⃣ Đọc ảnh đầu vào
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    # 2️⃣ Chuẩn bị ảnh (resize chiều cao về 64 cho đồng nhất)
    imgPrepared = prepareImg(img, 64)

    # 3️⃣ Tách từ
    words = wordSegmentation(imgPrepared, kernelSize=25, sigma=11, theta=7, minArea=100)

    # 4️⃣ Vẽ bounding boxes quanh các vùng phát hiện được
    imgBoxed = cv2.cvtColor(imgPrepared, cv2.COLOR_GRAY2BGR)
    for (box, w_img) in words:
        x, y, w, h = box
        cv2.rectangle(imgBoxed, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 5️⃣ Lưu kết quả ra file
    cv2.imwrite(output_path, imgBoxed)
    print(f"✅ Đã lưu ảnh có bounding box tại: {output_path}")
    print(f"Phát hiện được {len(words)} vùng chữ (từ)")

    # 6️⃣ (Tuỳ chọn) Hiển thị để xem nhanh
    plt.imshow(cv2.cvtColor(imgBoxed, cv2.COLOR_BGR2RGB))
    plt.title("Segmentation Result")
    plt.show()