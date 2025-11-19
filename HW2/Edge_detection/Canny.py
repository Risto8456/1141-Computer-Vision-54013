import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt

# 1.Gaussian filter
# 產生 Gaussian Kernel
def gaussian_kernel(size, sigma=1.0):
    assert size % 2 == 1
    k = size // 2
    xs, ys = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = np.exp(-(xs*xs + ys*ys) / (2 * sigma*sigma))
    kernel /= np.sum(kernel)
    return kernel
# 通用 convolution (Zero padding)
def convolve(img, kernel):
    H, W = img.shape
    k = kernel.shape[0] // 2

    padded = np.pad(img, ((k,k),(k,k)), mode='constant')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i,j] = np.sum(region * kernel)

    return output

# 2. Sobel
# 產生 Gaussian Kernel
def generate_sobel_kernel(size):
    assert size % 2 == 1
    half = size // 2

    ys, xs = np.indices((size, size), dtype=np.float32)
    xs -= half
    ys -= half

    denom = xs*xs + ys*ys
    denom[half, half] = 1.0

    Gx = xs / denom
    Gy = ys / denom

    Gx[half, half] = 0
    Gy[half, half] = 0
    return Gx, Gy
# convolution
def sobel(img, ksize=3):
    Gx, Gy = generate_sobel_kernel(ksize)

    fx = convolve(img, Gx)
    fy = convolve(img, Gy)

    magnitude = np.sqrt(fx*fx + fy*fy)
    direction = np.arctan2(fy, fx)  # 弧度

    return magnitude, direction

# 3. Non-Maximum Suppression（NMS）
def non_maximum_suppression(mag, direction):
    H, W = mag.shape
    output = np.zeros((H, W), dtype=np.float32)

    # 角度轉換到 0,45,90,135 四類
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, H-1):
        for j in range(1, W-1):

            q = 255
            r = 255

            # 0°
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]

            # 45°
            elif (22.5 <= angle[i,j] < 67.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]

            # 90°
            elif (67.5 <= angle[i,j] < 112.5):
                q = mag[i+1, j]
                r = mag[i-1, j]

            # 135°
            elif (112.5 <= angle[i,j] < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]

            if (mag[i,j] >= q) and (mag[i,j] >= r):
                output[i,j] = mag[i,j]

    return output

# 4. Double Threshold + Hysteresis（邊緣追蹤）
def double_threshold_hysteresis(img, low, high):
    H, W = img.shape
    strong = 255
    weak = 75

    result = np.zeros((H, W), dtype=np.uint8)

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    # Hysteresis
    for i in range(1, H-1):
        for j in range(1, W-1):
            if result[i,j] == weak:
                if np.any(result[i-1:i+2, j-1:j+2] == strong):
                    result[i,j] = strong
                else:
                    result[i,j] = 0

    return result

# Canny
def canny(img, gaussian_size=5, sigma=1.0, sobel_size=3,
          low_ratio=0.05, high_ratio=0.15):

    # Step1 : Gaussian
    G = gaussian_kernel(gaussian_size, sigma)
    blur = convolve(img, G)

    # Step2 : Sobel
    mag, theta = sobel(blur, ksize=sobel_size)

    # Step3 : NMS
    nms = non_maximum_suppression(mag, theta)

    # Step4 : Threshold + Hysteresis
    high = nms.max() * high_ratio
    low = high * low_ratio

    final = double_threshold_hysteresis(nms, low, high)
    return final

if __name__ == "__main__":
    # 讀圖
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_path)
    img_path = os.path.join(parent_dir, "lena.bmp")

    data = np.fromfile(img_path, dtype=np.uint8)
    color_img = cv.imdecode(data, cv.IMREAD_COLOR)
    gray = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    # Canny
    result = canny(gray)

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.imshow(gray, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(result, cmap='gray')
    plt.title("Canny")
    plt.axis("off")

    # 儲存
    save_path = os.path.join(cur_path, "Canny.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 顯示
    plt.tight_layout()
    plt.show()