# Local Thresholding
# Sauvola's method: T = μ * [1 + k(σ/R - 1)], where R is the dynamic range of standard deviation (typically 128 for an 8-bit image)
# Sauvola 方法：T = μ * [1 + k(σ/R - 1)]，其中 R 是標準差的動態範圍（對於 8 位元影像通常為 128）

import cv2
import numpy as np

# ==============================
# Sauvola's method: T = μ * [1 + k(σ/R - 1)]
# ==============================
def local_thresholding_sauvola(image, window_size=15, k=0.5, R=128):
    # R 是標準差的動態範圍（對於 8 位元影像通常為 128）
    h, w = image.shape     # 影像高度與寬度
    r = window_size // 2   # 半徑
    binary_image = np.zeros_like(image, dtype=np.uint8)

    # 積分圖與平方積分圖
    integral = cv2.integral(image, sdepth=cv2.CV_64F)
    integral_sq = cv2.integral(np.square(image), sdepth=cv2.CV_64F)

    for y in range(h):
        for x in range(w):
            # 區域邊界 (不超出圖像)
            y1, y2 = max(0, y - r), min(h - 1, y + r)
            x1, x2 = max(0, x - r), min(w - 1, x + r)

            # 區域和與平方和
            S  = (integral[y2 + 1, x2 + 1] - integral[y1, x2 + 1]
                - integral[y2 + 1, x1] + integral[y1, x1])
            S2 = (integral_sq[y2 + 1, x2 + 1] - integral_sq[y1, x2 + 1]
                - integral_sq[y2 + 1, x1] + integral_sq[y1, x1])

            area = (y2 - y1 + 1) * (x2 - x1 + 1)    # 區域面積
            mean = S / area                         # 均值
            var = (S2 / area) - (mean ** 2)         # 變異數
            std = np.sqrt(max(var, 0))              # 標準差，避免負數

            # Sauvola 閾值公式
            T = mean * (1 + k * ((std / R) - 1))
            binary_image[y, x] = 255 if image[y, x] > T else 0

    return binary_image


# ==============================
# 主程式比較
# ==============================
if __name__ == "__main__":
    image = cv2.imread('HW1/lena.bmp', cv2.IMREAD_GRAYSCALE)

    # 參數設定
    window_size = 15    # 鄰域大小 (必須為奇數)
    k = 0            # 常見取值: -0.2 (較保守)、0 (等於mean)、0.2 (較寬鬆)
    R = 128             # 對於 8 位元影像，R 通常取 128

    # Sauvola 自適應閾值
    binary_sauvola = local_thresholding_sauvola(image, window_size, k, R)

    # 顯示結果
    cv2.imshow('Original', image)
    cv2.imshow(f'Sauvola Thresholding (k={k})', binary_sauvola)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
