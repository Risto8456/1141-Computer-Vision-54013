# Local Thresholding
# Niblack's method: T = μ + 𝑘σ, where μ is the local mean, σ is the local standard deviation, and 𝑘 is a constant
# Niblack 方法：T = μ + kσ，其中 μ 是局部均值，σ 是局部標準差，k 是常數

import cv2
import numpy as np

# ==============================
# Niblack's method: T = μ + kσ
# 使用積分圖與平方積分圖加速
# ==============================
def local_thresholding_niblack(image, window_size=15, k=-0.2):
    """
    Niblack's local thresholding method
    T = mean + k * std
    image: 灰階影像 (numpy array)
    window_size: 鄰域大小 (必須為奇數)
    k: 常數，一般介於 [-0.5, 0.5]
    """
    h, w = image.shape      # 影像高度與寬度
    r = window_size // 2    # 半徑
    binary_image = np.zeros_like(image, dtype=np.uint8) # 初始化二值影像

    # 建立積分圖與平方積分圖 (多一圈邊界)
    integral = cv2.integral(image, sdepth=cv2.CV_64F) # 使用 64 位元浮點數以防溢位
    integral_sq = cv2.integral(np.square(image), sdepth=cv2.CV_64F) # 平方積分圖

    for y in range(h):
        for x in range(w):
            # 區域邊界 (不超出圖像)
            y1, y2 = max(0, y - r), min(h - 1, y + r)
            x1, x2 = max(0, x - r), min(w - 1, x + r)

            # 使用積分圖計算區域總和與平方和
            S  = (integral[y2 + 1, x2 + 1] - integral[y1, x2 + 1]
                - integral[y2 + 1, x1] + integral[y1, x1])
            S2 = (integral_sq[y2 + 1, x2 + 1] - integral_sq[y1, x2 + 1]
                - integral_sq[y2 + 1, x1] + integral_sq[y1, x1])

            # 區域面積
            area = (y2 - y1 + 1) * (x2 - x1 + 1)

            # 均值與標準差
            mean = S / area                 # 均值
            var = (S2 / area) - (mean ** 2) # 變異數 = E[X^2] - (E[X])^2
            std = np.sqrt(max(var, 0))      # 避免浮點誤差造成負數

            # Niblack 閾值
            T = mean + k * std

            binary_image[y, x] = 255 if image[y, x] > T else 0

    return binary_image


if __name__ == "__main__":
    # 讀入灰階影像
    image = cv2.imread('HW1/lena.bmp', cv2.IMREAD_GRAYSCALE)

    # 參數設定
    window_size = 15 # 鄰域大小 (必須為奇數)
    k = -0.2  # 常見取值: -0.2 (較保守)、0 (等於mean)、0.2 (較寬鬆)

    # Niblack 自適應閾值
    binary_niblack = local_thresholding_niblack(image, window_size, k)

    # 顯示結果
    cv2.imshow('Original', image)
    cv2.imshow(f'Niblack Thresholding (k={k})', binary_niblack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
