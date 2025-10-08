# Local Thresholding
# Adaptive mean thresholding: Using the local mean minus a constant
# 自適應平均值閾值：使用局部平均值減去一個常數

import cv2
import numpy as np

def local_thresholding_adaptive (image, window_size=5, c=5):
    """
    使用積分圖加速的局部均值法閾值化
    image: 灰階影像 (numpy array)
    window_size: 鄰域大小 (必須為奇數)
    c: 常數，從計算出的均值中減去的值
    """
    h, w = image.shape      # 影像高度與寬度
    r = window_size // 2    # 半徑
    binary_image = np.zeros_like(image, dtype=np.uint8) # 初始化二值影像

    # 預先計算積分圖 (integral 輸出比原圖大一圈)
    # 積分圖就是二維前綴和 (2D prefix sum)
    integral = cv2.integral(image, sdepth=cv2.CV_64F) # 使用 64 位元浮點數以防溢位

    for y in range(h):
        for x in range(w):
            # 鄰域範圍 (使用 integral 圖座標多 +1)
            y1, y2 = max(0, y - r), min(h - 1, y + r)
            x1, x2 = max(0, x - r), min(w - 1, x + r)

            # 轉成積分圖索引 (+1)
            sum_region = ( # 計算區域和
                integral[y2 + 1, x2 + 1] - integral[y1, x2 + 1]
                - integral[y2 + 1, x1] + integral[y1, x1]
            )

            # 區域面積 = window_size^2 (邊界處會小於)
            area = (y2 - y1 + 1) * (x2 - x1 + 1)
            mean_val = sum_region / area    # 計算均值
            T = mean_val - c                # 閾值
            binary_image[y, x] = 255 if image[y, x] > T else 0  # 二值化

    return binary_image

if __name__ == "__main__":
    # 讀入灰階影像
    image = cv2.imread('HW1/lena.bmp', cv2.IMREAD_GRAYSCALE)

    # 參數設定
    window_size = 15    # 鄰域大小 (必須為奇數)
    c = 5               # 常數

    # 自適應平均值閾值
    binary = local_thresholding_adaptive(image, window_size, c)
    # binary_0 = local_thresholding_adaptive(image, window_size, 0)

    # 顯示結果
    cv2.imshow('Original', image)
    cv2.imshow('Local : Adaptive mean thresholding', binary)
    # cv2.imshow('Local : Adaptive mean thresholding_0', binary_0)

    cv2.waitKey(0) # 等待按鍵
    cv2.destroyAllWindows() # 關閉所有視窗