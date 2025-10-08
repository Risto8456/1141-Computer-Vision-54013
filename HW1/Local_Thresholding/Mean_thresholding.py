# Local Thresholding
# Mean thresholding: Using the local mean intensity value
# 平均閾值：使用局部平均強度值

import cv2
import numpy as np

# ==============================
# 直觀版本：逐像素取區域平均，時間較長
# 時間複雜度 O(h*w*window_size^2)
# ==============================
def local_thresholding_naive(image, window_size=5):
    """
    使用局部均值法進行閾值化 (直觀版本)
    image: 灰階影像 (numpy array)
    window_size: 鄰域大小 (必須為奇數)
    """
    h, w = image.shape      # 影像高度與寬度
    r = window_size // 2    # 半徑
    binary_image = np.zeros_like(image, dtype=np.uint8) # 初始化二值影像

    for y in range(h):
        for x in range(w):
            # 鄰域範圍 (防止超出邊界)
            y1, y2 = max(0, y - r), min(h, y + r + 1)   # y1, y2 是區域的上下邊界
            x1, x2 = max(0, x - r), min(w, x + r + 1)   # x1, x2 是區域的左右邊界

            region = image[y1:y2, x1:x2]                # 取出鄰域
            T = np.mean(region)                         # 計算均值作為閾值
            binary_image[y, x] = 255 if image[y, x] > T else 0 # 二值化

    return binary_image


# ==============================
# 積分圖版本：快速平均值計算，時間較短
# 時間複雜度 O(h*w)
# ==============================
def local_thresholding_integral(image, window_size=5):
    """
    使用積分圖加速的局部均值法閾值化
    image: 灰階影像 (numpy array)
    window_size: 鄰域大小 (必須為奇數)
    """
    h, w = image.shape      # 影像高度與寬度
    r = window_size // 2    # 半徑
    binary_image = np.zeros_like(image, dtype=np.uint8) # 初始化二值影像

    # 預先計算積分圖 (integral 輸出比原圖大一圈)
    # 積分圖其實就是二維前綴和 (2D prefix sum)
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
            T = mean_val                    # 閾值
            binary_image[y, x] = 255 if image[y, x] > T else 0  # 二值化

    return binary_image

if __name__ == "__main__":
    # 讀入灰階影像
    image = cv2.imread('HW1/lena.bmp', cv2.IMREAD_GRAYSCALE)

    # 參數設定
    window_size = 15 # 鄰域大小 (必須為奇數)

    # 直觀版本
    # binary = local_thresholding_naive(image, window_size)
    # 積分版本
    binary = local_thresholding_integral(image, window_size)

    # 顯示結果
    cv2.imshow('Original', image)
    cv2.imshow('Local : Mean thresholding', binary)
    cv2.waitKey(0) # 等待按鍵
    cv2.destroyAllWindows() # 關閉所有視窗