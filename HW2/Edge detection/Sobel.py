# Sobel edge detection 
# Sobel 邊緣偵測

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 根據 StackOverflow 上的方法，用公式 Gx 和 Gy 產生 Sobel kernel
# 參考 : https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size
def generate_sobel_kernel(size):
    assert size % 2 == 1, "Kernel size 必須是奇數"
    half = size // 2
    # 產生坐標網格 (i, j)
    # i 是 x 偏移 (水平偏移)，j 是 y 偏移 (垂直偏移)
    # 注意 indices: 第一維是 row (y, 對應 j)，第二維是 col (x, 對應 i)
    ys, xs = np.indices((size, size), dtype=np.float32)
    # 把中心移到 (0,0)
    xs = xs - half
    ys = ys - half

    # 計算距離平方 (i^2 + j^2)
    denom = xs * xs + ys * ys
    # 避免中心除以 0
    denom[half, half] = 1.0  # 先暫時設為 1，之後再把 kernel 中心設成 0

    # Gx, Gy
    Gx = xs / denom
    Gy = ys / denom

    # 中心 (i=0, j=0) 設為 0
    Gx[half, half] = 0.0
    Gy[half, half] = 0.0

    return Gx, Gy

# 自製 Sobel 運算函式
def Sobel_operator(img, direction='x', ksize=3):
    """
    自己實作 Sobel 運算
    img: 單通道灰階影像
    direction: 'x' 或 'y'
    ksize: 3, 5, 7 (奇數)
    """
    Gx, Gy = generate_sobel_kernel(ksize)
    kernel = Gx if direction == 'x' else Gy

    # convolution 邊界填充 0
    pad = ksize // 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode='constant')

    H, W = img.shape
    output = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            region = padded[i:i+ksize, j:j+ksize]
            output[i, j] = np.sum(region * kernel)

    return output

def Sobel_edge_detection_test(img):
    plt.figure(figsize=(15, 10))

    # 嘗試 mask size = 3, 5, 7, 9
    s = 1
    e = 5
    for n in range(s, e):
        kernel_size = 1 + (n * 2)

        # 使用自製 Sobel 函式
        sx = Sobel_operator(img, direction='x', ksize=kernel_size)
        sy = Sobel_operator(img, direction='y', ksize=kernel_size)

        # 轉成 uint8 顯示
        absX = cv.convertScaleAbs(sx)
        absY = cv.convertScaleAbs(sy)
        # 合併兩張影像, 兩方向的權重各 50%
        dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

        # 顯示
        idx = (n-1) * 3 + 1

        plt.subplot(e-s+1, 3, idx)
        plt.imshow(absX, cmap='gray')
        plt.title(f'Vertical (ksize={kernel_size})')
        plt.axis('off')

        plt.subplot(e-s+1, 3, idx + 1)
        plt.imshow(absY, cmap='gray')
        plt.title(f'Horizontal (ksize={kernel_size})')
        plt.axis('off')

        plt.subplot(e-s+1, 3, idx + 2)
        plt.imshow(dst, cmap='gray')
        plt.title(f'Complete (ksize={kernel_size})')
        plt.axis('off')

    # 儲存結果
    cur_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(cur_path, 'Sobel_operator_custom.jpg')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 讀取影像
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_path)
    img_path = os.path.join(parent_dir, "lena.bmp")

    data = np.fromfile(img_path, dtype=np.uint8)
    color_img = cv.imdecode(data, cv.IMREAD_COLOR)
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    Sobel_edge_detection_test(gray_img)

    # sobel kernel
    # print(generate_sobel_kernel(3)[0]*2)
    # print(generate_sobel_kernel(5)[0]*20)
    # print(generate_sobel_kernel(7)[0]*780)
    # print(generate_sobel_kernel(9)[0]*132600)
