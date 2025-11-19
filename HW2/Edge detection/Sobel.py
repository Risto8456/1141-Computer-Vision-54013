# Sobel edge detection 
# Sobel 邊緣偵測，自製 Sobel

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 自製 Sobel 運算函式
def Sobel_operator(img, direction='x', ksize=3):
    """
    自己實作 Sobel 運算
    img: 單通道灰階影像
    direction: 'x' 或 'y'
    ksize: 3, 5, 7 (奇數)
    """

    # Sobel kernel 尺寸 (ksize = 3,5,7 對應 3x3、5x5、7x7)
    real_kernel = ksize if ksize > 1 else 3

    # 生成 Sobel kernels
    if real_kernel == 3:
        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)

        Ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)

    elif real_kernel == 5:
        # 5x5 Sobel kernel（常用版本）
        Kx = np.array([
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
            [-4, -2, 0, 2, 4],
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2]
        ], dtype=np.float32)

        Ky = Kx.T

    elif real_kernel == 7:
        # 7x7 Sobel kernel（擴展版，類似 Gaussian 加梯度）
        # 這種 kernel 可自行設計或查表，這裡給常用版本
        g = np.array([1, 2, 3, 0, -3, -2, -1], dtype=np.float32)
        Kx = np.outer(np.ones(7), g)
        Ky = Kx.T

    else:
        raise ValueError("Unsupported Sobel size")

    # 選方向
    if direction == 'x':
        kernel = Kx
    else:
        kernel = Ky

    # convolution（零填充）
    pad = real_kernel // 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode='constant')

    H, W = img.shape
    output = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            region = padded[i:i+real_kernel, j:j+real_kernel]
            output[i, j] = np.sum(region * kernel)

    return output


def Sobel_edge_detection_test(img):
    plt.figure(figsize=(15, 10))

    # 嘗試 mask size = 3, 5, 7
    for n in range(1, 4):
        kernel_size = 1 + (n * 2)

        # 使用自製 Sobel 函式
        sx = Sobel_operator(img, direction='x', ksize=kernel_size)
        sy = Sobel_operator(img, direction='y', ksize=kernel_size)

        # 轉成 uint8 顯示
        absX = cv.convertScaleAbs(sx)
        absY = cv.convertScaleAbs(sy)
        dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

        # 顯示
        idx = (n-1) * 3 + 1

        plt.subplot(3, 3, idx)
        plt.imshow(absX, cmap='gray')
        plt.title(f'Vertical (ksize={kernel_size})')
        plt.axis('off')

        plt.subplot(3, 3, idx + 1)
        plt.imshow(absY, cmap='gray')
        plt.title(f'Horizontal (ksize={kernel_size})')
        plt.axis('off')

        plt.subplot(3, 3, idx + 2)
        plt.imshow(dst, cmap='gray')
        plt.title(f'Complete (ksize={kernel_size})')
        plt.axis('off')

    # 儲存結果
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
