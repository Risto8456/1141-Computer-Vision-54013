# Prewitt edge detection
# Prewitt 邊緣偵測

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 產生 Prewitt kernel（任意奇數尺寸）
def generate_prewitt_kernel(size):
    assert size % 2 == 1, "Kernel size 必須是奇數"
    half = size // 2

    # 產生坐標 [-half, ..., half]
    ys, xs = np.indices((size, size), dtype=np.int32)
    xs = xs - half
    ys = ys - half

    # Gx：左右方向梯度 = x 的符號
    Gx = np.sign(xs).astype(np.float32)

    # Gy：上下方向梯度 = y 的符號
    Gy = np.sign(ys).astype(np.float32)

    # 中心軸線（x = 0、y = 0）保持 0 → Prewitt 特性
    Gx[:, half] = 0
    Gy[half, :] = 0

    return Gx, Gy

# 自製 Prewitt 運算
def Prewitt_operator(img, direction='x', ksize=3):
    """
    img: 單通道灰階影像
    direction: 'x' or 'y'
    ksize: odd number (3, 5, 7, 9 ...)
    """
    Gx, Gy = generate_prewitt_kernel(ksize)
    kernel = Gx if direction == 'x' else Gy

    pad = ksize // 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode='constant')

    H, W = img.shape
    output = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            region = padded[i:i+ksize, j:j+ksize]
            output[i, j] = np.sum(region * kernel)

    return output

# Prewitt edge detection 測試
def Prewitt_edge_detection_test(img):
    plt.figure(figsize=(15, 12))

    s = 1
    e = 5   # 產生 3,5,7,9 四種 kernel size
    for n in range(s, e):
        kernel_size = 1 + (n * 2)

        px = Prewitt_operator(img, direction='x', ksize=kernel_size)
        py = Prewitt_operator(img, direction='y', ksize=kernel_size)

        absX = cv.convertScaleAbs(px)
        absY = cv.convertScaleAbs(py)
        dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

        idx = (n-1) * 3 + 1

        plt.subplot(e-s+1, 3, idx)
        plt.imshow(absX, cmap='gray')
        plt.title(f'Prewitt Vertical (ksize={kernel_size})')
        plt.axis('off')

        plt.subplot(e-s+1, 3, idx + 1)
        plt.imshow(absY, cmap='gray')
        plt.title(f'Prewitt Horizontal (ksize={kernel_size})')
        plt.axis('off')

        plt.subplot(e-s+1, 3, idx + 2)
        plt.imshow(dst, cmap='gray')
        plt.title(f'Prewitt Complete (ksize={kernel_size})')
        plt.axis('off')

    # 儲存
    cur_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(cur_path, "Prewitt_operator_custom.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


# 主程式
if __name__ == "__main__":
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_path)
    img_path = os.path.join(parent_dir, "lena.bmp")

    data = np.fromfile(img_path, dtype=np.uint8)
    color_img = cv.imdecode(data, cv.IMREAD_COLOR)
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    Prewitt_edge_detection_test(gray_img)

    # Prewitt kernel
    # print(generate_prewitt_kernel(3)[0])
    # print(generate_prewitt_kernel(5)[0])
    # print(generate_prewitt_kernel(7)[0])
    # print(generate_prewitt_kernel(9)[0])