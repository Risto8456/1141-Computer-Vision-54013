# Morphology dilation 膨脹

import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Dilation function (binary image)
def dilation(binary_img, kernel=None):
    # 預設值：3x3 正方形結構元素
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad image
    padded = np.pad(binary_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    h, w = binary_img.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Perform dilation
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            if np.any(region[kernel == 1] == 255):  # 只要結構元素區域有白色
                result[i, j] = 255

    return result


# Main：載入影像並進行比較
def main():
    # 讀圖
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_path)
    img_path = os.path.join(parent_dir, "binary.png")

    data = np.fromfile(img_path, dtype=np.uint8)
    color_img = cv.imdecode(data, cv.IMREAD_COLOR)
    binary = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    dilated = dilation(binary)

    # Plot comparison
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Dilation')
    plt.imshow(dilated, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
