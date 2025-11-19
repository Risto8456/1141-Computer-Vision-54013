# Morphology Closing 閉運算 = Dilation -> Erosion

import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 兼容相對匯入與直接執行
try:
    from .Erosion import erosion
    from .Dilation import dilation
except Exception:
    # 當直接以 python Closing.py 執行（__main__），相對匯入會失敗
    # 此處回退到同資料夾的絕對匯入
    from Erosion import erosion
    from Dilation import dilation

# Closing function
def closing(binary_img, kernel=None):
    # 先膨脹，再侵蝕
    dilated = dilation(binary_img, kernel)
    closed = erosion(dilated, kernel)
    return closed


# Main：載入影像並進行比較
def main():
    # 讀圖
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_path)
    img_path = os.path.join(parent_dir, "binary.png")

    data = np.fromfile(img_path, dtype=np.uint8)
    color_img = cv.imdecode(data, cv.IMREAD_COLOR)
    binary = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    closed = closing(binary)

    # Plot comparison
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Closing')
    plt.imshow(closed, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
