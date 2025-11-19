# Morphology Opening 開運算 = Erosion -> Dilation

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
    
# Opening function
# 先侵蝕，再膨脹
def opening(binary_img, kernel=None):
    eroded = erosion(binary_img, kernel)
    opened = dilation(eroded, kernel)
    return opened


def main():
    # 讀圖
    cur_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_path)
    img_path = os.path.join(parent_dir, "binary.png")

    data = np.fromfile(img_path, dtype=np.uint8)
    color_img = cv.imdecode(data, cv.IMREAD_COLOR)
    binary = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    opened_img = opening(binary)

    # Plot comparison
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Opening')
    plt.imshow(opened_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
