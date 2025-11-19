import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 匯入已做好的的模組
from Edge_detection.Sobel import Sobel_operator
from Edge_detection.Prewitt import Prewitt_operator
from Edge_detection.Canny import canny

# 主程式
if __name__ == "__main__":

    cur_path = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(cur_path, "lena.bmp")

    data = np.fromfile(img_path, dtype=np.uint8)
    color = cv.imdecode(data, cv.IMREAD_COLOR)
    img = cv.cvtColor(color, cv.COLOR_BGR2GRAY)

    kernel_size = 3

    sx = Sobel_operator(img, direction='x', ksize=kernel_size)
    sy = Sobel_operator(img, direction='y', ksize=kernel_size)
    absX = cv.convertScaleAbs(sx)
    absY = cv.convertScaleAbs(sy)
    sobel_result = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    
    px = Prewitt_operator(img, direction='x', ksize=kernel_size)
    py = Prewitt_operator(img, direction='y', ksize=kernel_size)
    absX = cv.convertScaleAbs(px)
    absY = cv.convertScaleAbs(py)
    prewitt_result = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    
    canny_result = canny(img)

    plt.figure(figsize=(16,5))

    plt.subplot(1,4,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(sobel_result, cmap='gray')
    plt.title("Sobel")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(prewitt_result, cmap='gray')
    plt.title("Prewitt")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(canny_result, cmap='gray')
    plt.title("Canny")
    plt.axis("off")

    # 儲存結果
    save_path = os.path.join(cur_path, 'Edge_detection_comparison.jpg')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()