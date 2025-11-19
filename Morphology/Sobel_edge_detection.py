# Sobel edge detection 邊緣偵測

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def main():
    """
    Sobel operator
    """
    # 讀取影像
    cur_path = os.path.dirname(os.path.abspath(__file__)) # 目前資料夾
    img_path = os.path.join(cur_path, r"lena.bmp")      # 圖片位置
    
    # 用 fromfile 讀取二進位資料，支援中文路徑
    data = np.fromfile(img_path, dtype=np.uint8)
    # 用 imdecode 解碼成 OpenCV 影像
    gray_img = cv.imdecode(data, cv.IMREAD_COLOR)

    # 初始化輸出影像
    plt.figure(figsize=(15,10))

    # 嘗試不同 size 的 mask
    for n in range(1, 4):
        # 使用 sobel 算子
        # 第一個參數是要作用的影像
        # 第二個參數是影像深度 使用 16 可避免 overflow 問題
        # 第三 & 四個參數是控制是否使用兩種面罩的參數
        # 第五個參數是 mask size
        kernel_size = 1+(n*2)
        x = cv.Sobel(gray_img, cv.CV_16S, 1, 0, ksize=kernel_size) 
        y = cv.Sobel(gray_img, cv.CV_16S, 0, 1, ksize=kernel_size)

        # 轉換為影像原本儲存的格式 uint8
        absX = cv.convertScaleAbs(x) 
        absY = cv.convertScaleAbs(y)

        # 將兩個軸向的測邊結果相加，形成完整輪廓
        dst = cv.addWeighted(absX, 0.5, absX,0.5,0)

        plt.subplot(330+n+(2*n)-2)
        plt.imshow(absX, cmap='gray')
        plt.title('Vertical line result  (mask size = ' + str(kernel_size) + ')')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(330+n+(2*n)-1)
        plt.imshow(absY, cmap='gray')
        plt.title('Horizontal line result (mask size = ' + str(kernel_size) + ')')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(330+n+(2*n))
        plt.imshow(dst, cmap='gray')
        plt.title('Complete result (mask size = ' + str(kernel_size) + ')')
        plt.xticks([])
        plt.yticks([])

    # 儲存所有輸出的圖像
    Sobel_result_path = os.path.join(cur_path, 'Sobel_operator_result.jpg')
    plt.savefig(Sobel_result_path, bbox_inches='tight', dpi=300)
    plt.tight_layout()  # 自動調整間距
    plt.show()

if __name__ == "__main__":
    main()