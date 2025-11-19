# Morphology Comparison
# 比較: Original, Erosion, Dilation, Opening, Closing

import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 匯入 Morphology 資料夾中的 4 個函式
from Morphology.Erosion import erosion
from Morphology.Dilation import dilation
from Morphology.Opening import opening
from Morphology.Closing import closing


def main():
    # 取得路徑
    cur_path = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(cur_path, "binary.png")

    # 讀圖（避免中文路徑問題）
    data = np.fromfile(img_path, dtype=np.uint8)
    color_img = cv.imdecode(data, cv.IMREAD_COLOR)
    binary = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    # 執行 morphological operations
    eroded = erosion(binary)
    dilated = dilation(binary)
    opened = opening(binary)
    closed = closing(binary)

    # Plot 2 row × 3 columns
    plt.figure(figsize=(10, 4))

    titles = ["Original", "Erosion", "Dilation", "Opening", "Closing"]
    images = [binary, eroded, dilated, opened, closed]

    for i, (title, img) in enumerate(zip(titles, images), 1):
        plt.subplot(2, 3, i)
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    # 儲存
    save_path = os.path.join(cur_path, "Morphology_comparison.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
