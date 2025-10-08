import cv2
import numpy as np
import matplotlib.pyplot as plt
# pip install matplotlib

# 讀取灰階影像
img = cv2.imread(r"Morphology\lena.bmp", 0)
if img is None:
    raise FileNotFoundError("讀取影像失敗，請檢查路徑是否正確")

# 定義結構元素 (5x5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 各種形態學操作
eroded   = cv2.erode(img, kernel, iterations=1)
dilated  = cv2.dilate(img, kernel, iterations=1)
opening  = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# 結果與標籤
images = [img, eroded, dilated, opening, closing, gradient]
titles = ["Original", "Eroded", "Dilated", "Opening", "Closing", "Gradient"]

# 建立圖形
plt.figure(figsize=(12, 8))  # 調整整體大小

for i in range(len(images)):
    plt.subplot(2, 3, i+1)             # 2列3欄
    plt.imshow(images[i], cmap='gray') # 灰階顯示
    plt.title(titles[i], fontsize=14)  # 標題
    plt.axis('off')                    # 隱藏座標軸

plt.tight_layout()  # 自動調整間距
plt.show()
