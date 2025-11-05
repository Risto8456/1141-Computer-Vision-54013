# Hough Transform 霍夫變換
# 偵測圓形

from skimage import data
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 載入灰階影像
img_gray = data.coins()

# 使用 HoughCircles 檢測圓形
circles = cv2.HoughCircles(
    img_gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=20,
    maxRadius=50
)

# 顯示結果
output = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

plt.imshow(output)
plt.title("Detected Coins")
plt.axis("off")
plt.show()
