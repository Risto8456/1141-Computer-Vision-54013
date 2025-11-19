# Probabilistic Hough Transform 機率霍夫變換
# Hough Transform 霍夫變換

import matplotlib.pyplot as plt
from skimage import data
import cv2
import numpy as np

src = data.camera()
dst = cv2.Canny(src, 50, 200, None, 3)
cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1)

plt.imshow(cdstP)
plt.title("Detected Lines")
plt.show()