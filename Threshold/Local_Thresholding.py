import cv2
image = cv2.imread('Threshold\lena.bmp', cv2.IMREAD_GRAYSCALE)

window_size = 5
c = 2

binary_image = cv2.adaptiveThreshold(
    image,
    255,                        # 最大值，閾值化的像素設置為此值
    cv2.ADAPTIVE_THRESH_MEAN_C, # 閾值類型：局部均值閾值法
    cv2.THRESH_BINARY,          # 閾值應用類型：二值化
    window_size,                # 計算閾值時考慮的鄰域大小（必須是奇數）
    c                           # 常數，從計算出的均值中減去的值
)

cv2.imshow('Adaptive Mean Thresholding', binary_image)
cv2.waitKey(0)