import cv2
image = cv2.imread('Threshold\lena.bmp', cv2.IMREAD_GRAYSCALE)

ret, otsu_threshold = cv2.threshold(
    image,
    0,  # 使用 Otsu 時要忽略的值
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imshow('Adaptive Mean Thresholding', otsu_threshold)
cv2.waitKey(0)