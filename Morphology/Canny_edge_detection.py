# Canny edge detection 邊緣偵測

import cv2
import numpy as np

# Load the image
img = cv2.imread('Morphology/lena.bmp')

# Apply Gaussian blur for noise reduction
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred, 50, 150) # Adjust thresholds as needed

cv2.imshow('Original Image', img)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()