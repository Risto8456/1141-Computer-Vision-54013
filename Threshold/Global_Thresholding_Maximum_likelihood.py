# need debug
import cv2
import numpy as np

def maximum_likelihood_thresholding(image):
    # Convert image to grayscale if it is not
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Flatten the image to 1D array
    pixel_values = image.flatten()
    
    # Calculate the histogram
    histogram, bin_edges = np.histogram(pixel_values, bins=256, range=(0, 256))
    
    # Normalize the histogram
    histogram = histogram / float(np.sum(histogram))
    
    # Initialize variables
    max_likelihood = -np.inf
    optimal_threshold = -1
    epsilon = 1e-10
    
    for threshold in range(1, 256):
        # Split the histogram into two classes
        w0 = np.sum(histogram[:threshold])
        w1 = np.sum(histogram[threshold:])
        if w0 == 0 or w1 == 0:
            continue
        mean0 = np.sum(np.arange(0, threshold) * histogram[:threshold]) / w0
        mean1 = np.sum(np.arange(threshold, 256) * histogram[threshold:]) / w1
        variance0 = np.sum(((np.arange(0, threshold) - mean0) ** 2) * histogram[:threshold]) / w0
        variance1 = np.sum(((np.arange(threshold, 256) - mean1) ** 2) * histogram[threshold:]) / w1
        
        within_class_likelihood = w0 * np.log(variance0 + epsilon) + w1 * np.log(variance1 + epsilon)
        
        if within_class_likelihood > max_likelihood:
            max_likelihood = within_class_likelihood
            optimal_threshold = threshold
    
    return optimal_threshold

image = cv2.imread('Threshold\lena.bmp', cv2.IMREAD_GRAYSCALE)

Maximum_likelihood = maximum_likelihood_thresholding(image)
print("Best threshold =", Maximum_likelihood)

_, binary = cv2.threshold(
    image, 
    Maximum_likelihood, 
    255, 
    cv2.THRESH_BINARY
)
cv2.imshow('Adaptive Mean Thresholding', binary)
cv2.waitKey(0)