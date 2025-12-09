import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_threshold(image, threshold_value):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary

def apply_median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_morphological_operations(image, operation, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'Erosión':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'Dilatación':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'Apertura':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'Cierre':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def calculate_histogram(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(hist)
    ax.set_title('Histograma')
    ax.set_xlabel('Intensidad')
    ax.set_ylabel('Frecuencia')
    fig.tight_layout()
    return fig
