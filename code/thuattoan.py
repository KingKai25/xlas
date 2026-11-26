import cv2
import numpy as np

def _apply_threshold(edge_magnitude, thresh=None):
    """Nếu thresh=None thì dùng Otsu, ngược lại cắt theo ngưỡng thủ công"""
    abs_edge = cv2.convertScaleAbs(edge_magnitude)
    if thresh is None:
        _, binary_edge = cv2.threshold(abs_edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary_edge = cv2.threshold(abs_edge, thresh, 255, cv2.THRESH_BINARY)
    return binary_edge

def sobel(processed_img, thresh=None):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float32)
    sobel_x = cv2.filter2D(processed_img, cv2.CV_64F, kernel_x)
    sobel_y = cv2.filter2D(processed_img, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    return _apply_threshold(magnitude, thresh)

def prewitt(processed_img, thresh=None):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    prewitt_x = cv2.filter2D(processed_img, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(processed_img, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
    return _apply_threshold(magnitude, thresh)

def laplace(processed_img, thresh=None):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    laplace_img = cv2.filter2D(processed_img, cv2.CV_64F, kernel)
    return _apply_threshold(laplace_img, thresh)

def canny(processed_img, threshold1=None, threshold2=None):
    kernel_gauss_159 = np.array([
        [2,  4,  5,  4, 2],
        [4,  9, 12,  9, 4],
        [5, 12, 15, 12, 5],
        [4,  9, 12,  9, 4],
        [2,  4,  5,  4, 2]
    ], dtype=np.float32) / 159.0
    smoothed = cv2.filter2D(processed_img, -1, kernel_gauss_159)

    if threshold1 is None or threshold2 is None:
        high_thresh, _ = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = 0.5 * high_thresh
        return cv2.Canny(smoothed, low_thresh, high_thresh)

    return cv2.Canny(smoothed, threshold1, threshold2)