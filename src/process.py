import cv2
import numpy as np


def process_predicted_mask(predicted_mask):
    """
    Cleans up a blurry, grayscale predicted mask using a series of image processing steps.

    Args:
        predicted_mask (np.ndarray): A grayscale image (as a NumPy array) representing
                                     the raw output from a segmentation model.
                                     Values should be in the range [0, 255].

    Returns:
        np.ndarray: A clean, binary (black and white) mask.
    """
    # Ensure the input is an 8-bit grayscale image
    if predicted_mask.dtype != np.uint8:
        # Assuming the input is float [0, 1], scale to [0, 255] and convert
        if predicted_mask.max() <= 1.0:
            predicted_mask = (predicted_mask * 255).astype(np.uint8)
        else:
            predicted_mask = predicted_mask.astype(np.uint8)

    # 1. Denoise using a Gaussian Blur
    # This smooths the image and helps create more contiguous regions.
    # The kernel size (e.g., (5, 5)) can be tuned. A larger kernel means more blur.
    blurred_mask = cv2.GaussianBlur(predicted_mask, (5, 5), 0)

    # 2. Binarize the image using Otsu's Thresholding
    # This automatically finds the best threshold value to separate foreground and background.
    # The result is a sharp, black and white image.
    _, binary_mask = cv2.threshold(blurred_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Refine the shape using Morphological Operations
    # Define a kernel (structuring element). A 5x5 kernel is a good starting point.
    kernel = np.ones((5, 5), np.uint8)

    # Opening: Removes small, isolated white noise pixels (erosion followed by dilation)
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Closing: Fills small black holes within the main white area (dilation followed by erosion)
    final_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return final_mask
