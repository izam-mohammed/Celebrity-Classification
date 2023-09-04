"""
Wavelet Transformation Module

This module provides a function for applying a 2D wavelet transform to an image.

Functions:
- w2d(img, mode="haar", level=1): Apply 2D wavelet transform to an image.

Usage:
1. Import this module.
2. Use the 'w2d' function to apply a 2D wavelet transform to an image.

Parameters:
- img (numpy.ndarray): Input image as a NumPy array.
- mode (str, optional): Wavelet transform mode (default is "haar").
- level (int, optional): Number of decomposition levels (default is 1).

Returns:
- numpy.ndarray: Transformed image as a NumPy array.

Requirements:
- numpy
- pywavelets (pywt)
- cv2 (OpenCV)
"""

import numpy as np
import pywt
import cv2

def w2d(img, mode="haar", level=1):
    """
    Apply 2D wavelet transform to an image.

    Args:
        img (numpy.ndarray): Input image as a NumPy array.
        mode (str, optional): Wavelet transform mode (default is "haar").
        level (int, optional): Number of decomposition levels (default is 1).

    Returns:
        numpy.ndarray: Transformed image as a NumPy array.
    """
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H
