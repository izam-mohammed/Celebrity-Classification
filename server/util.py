"""
Image Classification Module

This module provides functions for classifying images using a pre-trained model. It includes functions to load saved
artifacts, classify images, convert base64-encoded images to OpenCV format, and crop images based on eye detection.

Functions:
- classify_image(image_base64_data, file_path=None): Classify an image and return classification results.
- class_number_to_name(class_num): Get the class name from the class number.
- load_saved_artifacts(): Load saved class mappings and the pre-trained model.
- get_cv2_image_from_base64_string(b64str): Convert a base64-encoded image string to a cv2 image.
- get_cropped_image_if_2_eyes(image_path, image_base64_data): Crop an image if two eyes are visible.

Globals:
- __class_name_to_number: Mapping of class names to class numbers.
- __class_number_to_name: Mapping of class numbers to class names.
- __model: Pre-trained classification model.

Requirements:
- joblib
- json
- numpy
- base64
- cv2 (OpenCV)
- wavelet (Custom module for wavelet transformation)
- pickle

Note:
Ensure that the required model artifacts and cascade classifiers are available in the 'artifacts' and 'opencv/haarcascades' directories.
"""

import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import pickle

__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def classify_image(image_base64_data, file_path=None):
    """
    Classify an image using a pre-trained model.
    
    Args:
        image_base64_data (str): Base64 encoded image data.
        file_path (str, optional): File path to an image (default is None).

    Returns:
        list: List of dictionaries containing classification results.
    """
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, "db1", 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack(
            (scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1))
        )

        len_image_array = 32 * 32 * 3 + 32 * 32

        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append(
            {
                "class": class_number_to_name(__model.predict(final)[0]),
                "class_probability": np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
                "class_dictionary": __class_name_to_number,
            }
        )

    return result


def class_number_to_name(class_num):
    """
    Get the class name from the class number.
    
    Args:
        class_num (int): Class number.

    Returns:
        str: Class name.
    """
    return __class_number_to_name[class_num]


def load_saved_artifacts():
    """
    Load saved artifacts including class mappings and the model.
    """
    print("Loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./server/artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open("./server/artifacts/saved_model.pickle", "rb") as f:
            __model = pickle.load(f)
    print("Loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    """
    Convert a base64 encoded image string to a cv2 image.

    Args:
        b64str (str): Base64 encoded image data.

    Returns:
        cv2 image: Image in cv2 format.
    """
    encoded_data = b64str.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    """
    Crop the image if two eyes are visible.

    Args:
        image_path (str): File path to an image.
        image_base64_data (str): Base64 encoded image data.

    Returns:
        list: List of cropped face images.
    """
    face_cascade = cv2.CascadeClassifier(
        "./server/opencv/haarcascades/haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier("./server/opencv/haarcascades/haarcascade_eye.xml")

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces
