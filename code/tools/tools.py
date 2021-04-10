import json
import os
from typing import Dict, List

import cv2
from numpy import ndarray

import matplotlib.pyplot as plt


def load_annotations(file_path: str) -> Dict:
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def load_image(file_path: str):
    return cv2.imread(file_path)


def show_images(images: List[ndarray]) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i])

    plt.show(block=True)


def rotate_scale_image(image: ndarray, angle: int, scale=1.0):
    height, width = image.shape[:2]

    image_center = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rot_mat[0, 2] += bound_w / 2 - image_center[0]
    rot_mat[1, 2] += bound_h / 2 - image_center[1]
    color = (0, 0, 0)

    return cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=color)