import json
import os
from typing import Dict

import cv2
from numpy import ndarray


def load_annotations(file_path: str) -> Dict:
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def load_images(directory: str) -> Dict:
    data = {}
    filename: str
    for filename in os.listdir(directory):
        # Read and add image to dictionary
        data[filename] = cv2.imread(directory + '/' + filename)
    return data

