import json
import os
from typing import Dict, List

import cv2
from numpy import ndarray
import numpy as np

import matplotlib.pyplot as plt


def load_annotations(file_path: str) -> Dict:
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def save_annotations(annotations, file_path: str):
    json_formatted_annotations = json.dumps(annotations)
    with open(file_path, 'w+') as json_file:
        json_file.write(json_formatted_annotations)


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


def split_images_annotations(image_information, annotations, coco_annotations, source_dir, target_dir):
    # Split annotations and image_information into train, validation, test sets
    validation_split = 0.20
    test_split = 0.10
    train_split = 1 - validation_split - test_split
    num_rows = len(image_information)
    train_split_index = int(np.floor((num_rows - 1) * train_split)) + 1
    test_split_index = int(np.floor((num_rows - 1) * (train_split + test_split))) + 1

    train_image_information = image_information[0:train_split_index]
    test_image_information = image_information[train_split_index:test_split_index]
    val_image_information = image_information[test_split_index:]

    annotations_iterator = iter(annotations)
    images_to_look_for = [test_image_information[0]['id'], val_image_information[0]['id']]
    split_indexes = []

    i = 0
    j = 0
    while i < 2:
        annotation = next(annotations_iterator)
        if images_to_look_for[i] == annotation['image_id']:
            i += 1
            split_indexes.append(j)
        j += 1

    train_annotations = annotations[0:split_indexes[0]]
    test_annotations = annotations[split_indexes[0]:split_indexes[1]]
    val_annotations = annotations[split_indexes[1]:]

    partial_f_name = '_coco_annotations.json'

    setup_dirs([target_dir + 'data/images/train/', target_dir + 'data/images/test/',
                target_dir + 'data/images/val/', target_dir + 'data/labels/'])

    coco_annotations['images'] = train_image_information
    coco_annotations['annotations'] = train_annotations
    save_annotations(coco_annotations, target_dir + 'data/labels/train' + partial_f_name)
    move_files(source_dir, target_dir + 'data/images/train/', train_image_information)

    coco_annotations['images'] = test_image_information
    coco_annotations['annotations'] = test_annotations
    save_annotations(coco_annotations, target_dir + 'data/labels/test' + partial_f_name)
    move_files(source_dir, target_dir + 'data/images/test/', test_image_information)

    coco_annotations['images'] = val_image_information
    coco_annotations['annotations'] = val_annotations
    save_annotations(coco_annotations, target_dir + 'data/labels/val' + partial_f_name)
    move_files(source_dir, target_dir + 'data/images/val/', val_image_information)


def setup_dirs(dirs: list):
    for path in dirs:
        os.makedirs(path, exist_ok=True)


def clear_directory_contents(directories: list):
    for directory in directories:
        for file in os.listdir(directory):
            os.remove(directory + file)


def move_files(source_dir: str, target_dir: str, files):
    for file in files:
        file_name = file['file_name']
        os.replace(source_dir + file_name, target_dir + file_name)


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
