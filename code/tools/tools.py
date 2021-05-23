import json
import os
import random
import shutil
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from numpy import ndarray


def load_json_data(file_path: str) -> Dict:
    """
    Load json data from a file
    :param file_path:
    :return: Dict - data dict
    """
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def save_json_data(data, file_path: str):
    """
    Store json data in file.
    :param data: Dict - data dict to store
    :param file_path: str - output file, will overwrite the content
    """
    json_formatted_data = json.dumps(data)
    with open(file_path, 'w+') as json_file:
        json_file.write(json_formatted_data)


def generate_path_dict(images_path, labels_path, partial_f_name):
    """
    Generate split data path dict, also contains the corresponding label directories.
    :param images_path: str - image top path
    :param labels_path: str - label top path
    :param partial_f_name: str - ending annotation name
    :return: Dict - resulting path dict
    """
    path_dict = {
        'images': {'train': images_path + 'train/',
                   'test': images_path + 'test/',
                   'val': images_path + 'val/'},

        'labels': {'train': labels_path + 'train' + partial_f_name,
                   'test': labels_path + 'test' + partial_f_name,
                   'val': labels_path + 'val' + partial_f_name},
        'labels_path': labels_path
    }
    return path_dict


def dump_config_file(config_path: str, config: Dict):
    """
    Dump a YAML config into a file.
    :param config_path: str
    :param config: Dict
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=True)


def load_config_file(file_path: str) -> Dict:
    """
    Load a YAML config file. Uses UnsafeLoader
    :rtype: Dict
    """
    with open(file_path, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.UnsafeLoader)
    return cfg


def load_image(file_path: str):
    """
    Load image in ndarray from file path
    :param file_path: str - file path
    :return: ndarray - image
    """
    return cv2.imread(file_path)


def show_images(images: List[ndarray]) -> None:
    """
    Plot images.
    :param images: List[ndarray] - list of images to plot
    """
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i])

    plt.show(block=True)


def split_background_images(source, target_dict: dict):
    """
    Split background images based on a source directory and a target dictionary. Files will be copied.
    Hence, keeping the originals.
    :param source: str - source path
    :param target_dict: dict - Containing all train, test and validation paths
    """
    backgrounds = os.listdir(source)
    random.shuffle(backgrounds)

    validation_split = 0.20
    test_split = 0.10
    train_split = 1 - validation_split - test_split

    num_rows = len(backgrounds)
    train_split_index = int(np.floor((num_rows - 1) * train_split)) + 1
    test_split_index = int(np.floor((num_rows - 1) * (train_split + test_split))) + 1

    train_backgrounds = backgrounds[0:train_split_index]
    test_backgrounds = backgrounds[train_split_index:test_split_index]
    val_backgrounds = backgrounds[test_split_index:]

    setup_dirs(list(target_dict.values()))
    copy_files(source, target_dict['train'], train_backgrounds)
    copy_files(source, target_dict['test'], test_backgrounds)
    copy_files(source, target_dict['val'], val_backgrounds)


def split_images_annotations(coco_annotations, source_dir, path_dict):
    """
    Split source images and corresponding annotations into train, test and validation sets.
    Will only copy and not delete the source material.
    :param coco_annotations: dict - coco dict
    :param source_dir: str - source directory containing the images to split
    :param path_dict: dict - train, test and validation directories.
    """

    # Split annotations and image_information into train, validation, test sets
    validation_split = 0.20
    test_split = 0.10
    train_split = 1 - validation_split - test_split
    image_information = coco_annotations['images']
    annotations = coco_annotations['annotations']

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

    # Setup directories on demand
    setup_dirs([path_dict['images']['train'], path_dict['images']['test'],
                path_dict['images']['val'], path_dict['labels_path']])

    # Store new annotation and image information
    coco_annotations['images'] = train_image_information
    coco_annotations['annotations'] = train_annotations
    save_json_data(coco_annotations, path_dict['labels']['train'])
    copy_files(source_dir, path_dict['images']['train'], train_image_information)

    coco_annotations['images'] = test_image_information
    coco_annotations['annotations'] = test_annotations
    save_json_data(coco_annotations, path_dict['labels']['test'])
    copy_files(source_dir, path_dict['images']['test'], test_image_information)

    coco_annotations['images'] = val_image_information
    coco_annotations['annotations'] = val_annotations
    save_json_data(coco_annotations, path_dict['labels']['val'])
    copy_files(source_dir, path_dict['images']['val'], val_image_information)


def setup_dirs(dirs: list):
    """
    Create directories if needed.
    :param dirs: list
    """
    for path in dirs:
        os.makedirs(path, exist_ok=True)


def clear_directory_contents(directories: list):
    """
    Remove contents in the directories. ignores any errors
    :param directories: list - directories
    """
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)


def move_files(source_dir: str, target_dir: str, files):
    """
    Move files from a source directory to a target directory. Uses os.replace
    :param source_dir: str - source dir
    :param target_dir: str - target dir
    :param files: list - files to move
    """
    for file in files:
        file_name = file['file_name']
        os.replace(source_dir + file_name, target_dir + file_name)


def copy_files(source_dir: str, target_dir: str, files):
    """
    Copy file from source dir to target dir.
    :param source_dir: str - source dir
    :param target_dir: str - target dir
    :param files: list - files to move
    """
    for file in files:
        file_name = file['file_name'] if 'file_name' in file else file
        shutil.copy2(source_dir + file_name, target_dir + file_name)


def rotate_scale_image(image: ndarray, angle: int, scale=1.0):
    """
    Rotate and scale the image using cv2.affine transformation
    :param image: ndarray - image to apply transformation
    :param angle: int - angle
    :param scale: float - scale of the resulting image
    :return:
    """
    height, width = image.shape[:2]

    image_center = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center and add the new image center coordinates
    rot_mat[0, 2] += bound_w / 2 - image_center[0]
    rot_mat[1, 2] += bound_h / 2 - image_center[1]
    color = (0, 0, 0)

    return cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=color)
