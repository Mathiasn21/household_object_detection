import os

import cv2

from tools.tools import load_json_data, load_image, save_json_data, load_config_file
import numpy as np


def calc_new_dimensions(max_size: int, width, height):
    """
    Calculate new minimum dimensions and corresponding scalar
    :param max_size: int
    :param width: int
    :param height: int
    :return: tuple - new dimensions and minimum scalar
    """
    width_scalar = max_size / width
    height_scalar = max_size / height

    best_fit_scalar = min(width_scalar, height_scalar)
    dimensions = (int(width * best_fit_scalar), int(height * best_fit_scalar))
    return dimensions, best_fit_scalar


def resize_segmentations(image_id, annotations_by_img_id, scalar):
    """
    Resize the segmentations such that they match the new image dimensions
    :param image_id: int
    :param annotations_by_img_id: dict - annotations corersponding to image ids
    :param scalar: float - scalar that will be used to alter the segmentations
    """
    for segmentations in annotations_by_img_id[image_id]:
        for index, segmentation in enumerate(segmentations):
            segmentations[index] = (np.array(segmentation) * scalar).tolist()


def resize_annotated_imgs(config):
    """
    resize the annotated images and teh corresponding annotations.
    :param config: dict - script config
    """

    # Extract information from config file
    annotations_out = config['annotations_out']
    ann_path = config['ann_path']
    images_path = config['images_path']
    max_size = config['original_max_size']

    # Load annotations
    coco_annotations = load_json_data(ann_path)
    annotations = coco_annotations['annotations']
    images_information = coco_annotations['images']

    # Sort image information
    images_by_name = dict((image_dict['file_name'], image_dict) for image_dict in images_information)

    # Sort annotations by image id
    annotations_by_img_id = {}
    for annotation_dict in annotations:
        key = annotation_dict['image_id']

        if key not in annotations_by_img_id:
            annotations_by_img_id[key] = []
        annotations_by_img_id[key].append(annotation_dict['segmentation'])

    # Iterate over all images and resize on demand. Also resizes corresponding annotations
    for file_name in os.listdir(images_path):
        full_path = images_path + file_name
        image = load_image(full_path)
        height, width = image.shape[:2]

        if width > max_size or height > max_size:
            dimensions, best_fit_scalar = calc_new_dimensions(max_size, width, height)

            if file_name in images_by_name:
                # Correct annotations as well
                image_information = images_by_name.get(file_name)
                image_id = image_information['id']
                image_information['width'] = dimensions[1]
                image_information['height'] = dimensions[0]

                if image_id in annotations_by_img_id:
                    resize_segmentations(image_id, annotations_by_img_id, best_fit_scalar)

                save_json_data(coco_annotations, annotations_out)
            cv2.imwrite(full_path, cv2.resize(image, dimensions))


def resize_bg_imgs(images_path, max_size):
    """
    Resize the background images
    :param images_path: str: directory path for the background images
    :param max_size: int - max dimension size
    """

    # Iterate over the images and resize on demand
    for file_name in os.listdir(images_path):
        full_path = images_path + file_name
        image = load_image(full_path)
        height, width = image.shape[:2]

        if width > max_size or height > max_size:
            dimensions, best_fit_scalar = calc_new_dimensions(max_size, width, height)
            cv2.imwrite(full_path, cv2.resize(image, dimensions))


if __name__ == '__main__':
    config_path = '../configs/resize_config.yaml'
    config = load_config_file(config_path)

    resize_bg_imgs(config['background_images_path'], config['background_max_size'])
    # resize_annotated_imgs(config)
