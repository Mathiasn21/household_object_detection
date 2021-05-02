import os

import cv2

from tools.tools import load_annotations, load_image, save_annotations
import numpy as np


def resize_annotated_imgs():
    annotations_out = '../data/testing_annotations/labels_household_object_detection_resized.json'
    ann_path = '../data/testing_annotations/labels_household_object_detection_newest.json'
    images_path = '../data/home_interior_images/'
    max_size = 2000

    coco_annotations = load_annotations(ann_path)
    annotations = coco_annotations['annotations']
    images_information = coco_annotations['images']

    # Sort image information
    images_by_name = dict((image_dict['file_name'], image_dict) for image_dict in images_information)

    annotations_by_img_id = {}
    for annotation_dict in annotations:
        key = annotation_dict['image_id']

        if key not in annotations_by_img_id:
            annotations_by_img_id[key] = []
        annotations_by_img_id[key].append(annotation_dict['segmentation'])

    for file_name in os.listdir(images_path):
        full_path = images_path + file_name
        image = load_image(full_path)
        height, width = image.shape[:2]

        if width > max_size or height > max_size:
            width_scalar = max_size / width
            height_scalar = max_size / height

            best_fit_scalar = width_scalar if width_scalar <= height_scalar else height_scalar
            dimensions = (int(width * best_fit_scalar), int(height * best_fit_scalar))

            if file_name in images_by_name:
                # Correct annotations as well
                image_information = images_by_name.get(file_name)
                image_id = image_information['id']
                image_information['width'] = dimensions[1]
                image_information['height'] = dimensions[0]

                if image_id in annotations_by_img_id:
                    for segmentations in annotations_by_img_id[image_id]:
                        for index, segmentation in enumerate(segmentations):
                            segmentations[index] = (np.array(segmentation) * best_fit_scalar).tolist()
                save_annotations(coco_annotations, annotations_out)
            cv2.imwrite(full_path, cv2.resize(image, dimensions))


def resize_bg_imgs():
    max_size = 2000
    images_path = '../data/background_imgs/'

    for file_name in os.listdir(images_path):
        full_path = images_path + file_name
        image = load_image(full_path)
        height, width = image.shape[:2]

        if width > max_size or height > max_size:
            width_scalar = max_size / width
            height_scalar = max_size / height

            best_fit_scalar = width_scalar if width_scalar <= height_scalar else height_scalar
            dimensions = (int(width * best_fit_scalar), int(height * best_fit_scalar))
            cv2.imwrite(full_path, cv2.resize(image, dimensions))


if __name__ == '__main__':
    # resize_annotated_imgs()
    resize_bg_imgs()
