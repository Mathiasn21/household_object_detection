# Load coco annotations into dict
# Extrapolate the object polygon information in image
# Paste polygon into new image
from typing import List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from tools.tools import load_annotations, rotate_scale_image, load_image


def crop_object_from_image(image_src: ndarray, points: ndarray, xywh: List[int]):
    x, y, width, height = xywh

    # Create mask for object
    object_mask = np.zeros((image_src.shape[0], image_src.shape[1]), dtype=np.uint8)
    cropped = image_src[y:y + height, x:x + width].copy()

    cv2.fillPoly(object_mask, [points], 255)
    object_fg = cv2.bitwise_and(cropped, cropped, mask=object_mask[y:y + height, x:x + width])

    bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=object_mask[y:y + height, x:x + width])
    dst2 = bg + object_fg

    object_fg_white = rotate_scale_image(dst2, -60)

    cv2.imwrite('testing_cropping_object.jpg', object_fg_white)


def round_all(numbers):
    return [round(number) for number in numbers]


def extract_objects_from_images(img_dir: str, annotation_dir: str):
    coco_annotations = load_annotations(annotation_dir)
    image_descriptions = coco_annotations['images']

    image_annotations = coco_annotations['annotations']

    # Sort coco_annotations by image_id
    image_annotations = sorted(image_annotations, key=lambda i: i['image_id'])
    image_annotations = iter(image_annotations)

    for image_desc in image_descriptions:
        image_id: int = image_desc['id']

        image_annotation: Dict = next(image_annotations, {'image_id': -1})
        ann_img_id: int = image_annotation['image_id']
        if ann_img_id == -1:
            break

        while image_id == ann_img_id:
            segmentation = np.array(image_annotation['segmentation'][0], dtype=np.int32)
            xywh = round_all(image_annotation['bbox'])

            file_name = coco_annotations['images'][0]['file_name']

            image = load_image(img_dir + '/' + file_name)
            points = np.reshape(segmentation, (-1, 2))
            crop_object_from_image(image, points, xywh)

            image_annotation = next(image_annotations, default={'image_id': -1})
            ann_img_id = image_annotation['image_id']


if __name__ == '__main__':
    img_dir = '../data/testing_imgs'
    annotations_dir = '../data/testing_annotations/household_objects.json'
    extract_objects_from_images(img_dir, annotations_dir)

    """
    segmentation = annotations['annotations'][0]['segmentation'][0]
    np.reshape(segmentation, (-1, 2))
    poly = Polygon(contour)
    poly = poly.simplify(1.0, preserve_topology=False)
    polygons.append(poly)
    segmentation2 = np.array(poly.exterior.coords).ravel().tolist()

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area
    """

    """
    # background_image = cv2.imread('../data/background_imgs/background_1.jpeg')
    # images = [image_src, background_image]
    # roi = background_image[150: 150 + height, 250:250 + width, :]
    # mask_inv = cv2.bitwise_not(object_mask)

    # result = rotate_image(object_fg, -60)

    # roi_background = cv2.bitwise_and(roi, roi, mask=mask_inv[y:y + height, x:x + width])
    # dst = cv2.add(roi_background, object_fg)

    # background_image[150: 150 + height, 250:250 + width, :] = dst
    """
