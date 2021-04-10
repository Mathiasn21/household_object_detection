# Load coco annotations into dict
# Extrapolate the object polygon information in image
# Paste polygon into new image
import os
import random
import uuid
from typing import List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from tools.tools import load_annotations, rotate_scale_image, load_image, show_images


def crop_object_from_image(image_src: ndarray, points: ndarray, xywh: List[int], padding=2):
    x, y, width, height = xywh

    width += padding
    height += padding

    # Create mask for object and fill poly shape with white
    object_mask = np.zeros((image_src.shape[0], image_src.shape[1]), dtype=np.uint8)
    cv2.fillPoly(object_mask, [points], 255)

    # Crop to ROI - Region Of Interest
    roi = image_src[y:y + height, x:x + width].copy()
    object_mask = object_mask[y:y + height, x:x + width]

    object_fg = cv2.bitwise_and(roi, roi, mask=object_mask)

    # Set background to color
    background = np.zeros_like(roi, np.uint8)
    cv2.bitwise_not(background, background, mask=object_mask)

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
    return background + object_fg


def augment_image(image: ndarray, directory: str):
    # Probability for each augmentation to happen
    prob_perform_rotation = 0.8
    prob_perform_scaling = 0.3
    image_files_names = []

    for x in range(10):
        rotation = 0
        scale = 1.0

        # Set randomly rotation and scaling
        if random.random() < prob_perform_rotation:
            rotation = random.randrange(10, 360, 5)
            if random.random() < 0.5:
                rotation *= -1

        if random.random() < prob_perform_scaling:
            scale = random.random() + 0.3

        # Perform rotation and scaling on object if defined
        if rotation != 0 or scale != 1.0:
            object_fg = rotate_scale_image(image, rotation, scale=scale)
        else:
            object_fg = image

        file_name = str(uuid.uuid4()).replace('-', '') + '.jpg'
        image_files_names.append(file_name)
        cv2.imwrite(directory + file_name, object_fg)
    return image_files_names


def round_all(numbers):
    return [round(number) for number in numbers]


def generate_objects_from_images(img_dir: str, coco_annotations, categories):
    image_descriptions = coco_annotations['images']
    image_annotations = coco_annotations['annotations']
    clazz_image_information = {x['id']: {'name': x['name'], 'images': []} for x in categories}

    # Sort coco_annotations by image_id
    image_annotations = sorted(image_annotations, key=lambda i: i['image_id'])
    image_annotations = iter(image_annotations)

    for image_desc in image_descriptions:
        image_id: int = image_desc['id']

        image_annotation: Dict = next(image_annotations, {'image_id': -1})
        ann_img_id: int = image_annotation['image_id']

        if ann_img_id == -1:
            break

        file_name = image_desc['file_name']
        image = load_image(img_dir + '/' + file_name)

        while image_id == ann_img_id:
            segmentation = np.array(image_annotation['segmentation'][0], dtype=np.int32)
            xywh = round_all(image_annotation['bbox'])
            category_id = image_annotation['category_id']

            # Reshape from xy to (x, y) coordinates
            points = np.reshape(segmentation, (-1, 2))
            object_fg = crop_object_from_image(image, points, xywh)

            file_new_name = str(uuid.uuid4()).replace('-', '')
            cv2.imwrite('../data/extrapolated_objects/' + file_new_name + '.jpg', object_fg)

            generated_images_names = augment_image(object_fg, '../data/augmented_objects/')
            clazz_image_information[category_id]['images'].extend(generated_images_names)

            image_annotation = next(image_annotations, {'image_id': -1})
            ann_img_id = image_annotation['image_id']
    return clazz_image_information


def embed_object_into_image(object_image, background_image):
    # Create mask
    # background_image = cv2.imread('../data/background_imgs/background_1.jpeg')
    images = [object_image, background_image]
    # roi = background_image[150: 150 + height, 250:250 + width, :]
    # mask_inv = cv2.bitwise_not(object_mask)

    # roi_background = cv2.bitwise_and(roi, roi, mask=mask_inv[y:y + height, x:x + width])
    # dst = cv2.add(roi_background, object_fg)
    gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = thresh.shape[:2]
    roi = background_image[150:150 + height, 250:250 + width].copy()
    mask_inv = cv2.bitwise_not(thresh)

    roi_background = cv2.bitwise_and(roi, roi, mask=thresh)
    dst = cv2.add(roi_background, object_image)

    background_image[150: 150 + height, 250:250 + width, :] = dst
    cv2.imwrite('./testing_embedding.jpg', background_image)

    # draw the contours on the empty image
    # mask = np.zeros(object_image.shape[:2], np.uint8)

    # cv2.drawContours(mask, [contours[1]], -1, 255, 3)

    images.append(thresh)
    images.append(mask_inv)
    images.append(dst)
    images.append(roi)
    images.append(background_image)
    show_images(images)


def generate_images_using(objects_dir: str, images_dir: str, classes, min_images_per_class: int = 10):
    prob_additional_objects = 0.7
    background_images = os.listdir(images_dir)
    numb_backgrounds = len(background_images)

    # Generate 1000 images per class.
    for clazz_id, value in classes.items():
        clazz_images_names = value['images']
        numb_clazz_objects = len(clazz_images_names)

        for i in range(min_images_per_class):
            # Pick random background
            background_image_path = background_images[random.randint(0, numb_backgrounds - 1)]
            # load background image
            background_image = load_image(images_dir + background_image_path)

            # Pick random background
            foreground_object_path = clazz_images_names[random.randint(0, numb_clazz_objects)]
            # load background image
            foreground_object = load_image(images_dir + foreground_object_path)

            embed_object_into_image(foreground_object, background_image)
            # Add additional objects into the image
            while random.random() < prob_additional_objects:
                # Pick random object from class
                embed_object_into_image(foreground_object, background_image)

    # Create annotations for the object
    # Store annotations for that object along with image information


if __name__ == '__main__':
    img_dir = '../data/testing_imgs'
    annotation_dir = '../data/testing_annotations/household_objects.json'
    augmented_objects_dir = '../data/augmented_objects/'
    background_images_dir = '../data/background_imgs/'

    coco_annotations = load_annotations(annotation_dir)
    categories = coco_annotations['categories']

    # generated_images_information = generate_objects_from_images(img_dir, coco_annotations, categories)

    # generate_images_using(augmented_objects_dir, background_images_dir, generated_images_information)

    embed_object_into_image(load_image(
        r'D:\projects_git\image_analysis\household_object_detection\data\augmented_objects\2aabde033038447eae39b980aa0e18eb.jpg'),
                            load_image(
                                r'D:\projects_git\image_analysis\household_object_detection\data\background_imgs\background_1.jpeg'))
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
