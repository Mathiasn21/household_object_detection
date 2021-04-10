# Load coco annotations into dict
# Extrapolate the object polygon information in image
# Paste polygon into new image
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from tools.tools import load_annotations, load_images


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

    return cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def crop_object_from_image(image_src: ndarray, points: ndarray, x, y, width, height):
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


if __name__ == '__main__':
    img_dir = '../data/testing_imgs'
    annotations = load_annotations('../data/testing_annotations/household_objects.json')
    images = load_images(img_dir)
    background_image = cv2.imread('../data/background_imgs/background_1.jpeg')

    segmentations = []
    polygons = []
    segmentation = np.array(annotations['annotations'][0]['segmentation'][0], dtype=np.int32)
    x = round(annotations['annotations'][0]['bbox'][0])
    y = round(annotations['annotations'][0]['bbox'][1])
    width = round(annotations['annotations'][0]['bbox'][2])
    height = round(annotations['annotations'][0]['bbox'][3])
    file_name = annotations['images'][0]['file_name']

    points = np.reshape(segmentation, (-1, 2))
    crop_object_from_image(images[file_name], points, x, y, width, height)

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
    # images = [image_src, background_image]
    # roi = background_image[150: 150 + height, 250:250 + width, :]
    # mask_inv = cv2.bitwise_not(object_mask)

    # result = rotate_image(object_fg, -60)

    # roi_background = cv2.bitwise_and(roi, roi, mask=mask_inv[y:y + height, x:x + width])
    # dst = cv2.add(roi_background, object_fg)

    # background_image[150: 150 + height, 250:250 + width, :] = dst
    """
