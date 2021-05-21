# Load coco annotations into dict
# Extrapolate the object polygon information in image
# Paste polygon into new image
import os
import random
import uuid
from typing import List, Dict

import cv2
import numpy as np
import torchvision
from numpy import ndarray
from randomdict import RandomDict
from shapely.geometry import Polygon, box
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomErasing, \
    RandomPerspective, RandomSolarize, ColorJitter, transforms

from tools.tools import load_annotations, load_image, save_annotations, split_images_annotations, \
    split_background_images, setup_dirs, clear_directory_contents, generate_path_dict, load_config_file


def crop_object_from_image(image_src: ndarray, points: ndarray, xywh: List[int], padding=1):
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
    return cv2.bitwise_and(background, object_fg)


def augment_image(image: ndarray, directory: str, num_augmentations=50):
    # Probability for each augmentation to happen
    image_files_names = []
    augmentations = [RandomHorizontalFlip(),
                     RandomVerticalFlip(),
                     RandomRotation((-360, 360)),
                     # RandomErasing(scale=(0.02, 0.15), p=0.3),
                     RandomPerspective(distortion_scale=0.65),
                     ColorJitter(brightness=0.0, contrast=0.0, saturation=0.5, hue=0.5),
                     RandomSolarize(0.4, p=0.2)]

    aug_transformations = transforms.Compose(augmentations)

    for x in range(num_augmentations):
        rotation = 0
        scale = 1.0

        # Set randomly rotation and scaling
        # if random.random() < prob_perform_rotation:
        #    rotation = random.randrange(10, 360, 5)
        #    if random.random() < 0.5:
        #        rotation *= -1

        # if random.random() < prob_perform_scaling:
        #    scale = random.random() + 0.5

        # Perform rotation and scaling on object if defined
        # if rotation != 0 or scale != 1.0:
        #    object_fg = rotate_scale_image(image, rotation, scale=scale)
        # else:
        #    object_fg = image

        # Perform random contrast changes

        # Flip vertically

        # Flip horizontal

        # Perform random color shift
        tran = torchvision.transforms.ToTensor()
        tensored_image = tran(image)

        augmented = aug_transformations(tensored_image)
        img_altered = augmented.T.numpy() * 255
        object_fg = np.swapaxes(img_altered, 0, 1)

        file_name = str(uuid.uuid4()).replace('-', '') + '.jpg'
        image_files_names.append(file_name)
        cv2.imwrite(directory + file_name, object_fg)
    return image_files_names


def round_all(numbers):
    return [round(number) for number in numbers]


def generate_objects_from_images(img_dir: str, coco_annotations, extrapolated_objects_dir, augmented_objects_dir):
    image_descriptions = coco_annotations['images']
    image_annotations = coco_annotations['annotations']
    categories = coco_annotations['categories']

    clazz_image_information = RandomDict({x['id']: {'name': x['name'], 'images': []} for x in categories})

    # Sort coco_annotations by image_id
    image_annotations = sorted(image_annotations, key=lambda i: i['image_id'])
    image_annotations = iter(image_annotations)

    image_annotation: Dict = next(image_annotations, {'image_id': -1})
    ann_img_id: int = image_annotation['image_id']

    for image_desc in image_descriptions:
        image_id: int = image_desc['id']

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

            cv2.imwrite(extrapolated_objects_dir + file_new_name + '.jpg', object_fg)

            generated_images_names = augment_image(object_fg, augmented_objects_dir)
            clazz_image_information[category_id]['images'].extend(generated_images_names)

            image_annotation = next(image_annotations, {'image_id': -1})
            ann_img_id = image_annotation['image_id']
    return clazz_image_information


def safe_polygon_placement(xywh, existing_polygons):
    x, y, w, h = xywh
    b = box(x, y, x + w, y + h)
    return not np.any([poly.intersects(b) for poly in existing_polygons])


def embed_object_into_image(object_image, background_image, xy_offset):
    x_offset, y_offset = xy_offset
    o_height, o_width = object_image.shape[:2]

    gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)[1]
    thresh_inv = 255 - thresh
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resulting_contours = max(contours, key=cv2.contourArea)[:, 0, :] + [x_offset, y_offset]

    roi = background_image[y_offset:y_offset + o_height, x_offset:x_offset + o_width].copy()

    roi_background = cv2.bitwise_and(roi, roi, mask=thresh_inv)
    dst = cv2.add(roi_background, object_image)

    background_image[y_offset: y_offset + o_height, x_offset:x_offset + o_width, :] = dst

    return Polygon(np.array(resulting_contours))


def generate_random_offset(background_shape, object_shape):
    b_height, b_width = background_shape
    o_height, o_width = object_shape

    random_x = random.randrange(0, b_width - o_width, 1)
    random_y = random.randrange(0, b_height - o_height, 1)
    return random_x, random_y


def generate_images_using(objects_dir: str, images_dir: str, classes, out, min_images_per_class: int = 100, prob_add_object=0.3):
    prob_additional_objects = prob_add_object
    background_images = os.listdir(images_dir)
    numb_backgrounds = len(background_images)
    images = []
    new_annotations = []

    # Generate 1000 images per class.
    for clazz_id, value in classes.items():
        clazz_images_names = value['images']
        numb_clazz_objects = len(clazz_images_names)

        for i in range(min_images_per_class):
            # Variable storing created polygons
            object_list = []
            # Pick random background
            background_image_path = background_images[random.randint(0, numb_backgrounds - 1)]
            # load background image
            background_image = load_image(images_dir + background_image_path)
            # Generate name for new image with objects in it
            file_name = str(uuid.uuid4()).replace('-', '')

            b_height, b_width = background_image.shape[:2]
            # Pick random augmented object
            foreground_object_path = clazz_images_names[random.randint(0, numb_clazz_objects - 1)]
            # load object image
            foreground_object = load_image(objects_dir + foreground_object_path)

            o_height, o_width = foreground_object.shape[:2]
            if o_height >= b_height or o_width >= b_width:
                h_scalar = (b_height / 2) / o_height
                w_scalar = (b_width / 2) / o_width

                scalar = h_scalar if h_scalar < w_scalar else w_scalar

                dimensions = (int(o_width * scalar), int(o_height * scalar))
                foreground_object = cv2.resize(foreground_object, dimensions)
                o_height, o_width = foreground_object.shape[:2]
                small_bg = True  # Background image to small to support additional objects
            else:
                small_bg = False

            x_offset, y_offset = generate_random_offset((b_height, b_width), (o_height, o_width))
            polygon = embed_object_into_image(foreground_object, background_image, (x_offset, y_offset))
            object_list.append(polygon)
            image_annotation_dict = {
                "id": str(uuid.uuid4()).replace('-', ''),
                "iscrowd": 0,
                "image_id": file_name,
                "category_id": clazz_id,
                "segmentation": [np.array(polygon.exterior.coords).ravel().tolist()],
                "bbox": [x_offset, y_offset, o_width, o_height],
                "area": polygon.area
            }
            new_annotations.append(image_annotation_dict)

            # Add additional objects into the image
            while not small_bg and random.random() < prob_additional_objects:
                # Pick random class
                random_class_id, random_class_value = classes.random_item()
                random_clazz_images_names = random_class_value['images']
                numb_random_clazz_objects = len(random_clazz_images_names)

                # Pick random object from class
                random_index = random.randint(0, numb_random_clazz_objects - 1)
                random_foreground_object_path = random_clazz_images_names[random_index]

                random_foreground_object = load_image(objects_dir + random_foreground_object_path)
                o_height, o_width = random_foreground_object.shape[:2]

                # Object already placed, no need to overcrowd the image.
                if o_height >= b_height or o_width >= b_width:
                    break

                max_tries = 10
                tries = 0
                while tries < max_tries:
                    x_offset, y_offset = generate_random_offset((b_height, b_width), (o_height, o_width))
                    if safe_polygon_placement((x_offset, y_offset, o_width, o_height), object_list):
                        break
                    tries += 1

                if tries >= max_tries:
                    continue

                random_polygon = embed_object_into_image(random_foreground_object, background_image,
                                                         (x_offset, y_offset))
                object_list.append(random_polygon)
                random_image_annotation_dict = {
                    "id": str(uuid.uuid4()).replace('-', ''),
                    "iscrowd": 0,
                    "image_id": file_name,
                    "category_id": random_class_id,
                    "segmentation": [np.array(random_polygon.exterior.coords).ravel().tolist()],
                    "bbox": [x_offset, y_offset, o_width, o_height],
                    "area": random_polygon.area
                }
                new_annotations.append(random_image_annotation_dict)

            cv2.imwrite(out + file_name + '.jpg', background_image)

            image_dict = {'id': file_name, 'width': b_width, 'height': b_height, 'file_name': file_name + '.jpg'}
            images.append(image_dict)

    return images, new_annotations


if __name__ == '__main__':
    config_path = '../configs/preprocess_images_config.yaml'
    config = load_config_file(config_path)

    bck_base_path = config['bck_base_path']
    split_backgrounds = {'train': bck_base_path + 'train/',
                         'test': bck_base_path + 'test/',
                         'val': bck_base_path + 'val/'}

    generated_data_path_dict = generate_path_dict(config['generated_images_path'],
                                                  config['generated_labels_path'],
                                                  config['partial_f_name'])
    path_dict = generate_path_dict(config['images_path'],
                                   config['labels_path'],
                                   config['partial_f_name'])

    clear_directory_contents([config['augmented_objects_dir'],
                              bck_base_path,
                              '../generated_data/',
                              config['extrapolated_objects_dir'],
                              config['images_path'],
                              config['labels_path']])
    directories = list(generated_data_path_dict['images'].values()) + [config['generated_labels_path'],
                                                                       config['augmented_objects_dir'],
                                                                       config['extrapolated_objects_dir']]
    setup_dirs(directories)

    coco_annotations = load_annotations(config['annotation_dir'])
    split_images_annotations(coco_annotations, config['img_dir'], path_dict)
    split_background_images(config['background_images_dir'], split_backgrounds)

    for key, image_dir in path_dict['images'].items():
        label_path = path_dict['labels'][key]
        backgrounds = split_backgrounds[key]
        img_dir_coco_annotations = load_annotations(label_path)

        annotation_out_path = generated_data_path_dict['labels'][key]
        images_out_path = generated_data_path_dict['images'][key]

        generated_images_information = generate_objects_from_images(image_dir, img_dir_coco_annotations,
                                                                    config['extrapolated_objects_dir'],
                                                                    config['augmented_objects_dir'])
        gen_image_information, gen_annotations = generate_images_using(config['augmented_objects_dir'], backgrounds,
                                                                       generated_images_information, images_out_path)
        img_dir_coco_annotations['images'] = gen_image_information
        img_dir_coco_annotations['annotations'] = gen_annotations

        save_annotations(img_dir_coco_annotations, annotation_out_path)
