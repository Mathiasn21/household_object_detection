from typing import Dict

from tools.coco_dataset_metrics import COCODatasetMetrics
from tools.tools import load_json_data
from detectron2_metrics import TrainingMetrics, InferenceMetrics


def load_annotations(annotation_paths: Dict):
    """
    Load annotations from path and assign them to corresponding key.
    :param annotation_paths: Dict
    :return: Dict - {key: annotations}
    """

    annotations = {}
    for key, path in annotation_paths.items():
        annotations[key] = load_json_data(path)
    return annotations


if __name__ == '__main__':
    # Setup path variables
    annotation_org_dataset_path = {
        'Original dataset': '../data/testing_annotations/labels_household_object_detection_newest.json'
    }

    annotation_split_dataset_path = {
        'Split data - train': '../data/labels/train_coco_annotations.json',
        'Split data - validation': '../data/labels/val_coco_annotations.json',
        'Split data - test': '../data/labels/test_coco_annotations.json'
    }

    annotation_generated_split_dataset_path = {
        'Generated data - train': '../generated_data/labels/train_coco_annotations.json',
        'Generated data - validation': '../generated_data/labels/val_coco_annotations.json',
        'Generated data - test': '../generated_data/labels/test_coco_annotations.json'
    }

    all_annotations = {
        'original_dataset': annotation_org_dataset_path,
        'original_split_dataset': annotation_split_dataset_path,
        'generated_split_dataset': annotation_generated_split_dataset_path
    }

    # Iterate over the annotations and plot corresponding dataset distributions
    for key, annotation_paths in all_annotations.items():
        annotations = load_annotations(annotation_paths)

        dataset_metrics = COCODatasetMetrics(annotations)
        dataset_metrics.plot_metrics(key)

    # Plot training and inference metrics for all training runs
    training_metrics = TrainingMetrics()
    inference_metrics = InferenceMetrics()

    training_metrics.plot_training_metrics()
    inference_metrics.plot_inference_metrics()
