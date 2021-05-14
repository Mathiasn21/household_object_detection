import os

from cv2 import cv2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, coco_evaluation
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer


# Custom simple CocoTrainer class. Used for training a model
from tools.tools import load_annotations, clear_directory_contents


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


if __name__ == '__main__':
    # Variables for COCO annotations path
    train_ann_path = '../generated_data/labels/train_coco_annotations.json'
    # train_ann_path = '../data/labels/train_coco_annotations.json'
    test_ann_path = '../data/labels/test_coco_annotations.json'
    generated_val_ann_path = '../generated_data/labels/val_coco_annotations.json'
    val_ann_path = '../data/labels/val_coco_annotations.json'

    # Variables for tattoo images root path
    # train_img_path = '../data/images/train'
    train_img_path = '../generated_data/images/train'
    test_img_path = '../data/images/test'
    val_img_path = '../data/images/val'
    generated_val_img_path = '../generated_data/images/val'
    categories = load_annotations(train_ann_path)['categories']

    class_dict = {}
    # Register the datasets and corresponding annotations in COCO format
    register_coco_instances('objects_train', class_dict, train_ann_path, train_img_path)
    register_coco_instances('objects_val', class_dict, val_ann_path, val_img_path)
    register_coco_instances('generated_objects_val', class_dict, generated_val_ann_path, generated_val_img_path)
    register_coco_instances('objects_test', {}, test_ann_path, test_img_path)

    # Setup model configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
    cfg.DATASETS.TRAIN = ('objects_train', 'generated_objects_val')
    cfg.DATASETS.TEST = ('objects_val',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 8000
    cfg.SOLVER.STEPS = (1000, 1500, 2000, 3000, 4000, 6000, 7000)
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(categories)

    # Create directories if not existing
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # setup_logger() # Logs additional information during training

    ##################################
    # Set weights used for the model. Comment this if one is required to train the model first
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    ##################################

    # Set confidence score threshold
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75

    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)

    ##################################
    # trainer.train()  # Uncomment this to train the network first
    ##################################

    # Evaluate and test the trained model
    evaluator = COCOEvaluator('objects_test', distributed=False, output_dir='output/evaluation')
    det_test_loader = build_detection_test_loader(cfg, 'objects_test')

    # Run inference on the test data set
    inference_on_dataset(trainer.model, det_test_loader, evaluator)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.DATASETS.TEST = ('objects_test',)

    # Setup a simple predictor
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get(test_img_path)

    # Draw bounding boxes on images and store those images in a folder
    for imageName in os.listdir(test_img_path):
        im = cv2.imread(test_img_path + '//' + imageName)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=test_metadata,
                       scale=0.8
                       )
        out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(
            '../results/{}'.format(
                imageName), out.get_image()[:, :, ::-1])
