import os

from detectron2.config import get_cfg, CfgNode
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.model_zoo import model_zoo

# Custom simple CocoTrainer class. Used for training a model
from detectron2_metrics import TrainingMetrics
from tools.tools import load_json_data, clear_directory_contents


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


if __name__ == '__main__':
    # Variables for COCO annotations path
    generated_train_ann_path = '../generated_data/labels/train_coco_annotations.json'
    train_ann_path = '../data/labels/train_coco_annotations.json'
    test_ann_path = '../data/labels/test_coco_annotations.json'
    generated_val_ann_path = '../generated_data/labels/val_coco_annotations.json'
    val_ann_path = '../data/labels/val_coco_annotations.json'

    # Variables for images root path
    generated_train_img_path = '../generated_data/images/train'
    train_img_path = '../data/images/train'
    test_img_path = '../data/images/test'
    val_img_path = '../data/images/val'
    generated_val_img_path = '../generated_data/images/val'
    model_out_name = 'synth_2000_scaling'

    # Clear previous output content
    clear_directory_contents(['./output'])

    # Retrieve categories
    categories = load_json_data(generated_train_ann_path)['categories']

    class_dict = {}
    # Register the datasets and corresponding annotations in COCO format
    register_coco_instances('generated_objects_train', class_dict, generated_train_ann_path, generated_train_img_path)
    register_coco_instances('objects_train', class_dict, train_ann_path, train_img_path)

    register_coco_instances('objects_val', class_dict, val_ann_path, val_img_path)
    register_coco_instances('generated_objects_val', class_dict, generated_val_ann_path, generated_val_img_path)

    register_coco_instances('objects_test', {}, test_ann_path, test_img_path)

    # Setup model configuration
    cfg: CfgNode = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
    cfg.DATASETS.TRAIN = ('generated_objects_train',)
    cfg.DATASETS.TEST = ('objects_val',)  # Notice that the utilized validation set is real images not synthetic ones

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 2000
    cfg.SOLVER.STEPS = (500, 1500)
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(categories)

    # Create directories if not existing
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # setup_logger() # Logs additional information during training

    # Set confidence score threshold
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75

    # Create a default trainer and train the model
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    try:
        trainer.train()
    except Exception as e:
        print(e)
    finally:
        # Save current training metrics to disk
        metrics = TrainingMetrics()
        metrics.save_latest_iteration(model_out_name)
