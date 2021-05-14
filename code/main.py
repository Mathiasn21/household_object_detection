import json
from typing import Dict

from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes


def create_info_dict():
    new_info_dict = {}
    for k in information_keys:
        new_info_dict[k] = []

    return new_info_dict


if __name__ == '__main__':
    metric_path = './code/output/metrics.json'
    detectron2_train_runs = {}
    run_iteration = 0

    information_keys = ['fast_rcnn/cls_accuracy',
                        'fast_rcnn/false_negative', 'fast_rcnn/fg_cls_accuracy',
                        'loss_box_reg',
                        'loss_cls', 'loss_mask',
                        'loss_rpn_cls', 'loss_rpn_loc',
                        'lr', 'mask_rcnn/accuracy',
                        'mask_rcnn/false_negative', 'mask_rcnn/false_positive',
                        'roi_head/num_bg_samples', 'roi_head/num_fg_samples',
                        'rpn/num_neg_anchors', 'rpn/num_pos_anchors', 'total_loss', 'iteration']

    keys_to_del = {'time'}

    iterations = {0: {'iteration': []}}
    detectron2_train_runs[run_iteration] = create_info_dict()
    current_iteration = run_iteration
    last_iteration = run_iteration

    with open(metric_path) as json_file:
        for line in json_file:
            data: Dict = json.loads(line)
            current_iteration = data['iteration']

            for key in information_keys:
                if key not in data:
                    continue
                detectron2_train_runs[run_iteration][key].append(data[key])

            if data['eta_seconds'] == 0.0 or current_iteration < last_iteration:
                run_iteration += 1
                detectron2_train_runs[run_iteration] = create_info_dict()
                last_iteration = 0
                continue

            last_iteration = current_iteration

    del detectron2_train_runs[run_iteration]
    rows = len(detectron2_train_runs)
    columns = len(information_keys) - 1
    fig, axs = plt.subplots(nrows=rows, ncols=columns, squeeze=False, figsize=(70, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for row in range(rows):
        train_run = detectron2_train_runs[row]
        for column, key in enumerate(information_keys):
            if column == columns:
                break
            if key == 'iteration' or key not in train_run:
                continue
            x_iterations = train_run['iteration']
            y_info = train_run[key]

            ax: Axes = axs[row, column]
            ax.plot(x_iterations, y_info)
            ax.set_xlabel('Iterations')
            ax.set_ylabel(key)
    plt.savefig("baseline.svg")
    plt.show()
