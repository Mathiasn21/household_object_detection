import json
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes

from tools.tools import load_json_data, save_json_data


# Class used for extracting and storing detectron2 training metrics
class TrainingMetrics:
    # Information keys to extract from the training metrics
    information_keys = [
        # 'fast_rcnn/cls_accuracy',
        # 'fast_rcnn/false_negative',
        'fast_rcnn/fg_cls_accuracy',
        # 'loss_box_reg',
        # 'loss_cls',
        # 'loss_mask',
        # 'loss_rpn_cls', 'loss_rpn_loc',
        # 'lr', 'mask_rcnn/accuracy',
        # 'mask_rcnn/false_negative', 'mask_rcnn/false_positive',
        'roi_head/num_bg_samples', 'roi_head/num_fg_samples',
        # 'rpn/num_neg_anchors', 'rpn/num_pos_anchors',
        # 'total_loss',
        'iteration']

    # path for the detectron2 metrics
    metric_path = 'output/metrics.json'

    # Output path to store the metrics
    training_iterations_metrics_path = '../statistical_results/training/training_run_metrics.json'

    def create_info_dict(self):
        """
        Create a new information dict.
        :rtype: Dict
        """
        new_info_dict = {}
        for k in self.information_keys:
            new_info_dict[k] = []

        return new_info_dict

    def save_latest_iteration(self, iteration_name: str):
        """
        Store the latest iterations metrics
        :type iteration_name: str - Name of the iteration run
        """
        latest_iteration = self.extract_latest_training_metric()

        # Append this iteration metrics to the existing ones
        training_metrics = load_json_data(self.training_iterations_metrics_path)
        training_metrics[iteration_name] = latest_iteration
        save_json_data(training_metrics, self.training_iterations_metrics_path)

    def extract_latest_training_metric(self):
        """
        Extract the latest metric pertaining to the latest training iteration
        :rtype: Dict - Containing the training runs
        """
        detectron2_train_runs = {}
        run_iteration = 0
        information_keys = self.information_keys

        detectron2_train_runs[run_iteration] = self.create_info_dict()
        last_iteration = run_iteration

        with open(self.metric_path) as json_file:
            for line in json_file:
                data: Dict = json.loads(line)
                current_iteration = data['iteration']

                for key in information_keys:
                    if key not in data:
                        continue
                    detectron2_train_runs[run_iteration][key].append(data[key])

                # Check if we overstep into another training iteration
                if data['eta_seconds'] == 0.0 or current_iteration < last_iteration:
                    run_iteration += 1
                    detectron2_train_runs[run_iteration] = self.create_info_dict()
                    last_iteration = 0
                    continue

                last_iteration = current_iteration

        # Remove extra redundant empty entry
        del detectron2_train_runs[run_iteration]
        return detectron2_train_runs[0]

    def plot_training_metrics(self):
        """
        Plot the training metrics and store that information on disk
        """
        training_metrics = load_json_data(self.training_iterations_metrics_path)
        fig_name = 'training_metrics'
        information_keys = self.information_keys

        # Number of axes excluding the iteration x iteration entry.
        columns = len(information_keys) - 1

        # Create subplots to plot on
        fig, axs = plt.subplots(ncols=columns)

        # Plot going vertically over the column entries.
        for index, key in enumerate(information_keys):
            ax: Axes = axs[index]
            ax.set_xlabel(key.upper(), fontweight='bold')

            for name, metric in training_metrics.items():
                x_iterations = metric['iteration']
                y_info = metric[key]

                ax.plot(x_iterations, y_info, label=name)
                ax.set_xlabel('Iterations')
                ax.set_ylabel(key)

            ax.legend(loc='best')
            if index == columns - 1:
                break

        fig.set_size_inches([20, 3])
        fig.tight_layout(pad=1.0)

        # Save and display resulting figure
        plt.savefig('../statistical_results/training/{}.svg'.format(fig_name))
        plt.show()


# Class used for extracting and storing detectron2 inference metrics
class InferenceMetrics:
    # Path to store the inference metrics
    inference_metrics_path = '../statistical_results/inferences/all_inferences.json'

    # Key and categories to plot
    information_keys = ['bbox', 'segm']
    evaluation_keys = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AP-Chair', 'AP-Painting', 'AP-Plant', 'AP-Couch',
                       'AP-Cup']

    # Reverse to allow for a more pretty plot.
    evaluation_keys.reverse()

    # Save the inference metrics on disk
    def save_inference_metrics(self, metrics, iteration_name):
        training_metrics = load_json_data(self.inference_metrics_path)
        training_metrics[iteration_name] = metrics
        save_json_data(training_metrics, self.inference_metrics_path)

    # Plot the inference metrics from disk
    def plot_inference_metrics(self):
        inference_metrics = load_json_data(self.inference_metrics_path)
        information_keys = self.information_keys

        # Set number of total column, axes, and row, stored inferences.
        columns = len(information_keys)
        num_inferences = len(inference_metrics)
        num_eval_keys = len(self.evaluation_keys)

        # Create subplots for plotting
        fig, axs = plt.subplots(ncols=columns)

        # Set bar width
        width = 0.25

        # Set shared y label across all axes
        axs[0].set_ylabel('Metric', fontweight='bold')

        # plot vertically the columns
        for index, key in enumerate(information_keys):
            ax: Axes = axs[index]

            y_positions = np.arange(num_eval_keys)
            ax.set_xlabel(key.upper(), fontweight='bold')

            for name, inference in inference_metrics.items():

                # Calculate the y positions used for plotting the bars
                y_positions = [x + width for x in y_positions]

                inference_dict = inference[key]
                data_values = [inference_dict[x] for x in self.evaluation_keys]

                ax.barh(y_positions, data_values, label=name, height=width)

            ax.set(yticks=[(r + (width / 2) * (num_inferences + 1)) for r in range(num_eval_keys)],
                   yticklabels=self.evaluation_keys)

        axs[1].legend(loc='lower right')
        fig.tight_layout(pad=1.0)

        # Save and display the inference metrics
        plt.savefig('../statistical_results/inferences/inference_metrics.svg')
        plt.show()
