from typing import List

import matplotlib.pyplot as plt
import numpy as np


class COCODatasetMetrics:
    unique_categories: List = []
    colors = []

    def __init__(self, annotations):
        self.annotations = annotations
        self.key_category_count = self.extract_instances_per_class()

    def extract_instances_per_class(self):
        """
        Extract training runs and the corresponding metrics.
        :return: Dict - Contains the category key and corresponding dataset count.
        """
        key_category = {}
        count_key = 'count'

        # Count all annotation instances and store that in key_category dict.
        for key, raw_annotation in self.annotations.items():
            categories_dict_tmp = raw_annotation['categories']
            categories_dict = {x['id']: {'name': x['name'], count_key: 0} for x in categories_dict_tmp}

            if len(self.unique_categories) == 0:
                self.unique_categories = [x['name'] for x in categories_dict.values()]

            # count all instances belonging to each category
            for annotation in raw_annotation['annotations']:
                category_id = annotation['category_id']
                categories_dict[category_id][count_key] += 1

            categories_dict = {x['name']: x[count_key] for x in categories_dict.values()}

            # Add total count to category dict
            categories_dict['total'] = sum([i for i in categories_dict.values()])

            key_category[key] = categories_dict
        return key_category

    def plot_metrics(self, name):
        """
        Plot the class distribution in a coco annotated dataset.
        :param name: str - file name for plot.
        """
        bar_width = 0.25

        num_bars_pr_class = len(self.key_category_count.keys())
        unique_categories = self.unique_categories
        num_categories = len(unique_categories)
        bar_positions = np.arange(num_categories)

        for key, category_dict in self.key_category_count.items():
            counts_list = [category_dict[category] for category in unique_categories]
            bar_positions = [x + bar_width for x in bar_positions]

            plt.bar(bar_positions, counts_list, width=bar_width,
                    label='{} Total: {}'.format(key, category_dict['total']))

        plt.xlabel('Categories', fontweight='bold')
        plt.ylabel('Number', fontweight='bold')

        plt.xticks([(r + (bar_width * (num_bars_pr_class + 1) / 2)) for r in range(num_categories)], unique_categories)
        plt.legend()

        plt.savefig('../statistical_results/inferences/{}.svg'.format(name))
        plt.show()
