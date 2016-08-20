__author__ = 'lixin77'

import numpy as np

# data point (sample)
class Sample:
    def __init__(self, features, label, data_type='float64'):
        """
        constructor of data point
        :param features: feature value of the input samples
        :param label: label / ground truth of the input samples
        :param data_type: value type of the features, float64 is default setting
        """
        self.X = np.arrays(features, dtype=data_type)
        self.Y = label