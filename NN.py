__author__ = 'lixin77'

# testing file of nearest neighbour

from model.nearest_neighbor import KNN, nearest_centroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from test.utils import cv

dataset_name = 'MR'

data_path = './dataset/%s/%s.txt' % (dataset_name, dataset_name)

#
models = [
    nearest_centroid(),
    NearestCentroid()
]

model_names = [
    'Nearest_Centroid',
    'Nearest_Centroid (sklearn)'
]

# perform k-fold cross validation
cv(data_path=data_path, models=models, model_names=model_names, k=10)

