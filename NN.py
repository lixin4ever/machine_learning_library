__author__ = 'lixin77'

# testing file of nearest neighbour

from model.nearest_neighbor import KNN
from sklearn.neighbors import KNeighborsClassifier
from test.utils import cv

dataset_name = 'MR'

data_path = './dataset/%s/%s.txt' % (dataset_name, dataset_name)

#
models = [
    KNN(K=10), KNeighborsClassifier(n_neighbors=10)
]

model_names = [
    'KNN',
    'KNN (sklearn)'
]

# perform k-fold cross validation
cv(data_path=data_path, models=models, model_names=model_names, k=10)

