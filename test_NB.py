__author__ = 'lixin77'

from sklearn.naive_bayes import MultinomialNB
from preprocess import *
import random


dataset_name = 'MR'

data = {}

n_samples = 0

with open('dataset/%s/%s.txt' % (dataset_name, dataset_name)) as fp:
    for line in fp:
        label, text = line.strip().split('\t')
        try:
            data[label].append(line.strip())
        except KeyError:
            data[label] = [line.strip()]
        n_samples += 1

fold = 10

for i in xrange(fold):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    train_count = int(n_samples * 0.9)
    ctr = 0
    for y in data:
        random.shuffle(data[y])
        for sample in data[y]:
            y, x = sample.split('\t')
            if ctr < train_count:
                X_train.append(x)
                Y_train.append(y)
            else:
                X_test.append(x)
                Y_test.append(y)
            ctr += 1












