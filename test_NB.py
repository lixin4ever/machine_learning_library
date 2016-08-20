__author__ = 'lixin77'

from sklearn.naive_bayes import MultinomialNB
from preprocess import *
import random
from naive_bayes import multinomial_NB
import numpy as np
import time


def compute_accu(Y_gold, Y_pred):
    assert len(Y_gold) == len(Y_pred)
    hit_count = 0
    for i in xrange(len(Y_gold)):
        if Y_gold[i] == Y_pred[i]:
            hit_count += 1
    return float(hit_count) / len(Y_gold)

dataset_name = 'MR'

data = {}

n_samples = 0

with open('dataset/%s/%s.txt' % (dataset_name, dataset_name)) as fp:
    for line in fp:
        label, text = line.strip().split('\t')
        try:
            data[int(label)].append(line.strip())
        except KeyError:
            data[int(label)] = [line.strip()]
        n_samples += 1

N = 10
accu_mnb_sklearn = 0.0
accu_mnb_manual = 0.0

for y in data:
    random.shuffle(data[y])


for i in xrange(N):
    print "in the round %s..." % i
    train_sen = []
    test_sen = []
    Y_train = []
    Y_test = []
    # stratified sampling, to ensure distribution of each class will not change after the split
    for y in data:
        # split dataset as 10 subset
        single_fold = int(len(data[y]) * 0.1)
        for j in xrange(len(data[y])):
            label, x = data[y][j].split('\t')
            if i * single_fold <= j < (i + 1) * single_fold:
                Y_test.append(int(label))
                test_sen.append(x)
            else:
                Y_train.append(int(label))
                train_sen.append(x)
    print 'number of training samples', len(train_sen)
    print 'number of testing samples', len(test_sen)
    Y_test = np.array(Y_test)
    print "build the dataset..."
    start_time = time.time()
    # build the training set
    X_train, X_train_sparse, X_test, X_test_sparse, vocab = build_dataset(train_sen=train_sen, test_sen=test_sen)
    print 'time cost of processing the data: %s seconds' % (time.time() - start_time)
    print "run the model in sklearn....."
    start_time = time.time()
    mnb_sklearn = MultinomialNB(alpha=0.01)
    mnb_sklearn.fit(X_train, Y_train)
    Y_pred_mnb_sklearn = mnb_sklearn.predict(X_test)
    print 'time cost of sklearn: %s seconds' % (time.time() - start_time)
    assert Y_pred_mnb_sklearn.shape[0] == len(Y_test)
    perf_mnb_sklearn = compute_accu(Y_gold=Y_test, Y_pred=Y_pred_mnb_sklearn)
    accu_mnb_sklearn += perf_mnb_sklearn

    print "run the manually implemented model..."
    start_time = time.time()
    mnb_manual = multinomial_NB(use_prior=True, alpha=0.01)
    mnb_manual.train(X=X_train_sparse, Y=Y_train, vocab=vocab)
    p_Y_X, Y_pred_mnb_manual = mnb_manual.predict(X=X_test_sparse)
    print 'time cost of manual implementation: %s seconds' % (time.time() - start_time)
    assert Y_pred_mnb_manual.shape[0] == len(Y_test)
    perf_mnb_manual = compute_accu(Y_gold=Y_test, Y_pred=Y_pred_mnb_manual)
    accu_mnb_manual += perf_mnb_manual
    print "In round %s, sklearn: %s, manual: %s" % (i, perf_mnb_sklearn, perf_mnb_manual)

accu_mnb_manual /= N
accu_mnb_sklearn /= N

print "performance of multinomial NB model in sklearn is %s%%" % (100 * accu_mnb_sklearn)
print "performance of multinomial NB model implemented by myself is %s%%" % (100 * accu_mnb_manual)











