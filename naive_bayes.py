__author__ = 'lixin77'

from preprocess import *
import math
import numpy as np

# naive bayes models
class NB(object):
    pass

class multinomial_NB(object):
    def __init__(self, use_prior=True, alpha=0.01):
        self.use_prior = use_prior
        self.alpha = alpha

    def train(self, X, Y, vocab):
        """
        train the multinomial NB model although NB does not have a explicit training process
        :param X: sparse representation of the sentence-term matrix
        :param Y: ground truth label of each sentence
        :param cv: count vectorizor based on the statics of training set
        """
        #count_vectorizor, _, sparse_feature_mat = build_dataset(sentences=texts)
        self.vocab = vocab
        self.indexed_vocab = {}
        for (k, v) in self.vocab.iteritems():
            self.indexed_vocab[v] = k
        # self.class_count ==>> number of training samples in each class
        # self.class_count_summary ==>> word count in each class
        self.class_count, self.class_count_summary = np.zeros(len(Y)), {}
        # number of training samples
        self.n_train = len(X)
        for y in Y:
            self.class_count[y] = 0
            self.class_count_summary[y] = {}
        for (sen, y) in zip(X, Y):
            self.class_count[y] += 1
            for wid in sen:
                count = sen[wid]
                try:
                    self.class_count_summary[y][wid] += count
                except KeyError:
                    self.class_count_summary[y][wid] = count

    def predict(self, X):
        """
        predict labels of the testing documents and return the probabilities over labels for each document
        :param X: sparse form of testing sentences matrix
        """
        labels = self.class_count_summary.keys()
        # score is a 2d matrix recording probability of the event that document is assigned to one label
        score = np.zeros((len(X), len(labels)))
        # avoid some class(es) is empty
        p_y = [float(self.class_count[i] + self.alpha) / (self.n_train + len(self.class_count) * self.alpha)
               for i in xrange(len(self.class_count))]
        p_x_y = {}
        for y in labels:
            p_x_y[y] = {}
            # number of words occurs in the class
            word_count = 0
            for w in self.class_count_summary[y]:
                word_count += self.class_count_summary[y][w]
            for w in self.class_count_summary[y]:
                p_x_y[y][w] = (self.class_count_summary[y][w] + self.alpha) / (word_count + len(self.vocab) * self.alpha)
        for i in xrange(len(X)):
            sen = X[i]
            # pre-process and tokenize the input testing text
            for y in labels:
                score[i][y] += math.log(p_y[y])
                for j in sen:
                    wid = j
                    # times ==>> times that word occurs in current sentence
                    times = sen[j]
                    try:
                        score[i][y] += (times * math.log(p_x_y[y][wid]))
                    except KeyError:
                        # new word that does not occur in the training set
                        p_new = self.alpha / (word_count + len(self.vocab) * self.alpha)
                        score[i][y] += (times * math.log(p_new))
        return score, np.argmax(score, axis=1)

class bernoulli_NB(object):
    def __init__(self, use_prior=True):
        self.use_prior = use_prior

    def set_prior(self, alpha=0):
        self.alpha = alpha

class gaussian_NB(object):
    pass

