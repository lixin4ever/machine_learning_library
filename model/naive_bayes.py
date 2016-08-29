__author__ = 'lixin77'

import math
import numpy as np

# naive bayes models

class NB(object):
    pass

class multinomial_NB(object):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def train(self, X, Y, vocab):
        """
        train the multinomial NB model although NB does not have a explicit training process
        :param X: sparse representation of the sentence-term matrix
        :param Y: ground truth label of each sentence
        :param cv: count vectorizor based on the statics of training set
        """
        self.vocab = vocab
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
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def train(self, X, Y, vocab):
        """
        train the bernoulli NB model although NB does not have a explicit training process
        :param X: sparse representation of the sentence-term matrix
        :param Y: ground truth label of each sentence
        :param cv: count vectorizor based on the statics of training set
        """
        self.vocab = vocab
        self.n_class = len(set(Y))
        # self.class_count ==>> number of training samples in each class
        # self.class_count_summary ==>> word count in each class
        self.class_count, self.class_count_summary = np.zeros(self.n_class), np.zeros((self.n_class, len(vocab)))
        # number of training samples
        self.n_train = len(X)
        for (sen, y) in zip(X, Y):
            self.class_count[y] += 1
            for wid in sen:
                self.class_count_summary[y][wid] += 1

    def predict(self, X):
        """
        predict labels of the testing documents and return the probabilities over labels for each document
        :param X: mapping between testing sentences and terms
        """
        labels = range(self.n_class)
        # score is a 2d matrix recording probability of the event that document is assigned to one label
        score = np.zeros((len(X), len(labels)))
        # avoid some class(es) is empty
        p_y = [float(self.class_count[i] + self.alpha) / (self.n_train + len(self.class_count) * self.alpha)
               for i in xrange(len(self.class_count))]
        p_x_y = np.zeros((self.n_class, len(self.vocab)))
        class_word_count = np.sum(self.class_count_summary, axis=1)
        for y in labels:
            p_x_y[y, :] = (self.class_count_summary[y] + self.alpha) / (class_word_count[y] + 2 * self.alpha)
        for i in xrange(len(X)):
            x = X[i]
            # pre-process and tokenize the input testing text
            for y in labels:
                score[i][y] += math.log(p_y[y])
                for j in xrange(len(x)):
                    wid = j
                    # x[wid] ==>> word count
                    if x[wid] > 0:
                        score[i][y] += math.log(p_x_y[y][wid])
                    else:
                        score[i][y] += math.log((1 - p_x_y[y][wid]))
        return score, np.argmax(score, axis=1)

class gaussian_NB(object):

    def __init__(self, alpha=0.01):
        # note: the smoother alpha in gaussian NB model is just used in computing class probability
        self.alpha = alpha

    def train(self, X, Y, vocab):
        """
        train a gaussian naive bayes model for text classification
        :param X: document-term matrix of the training set
        :param Y: ground truth label of the training set
        :param vocab: vocabulary built from the training documents
        """
        self.vocab = vocab
        self.n_class = len(set(Y))
        self.n_train = len(X)
        self.class_count = np.zeros(self.n_class)
        for y in Y:
            self.class_count[y] += 1
        self.means = np.zeros((self.n_class, len(self.vocab)))
        self.vars = np.zeros((self.n_class, len(self.vocab)))

        for y in xrange(self.n_class):
            class_documents = [X[i] for i in xrange(len(Y)) if Y[i] == y]
            class_documents = np.array(class_documents)
            # calculate mean and variance of each dimension (feature) given the feature vectors
            self.means[y, :] = np.mean(class_documents, axis=0)
            self.vars[y, :] = np.var(class_documents, axis=0)
        epsilon = 1e-9 * np.var(X, axis=0).max()
        self.vars += epsilon


    def predict(self, X):
        """
        predict labels for the testing set
        :param X: the document-term matrix derived from testing set
        """
        labels = range(self.n_class)
        # 2-d matrix records the log likelihood value
        score = np.zeros((len(X), self.n_class))
        # class probability
        p_y = np.zeros(self.n_class)
        for y in labels:
            p_y[y] = (self.class_count[y] + self.alpha) / (self.n_train + self.n_class * self.alpha)
        pi = math.pi
        for i in xrange(len(X)):
            sen = X[i]
            assert sen.shape[0] == self.means.shape[1]
            for y in labels:
                # calculate log likelihood
                score[i][y] += math.log(p_y[y])
                score[i][y] += (-0.5) * np.sum(np.log(2 * pi * self.vars[y]))
                score[i][y] += (-0.5) * np.sum((sen - self.means[y]) ** 2 / self.vars[y])
        return score, np.argmax(score, axis=1)

