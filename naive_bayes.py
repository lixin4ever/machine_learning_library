__author__ = 'lixin77'

from preprocess import *
import cmath
import numpy as np

# naive bayes models
class NB(object):
    pass

class multinomial_NB(object):
    def __init__(self, use_prior=True):
        self.use_prior = use_prior

    def set_prior(self, alpha=0):
        self.alpha = alpha

    def train(self, texts, labels):
        """
        train the multinomial NB model although NB does not have a explicit training process
        :param texts: text in the training set
        :param labels: labels of the training documents
        """
        count_vectorizor, sparse_feature_mat = get_word_count(sentences=texts)
        self.vocab = count_vectorizor.vocabulary_
        self.indexed_vocab = {}
        for (k, v) in self.vocab.iteritems():
            self.indexed_vocab[v] = k
        # analyser can perform pre-processing and tokenization
        self.analyser = count_vectorizor.build_analyser()
        self.class_count, self.class_count_summary = np.zeros(len(labels)), {}
        # number of training samples
        self.n_train = len(texts)
        for y in labels:
            self.class_count[y] = 0
            self.class_count_summary[y] = {}
        for (doc, y) in zip(sparse_feature_mat, labels):
            self.class_count[y] += 1
            for w in doc:
                count = doc[w]
                try:
                    self.class_count_summary[y][w] += count
                except KeyError:
                    self.class_count_summary[y][w] = count

    def predict(self, texts):
        """
        predict labels of the testing documents and return the probabilities over labels for each document
        :param texts: testing sentences or documents
        """
        clear_texts = [preprocess(sentence=text) for text in texts]
        labels = self.class_count_summary.keys()
        # score is a 2d matrix recording probability of the event that document is assigned to one label
        score = np.zeros(len(clear_texts), len(labels))
        # avoid some class(es) is empty
        p_y = float(self.class_count + self.alpha) / (self.n_train + len(self.class_count) * self.alpha)
        p_x_y = {}
        for y in labels:
            p_x_y[y] = {}
            # number of words occurs in the class
            word_count = 0
            for w in self.class_count_summary[y]:
                word_count += self.class_count_summary[y][w]
            for w in self.class_count_summary[y]:
                p_x_y[y][w] = (self.class_count_summary[y][w] + self.alpha) / (word_count + len(self.vocab) * self.alpha)
        for i in xrange(len(clear_texts)):
            text = clear_texts[i]
            words = self.analyser(text)
            for y in labels:
                score[i][y] += cmath.log(p_y[y])
                for j in xrange(words):
                    w = words[j]
                    try:
                        score[i][y] += cmath.log(p_x_y[y][w])
                    except KeyError:
                        # new word that does not occur in the training set
                        p_new = self.alpha / (word_count + len(self.vocab) * self.alpha)
                        score[i][y] += cmath.log(p_new)
        return score, np.argmax(score, axis=1)

class bernoulli_NB(object):
    def __init__(self, use_prior=True):
        self.use_prior = use_prior

    def set_prior(self, alpha=0):
        self.alpha = alpha

class gaussian_NB(object):
    pass

