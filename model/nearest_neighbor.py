__author__ = 'lixin77'

import numpy as np

def manhattan_distance(x1, x2):
    """
    Manhattan distance between vector x1 and x2
    """
    return np.linalg.norm(x1 - x2, ord=1)

def euclidean_distance(x1, x2):
    """
    Euclidean distance between vector x1 and x2
    """
    return np.linalg.norm(x1 - x2)


class KNN:

    def __init__(self, K=5):
        """
        default K value is 5
        """
        self.K = K

    def train(self, X, Y, vocab):
        """
        train a KNN classifier
        :param X: doc-term matrix of training data
        :param Y: list of ground truth labels
        :param vocab: vocabulary built on training data
        """
        # note: KNN actually does not have a explicit training process
        self.vocab = vocab
        self.n_class = len(set(Y))
        self.n_train = len(X)
        self.X_train = np.array(X, dtype='float')
        self.Y_train = np.array(Y)

    def predict(self, X):
        """
        predict label for each testing document / sentence
        :param X: doc-term matrix of testing data
        """
        Y_pred = []
        for i in xrange(X.shape[0]):
            distance = np.zeros(self.n_train)
            for j in xrange(self.n_train):
                distance[j] = manhattan_distance(x1=X[i], x2=self.X_train[j])
            sorted_pairs = np.array(zip(self.Y_train, distance), dtype=[('x', 'int'), ('y', 'float')])
            sorted_pairs.sort(order='y')
            class_count = np.zeros(self.n_class)
            for k in xrange(K):
                y, d = sorted_pairs[k]
                class_count[y] += 1
            Y_pred.append(np.argmax(class_count))
        return Y_pred





