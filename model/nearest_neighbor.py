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

    def __init__(self, K=5, dis='manhattan'):
        """
        default K value is 5
        """
        self.K = K
        self.dis = dis

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
                if self.dis == 'manhattan':
                    distance[j] = manhattan_distance(x1=X[i], x2=self.X_train[j])
                else:
                    distance[j] = euclidean_distance(x1=X[i], x2=self.X_train[j])
            sorted_pairs = np.array(zip(self.Y_train, distance), dtype=[('x', 'int'), ('y', 'float')])
            sorted_pairs.sort(order='y')
            class_count = np.zeros(self.n_class)
            for k in xrange(self.K):
                y, d = sorted_pairs[k]
                class_count[y] += 1
            Y_pred.append(np.argmax(class_count))
        return [], Y_pred


class nearest_centroid:

    def __init__(self, dis='euclidean'):
        self.dis = dis

    def train(self, X, Y, vocab):
        """
        train a nearest centroid classifier
        :param X: doc-term matrix of training documents
        :param Y: list of ground-truth label
        :param vocab: vocabulary derived from training documents
        """
        self.n_classes = len(set(Y))
        self.n_train, self.n_feature = X.shape
        self.class_centroid = []
        Y = np.array(Y)
        for i in xrange(self.n_classes):
            mask = (Y == i)
            self.class_centroid.append(X[mask].mean(axis=0))
        self.class_centroid = np.array(self.class_centroid)

    def predict(self, X):
        """
        predict label for testing documents
        :param X: doc-term matrix of training documents
        """
        # note, p_y_x will not be derived in this algorithm
        res, p_y_x = [], []
        for x in X:
            class_dis = []
            for i in xrange(self.n_classes):
                if self.dis == 'euclidean':
                    class_dis.append(euclidean_distance(x1=x, x2=self.class_centroid[i]))
                else:
                    class_dis.append(manhattan_distance(x1=x, x2=self.class_centroid[i]))
            res.append(np.argmin(class_dis))
        return p_y_x, res








