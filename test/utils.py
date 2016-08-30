__author__ = 'lixin77'

__author__ = 'lixin77'

from sklearn.feature_extraction.text import CountVectorizer
import re
import random
import numpy as np

def preprocess(sentence):
    """
    tokenize the input sentence
    :param sentence: list of input sentences
    """
    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    sentence = re.sub(r"\'s", " \'s", sentence)
    sentence = re.sub(r"\'ve", " \'ve", sentence)
    sentence = re.sub(r"n\'t", " n\'t", sentence)
    sentence = re.sub(r"\'re", " \'re", sentence)
    sentence = re.sub(r"\'d", " \'d", sentence)
    sentence = re.sub(r"\'ll", " \'ll", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    #print sentence
    return sentence.strip()


def build_dict(sentences):
    """
    build the dictionary(mapping) that link word to integer indices
    :param sentences:
    """
    vocab, indexed_vocab, word_to_count = {}, {}, {}
    word_arrays = []
    vocab_size = 0
    for sen in sentences:
        words = sen.split(' ')
        word_arrays.append(words)
        for w in words:
            try:
                word_to_count[w] += 1
            except KeyError:
                word_to_count[w] = 1
                vocab[w] = vocab_size
                indexed_vocab[vocab_size] = w
                vocab_size += 1
    print "number of distinct words in the training set:", vocab_size
    return vocab, indexed_vocab, word_to_count

def tokenization(sentences, word_to_count, vocab):
    """
    NOT USED
    """
    # feature matrix is actually sentence-word count matrix
    feature_mat = []
    feature_mat_sparse = []
    for sen in sentences:
        words = sen.split(' ')
        features = []
        features_sparse = {}
        for w in word_to_count:
            count = word_to_count[w]
            wid = vocab[w]
            if w in set(words):
                features.append(count)
                features_sparse[wid] = count
            else:
                features.append(0)
        feature_mat.append(features)
        feature_mat_sparse.append(features_sparse)
    return feature_mat, feature_mat_sparse



def build_dataset(train_sen, test_sen):
    """
    return count vectorizor of the training set, mapping between word and word count,
    :param sentences: input training sentences or documents
    """
    clear_train_sen = [preprocess(sen) for sen in train_sen]
    cv = CountVectorizer()
    # X_train is train-sentence to term matrix, whose size is n_train_sentences * |vocab|
    # X_train_sparse is the input of naive bayes implemented by myself, element is word id
    X_train = cv.fit_transform(clear_train_sen).toarray()
    X_train_sparse = []
    for x in X_train:
        x_sparse = {}
        for i in xrange(len(x)):
            word_count = x[i]
            if word_count > 0:
                x_sparse[i] = word_count
        X_train_sparse.append(x_sparse)
    clear_test_sen = [preprocess(sen) for sen in test_sen]
    # X_test is test-sentence to term matrix, whose size is n_test_sentences * |vocab|
    # X_test_sparse is sparse representation of X_test matrix, ele is word id
    X_test = cv.transform(clear_test_sen).toarray()
    X_test_sparse = []
    for x in X_test:
        x_sparse = {}
        for j in xrange(len(x)):
            word_count = x[j]
            if word_count > 0:
                x_sparse[j] = word_count
        X_test_sparse.append(x_sparse)
    vocab = cv.vocabulary_
    return X_train, X_train_sparse, X_test, X_test_sparse, vocab

def compute_accu(Y_gold, Y_pred):
    assert len(Y_gold) == len(Y_pred)
    hit_count = 0
    for i in xrange(len(Y_gold)):
        if Y_gold[i] == Y_pred[i]:
            hit_count += 1
    return float(hit_count) / len(Y_gold)

def cv(data_path, models, model_names, k=10):
    """
    perform K-fold cross validation on the dataset
    :param data_path: path of the data file
    :param models: models used in the experiment
    :param model_names: name of models
    :param k: number of fold, default value is 10, i.e., 10-fold cross-validation will be performed
    in the default case
    """
    data = {}
    n_sample = 0
    n_models = len(models)
    with open(data_path, 'r') as fp:
        for line in fp:
            label, text = line.strip().split('\t')
            label = int(label)
            try:
                data[label].append(line.strip())
            except KeyError:
                data[label] = [line.strip()]
            n_sample += 1
    for label in data:
        random.shuffle(data[label])
    perf = np.zeros(n_models)
    for i in xrange(k):
        print "in the round", i
        train_sen = []
        test_sen = []
        Y_train = []
        Y_test = []
        for label in data:
            n_one_fold = int(len(data[label]) * 0.1)
            for j in xrange(len(data[label])):
                y, x = data[label][j].split('\t')
                y = int(y)
                if i * n_one_fold <= j < (i + 1) * n_one_fold:
                    test_sen.append(x)
                    Y_test.append(y)
                else:
                    train_sen.append(x)
                    Y_train.append(y)
        Y_test = np.array(Y_test)
        X_train, X_train_sparse, X_test, X_test_sparse, vocab = build_dataset(train_sen=train_sen, test_sen=test_sen)
        for j in xrange(n_models):
            m = models[j]
            m_name = model_names[j]
            if j % 2:
                m.fit(X_train, Y_train)
            else:
                m.train(X=X_train, Y=Y_train, vocab=vocab)
            if j % 2:
                Y_pred = m.predict(X_test)
            else:
                Y_pred, p_y_x = m.predict(X=X_test)
            accu = compute_accu(Y_gold=Y_test, Y_pred=Y_pred)
            perf[j] += accu
            print '%s: %s' % (m_name, accu)
    for i in xrange(n_models):
        print '%s: %s' % (model_names[i], perf[i] / k)

