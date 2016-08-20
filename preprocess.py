__author__ = 'lixin77'

from sklearn.feature_extraction.text import CountVectorizer
import re

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




