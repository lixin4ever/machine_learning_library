__author__ = 'lixin77'

from sklearn.feature_extraction.text import CountVectorizer
import re

def preprocess(sentence):
    """
    tokenize the input sentence
    :param sentence:
    """
    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", "", sentence)
    sentence = re.sub(r"\'s", " \'s", sentence)
    sentence = re.sub(r"\'ve", " \'ve", sentence)
    sentence = re.sub(r"n\'t", " n\'t", sentence)
    sentence = re.sub(r"\'re", " \'re", sentence)
    sentence = re.sub(r"\'d", " \'d", sentence)
    sentence = re.sub(r"\'ll", " \'ll", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.strip()


def build_dict(sentences):
    """
    build the dictionary(mapping) that link word to integer indices
    :param sentences:
    """
    pass

def get_word_count(sentences):
    """
    return mapping between word and word count
    :param sentences: input training sentences or documents
    """
    clear_sentences = [preprocess(sen) for sen in sentences]
    cv = CountVectorizer()
    X = cv.fit_transform(clear_sentences)
    return cv.vocabulary_, X.toarray()




