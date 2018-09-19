import nltk
from nltk.corpus import stopwords
import string
import numpy as np
import math
from sklearn.svm import SVC

data_path = './dataset/CR/CR.txt'

# list of data records
dataset = []

# data id
id = 0

# obtain a list of stop words from nltk
stop_words = set(stopwords.words('english'))

# list of punctuations
puncs = string.punctuation

# prototype: maketrans(x, y, z)
# x, y must be strings of equal length
# z must be a string, whose characters will be mapped to None in the result
translator = str.maketrans('', '', puncs)

# read data from the data file
with open(data_path, mode='r', encoding='UTF-8') as fp:
    for line in fp:
        record = {}
        label, sent = line.strip().split('\t')

        # use blank space to split the sentence
        sent_no_punc = sent.translate(translator)
        words = sent_no_punc.split(' ')

        record['id'] = id
        record['y'] = int(label)
        record['sentence'] = sent
        # filter the stop words
        record['words'] = [w for w in words if w not in stop_words]
        dataset.append(record)
        id += 1

# build vocabulary and document frequency vector
# vocabulary and inverse vocabulary
word2wid, wid2word = {}, {}
# document frequency
df = {}
wid = 0  # wid starts from 0
for record in dataset:
    words = record['words']
    for w in words:
        if w not in word2wid:
            word2wid[w] = wid
            wid2word[wid] = w
            df[w] = 1
            wid += 1
        else:
            df[w] += 1

# construct numerical feature vector for each sentence / record, tf-idf metric is employed
for record in dataset:
    words = record['words']

    feat = np.zeros(len(word2wid), dtype='float32')
    n_words = len(words)
    for w in set(words):
        wid = word2wid[w]
        tf = 1.0 + math.log2(words.count(w) / float(n_words))
        idf = math.log2(len(dataset) / float(df[w]) + 1.0)
        feat[wid] = tf * idf
    record['features'] = np.array(feat, dtype='float32')


# randomly split the training and testing dataset
n_samples = len(dataset)
train_ids = np.random.choice(n_samples, int(n_samples * 0.8), replace=False)
print("Obtain %s training samples and %s testing samples" % (len(train_ids), n_samples - len(train_ids)))
train_X, test_X = [], []
train_Y, test_Y = [], []
for record in dataset:
    if record['id'] in train_ids:
        train_X.append(record['features'])
        train_Y.append(record['y'])
    else:
        test_X.append(record['features'])
        test_Y.append(record['y'])

clf = SVC()
print("Training...")
clf.fit(train_X, train_Y)
print("Prediction...")
pred_Y = clf.predict(test_X)

assert len(pred_Y) == len(test_Y)

hit_count = 0
for i in range(len(pred_Y)):
    if pred_Y[i] == test_Y[i]:
        hit_count += 1

print("\n%s testing samples are correctly predicted, "
      "the accuracy is %.3lf" % (hit_count, hit_count / float(len(pred_Y))))











