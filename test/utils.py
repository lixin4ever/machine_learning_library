__author__ = 'lixin77'


def compute_accu(Y_gold, Y_pred):
    assert len(Y_gold) == len(Y_pred)
    hit_count = 0
    for i in xrange(len(Y_gold)):
        if Y_gold[i] == Y_pred[i]:
            hit_count += 1
    return float(hit_count) / len(Y_gold)