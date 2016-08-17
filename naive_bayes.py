__author__ = 'lixin77'


# naive bayes models
class NB(object):
    pass


class multinomial_NB(object):
    def __init__(self, use_prior=True):
        self.use_prior = use_prior

    def set_prior(self, alpha=0):
        self.alpha = alpha

class bernoulli_NB(object):
    def __init__(self, use_prior=True):
        self.use_prior = use_prior

    def set_prior(self, alpha=0):
        self.alpha = alpha

class gaussian_NB(object):
    pass

