import math
from collections import Counter

import naive_bayes.util as util

class Classifier:

    def __init__(self, train_data, beta=1):
        self.counts = Counter()
        spam = 0
        total = 0
        spam_prior = Counter()
        ham_prior = Counter()
        for row in util.get_rows(train_data):
            if row['spam']:
                spam += 1
                del row['spam']
                spam_prior.update(row)
            else:
                del row['spam']
                ham_prior.update(row)
            total += 1
        ham = total - spam
        prior = ((ham + beta - 1) / (total + 2*beta - 2),
                 (spam + beta - 1) / (total + 2*beta - 2))
        for k, v in spam_prior.items():
            spam_prior[k] = (v + beta -1) / (spam + 2*beta - 2)
        for k, v in ham_prior.items():
            ham_prior[k] = (v + beta -1) / (ham + 2*beta - 2)
        self.weights = {}
        self.weights['base'] = math.log(prior[1]/prior[0])
        for k in set(spam_prior) | set(ham_prior):
            self.weights['base'] += math.log((1 - spam_prior[k]) /
                                             (1 - ham_prior[k]))
            self.weights[k] = (math.log(spam_prior[k] / ham_prior[k]) -
                               math.log((1 - spam_prior[k]) /
                                        (1 - ham_prior[k])))
      
    def __str__(self):
        str_ = '{}\n'.format(self.weights['base'])
        for k in sorted(self.weights):
            if k != 'base':
                str_ += '{}\t{}\n'.format(k, self.weights[k])
        return str_

    def classify(self, row):
        class_ = row['spam']
        del row['spam']
        score = self.weights['base']
        for k, v in row.items():
            if v:
                score += self.weights[k]
        prob = 1 / (1 + math.exp(-1 * score))
        return prob, class_
