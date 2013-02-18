import math
import progressbar
from collections import Counter

import classify.util as util


def sigmoid(x):
    return 1 / (1 + math.exp(-1 * x))


class Classifier:

    def __init__(self):
        raise NotImplementedError

   
    def __str__(self):
        str_ = '{}\n'.format(self.weights['base'])
        for k in sorted(self.weights):
            if k != 'base':
                str_ += '{}\t{}\n'.format(k, round(self.weights[k], 4))
        return str_

    def score(self, row):
        try:
            del row['spam']
        except KeyError:
            pass
        score = self.weights['base']
        for k, v in row.items():
            if v:
                score += self.weights[k]
        return score

    def get_prob(self, row):
        return sigmoid(self.score(row))

    def classify(self, row):
        return round(self.get_prob(row))

    def dump(self, model_file):
        with open(model_file, 'w') as f:
            f.write(str(self))


class NaiveBayesClassifier(Classifier):

    def __init__(self, train_data, beta=1):
        self.counts = Counter()
        beta = int(beta)
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


class LogisticRegressionClassifier(Classifier):
    
    def __init__(self, train_data, eta, sigma):
        pass


class PerceptronClassifier(Classifier):
    def __init__(self, train_data, eta):
        eta = float(eta)
        headers = util.get_headers(train_data)
        self.weights = dict.fromkeys(headers, 0)
        self.weights['base'] = 0
        p = progressbar.ProgressBar(widgets=[progressbar.Percentage()], maxval=100).start()
        for i in range(100):
            error = False
            for row in util.get_rows(train_data):
                target = row['spam']
                del row['spam']
                for x in row:
                    if x:
                        output = self.classify(row)
                        if row != target:
                            self.weights[x] += eta * (target - output)
                            error = True
            p.update(i+1)
            if not error:
                break
        p.finish()


CLASSIFIERS = {'naive_bayes' : NaiveBayesClassifier,
               'logistic'    : LogisticRegressionClassifier,
               'perceptron'  : PerceptronClassifier}

def get_classfier(name, train_data, opts):
    try:
        return CLASSIFIERS[name](train_data, *opts)
    except KeyError:
        raise KeyError('\'{}\' is not a valid classifier!\n'.format(name))
