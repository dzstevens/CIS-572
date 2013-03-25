import math
from collections import Counter, defaultdict

import classify.util as util


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0

def magnitude(X):
    return math.sqrt(sum(x**2 for x in X))

def cond_log_likelihood(weights, row):
    return sigmoid(sum(v * weights[k] for k,v in row.items()))

class Classifier:

    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        str_ = '{}\n'.format(self.weights['base'])
        for k in sorted(self.weights):
            if k != 'base':
                str_ += '{}\t{}\n'.format(k, round(self.weights[k], 4))
        return str_

    def score(self, row, false=0, target='spam'):
        try:
            del row[target]
        except KeyError:
            pass
        score = self.weights['base']
        for k, v in row.items():
            if v != 1:
                v = false
            score += self.weights[k] * v
        return score

    def get_prob(self, row, target='spam'):
        return sigmoid(self.score(row, target=target))

    def classify(self, row, target='spam'):
        x = round(self.get_prob(row, target=target))
        return x

    def dump(self, model_file):
        with open(model_file, 'w') as f:
            f.write(str(self))


class NaiveBayesClassifier(Classifier):

    def __init__(self, train_data, beta=1, t='spam'):
        self.counts = Counter()
        beta = int(beta)
        spam = 0
        total = 0
        spam_prior = Counter()
        ham_prior = Counter()
        for row in util.get_rows(train_data):
            if row[t]:
                spam += 1
                del row[t]
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
    
    def __init__(self, train_data, eta, sigma, t='spam'):
        eta = float(eta)
        sigma = float(sigma)
        headers = ['base'] + util.get_headers(train_data)
        self.weights = dict.fromkeys(headers, 0)
        for i in range(100):
            gradient = dict.fromkeys(headers, 0)
            for row in util.get_rows(train_data):
                target = row[t]
                del row[target]
                row['base'] = 1
                w = target - cond_log_likelihood(self.weights, row)
                for f, x in row.items():
                    gradient[f] += x*w
            for f in self.weights:
                gradient[f] -= self.weights[f] / (sigma**2)
            if magnitude(gradient.values()) < 0.01:
                break
            for f in self.weights:
                self.weights[f] += eta * gradient[f]



class StochasticLogisticRegressionClassifier(Classifier):
    
    def __init__(self, train_data, eta, sigma, t='spam'):
        eta = float(eta)
        sigma = float(sigma)
        headers = ['base'] + util.get_headers(train_data)
        self.weights = dict.fromkeys(headers, 0)
        for i in range(100):
            gradient = defaultdict(float)
            for row in util.get_rows(train_data, shuffle=True):
                target = row[t]
                del row[t]
                row['base'] = 1
                w = target - cond_log_likelihood(self.weights, row)
                for f, x in row.items():
                    gradient[f] = x*w - (self.weights[f] / (sigma**2))
                if magnitude(gradient.values()) < 0.01:
                    break
                for f in self.weights:
                    self.weights[f] += eta * gradient[f]


class PerceptronClassifier(Classifier):

    def __init__(self, train_data, eta, false=0, t='spam'):
        eta = float(eta)
        self.false = int(false)
        headers = ['base'] + util.get_headers(train_data)
        self.weights = dict.fromkeys(headers, 0)
        for i in range(1000):
            print(i+1)
            error = False
            for row in util.get_rows(train_data, false=self.false):
                target = row[t]
                del row[t]
                output = self.classify(row, target=t)
                if output != target:
                    error = True
                    delta = eta * (target - output)
                    row['base'] = 1
                    for x in row:
                        self.weights[x] += delta * row[x]
            if not error:
                break

    def classify(self, row, target='spam'):
        return 1 if self.score(row, self.false, target=target) > 0 else self.false

    def get_prob(self, row, target='spam'):
        return sigmoid(self.score(row, self.false, target=target))

CLASSIFIERS = {'naive_bayes'   : NaiveBayesClassifier,
               'logistic'      : LogisticRegressionClassifier,
               'stochastic_lr' : StochasticLogisticRegressionClassifier,
               'perceptron'    : PerceptronClassifier}

def get_classfier(name, train_data, opts):
    try:
        return CLASSIFIERS[name](train_data, *opts)
    except KeyError:
        raise KeyError('\'{}\' is not a valid classifier!\n'.format(name))
