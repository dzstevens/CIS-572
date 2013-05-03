import pickle
import numpy as np


def test(classifier, data, metric='prob', threshold=0.5, max_notes=None):
    with open('data/test_performance.pkl', 'rb') as f:
        performance = pickle.load(f) - 1
    if max_notes is None:
        max_notes=len(performance)
    with open('data/test_rates.pkl', 'rb') as f:
        rates = pickle.load(f)
    probs = classifier.predict_proba(data[0])[:,1]
    if metric == 'prob':
        k = lambda i: probs[i] if probs[i] >= threshold else 0
    elif metric == 'rate':
        k = lambda i: rates[i] if probs[i] >= threshold else 0
    elif metric == 'utility':
        k = lambda i: rates[i]*probs[i] if probs[i] >= threshold else 0
    choices = [i for i in sorted(range(len(data[0])), reverse=True, key=k)
               if k(i) > 0][:max_notes]
    if len(choices) != max_notes:
        print('Only {} of {} notes used!'.format(len(choices), max_notes))
    else:
        print('All {} notes used!'.format(max_notes))
    return '{:.2%} return!'.format(np.mean(performance[choices]))


def train(classifier, data, metric='prob', threshold=0.5, max_notes=None):
    with open('data/train_performance.pkl', 'rb') as f:
        performance = pickle.load(f) - 1
    if max_notes is None:
        max_notes=len(performance)
    with open('data/train_rates.pkl', 'rb') as f:
        rates = pickle.load(f)
    probs = classifier.predict_proba(data[0])[:,1]
    if metric == 'prob':
        k = lambda i: probs[i] if probs[i] >= threshold else 0
    elif metric == 'rate':
        k = lambda i: rates[i] if probs[i] >= threshold else 0
    elif metric == 'utility':
        k = lambda i: rates[i]*probs[i] if probs[i] >= threshold else 0
    choices = [i for i in sorted(range(len(data[0])), reverse=True, key=k)
               if k(i) > 0][:max_notes]
    if len(choices) != max_notes:
        print('Only {} of {} notes used!'.format(len(choices), max_notes))
    else:
        print('All {} notes used!'.format(max_notes))
    return '{:.2%} return!'.format(np.mean(performance[choices]))

def get_counts(classifier, data, metric='prob', threshold=0.5, max_notes=None):
    with open('data/train_performance.pkl', 'rb') as f:
        performance = pickle.load(f) - 1
    if max_notes is None:
        max_notes=len(performance)
    with open('data/train_rates.pkl', 'rb') as f:
        rates = pickle.load(f)
    probs = classifier.predict_proba(data[0])[:,1]
    if metric == 'prob':
        k = lambda i: probs[i] if probs[i] >= threshold else 0
    elif metric == 'rate':
        k = lambda i: rates[i] if probs[i] >= threshold else 0
    elif metric == 'utility':
        k = lambda i: rates[i]*probs[i] if probs[i] >= threshold else 0
    choices = [i for i in sorted(range(len(data[0])), reverse=True, key=k)
               if k(i) > 0][:max_notes]
    if len(choices) != max_notes:
        print('Only {} of {} notes used!'.format(len(choices), max_notes))
    else:
        print('All {} notes used!'.format(max_notes))
    c = data[0][choices]
    counts = {}
    for i, l in enumerate('ABCDEFG', 15):
        counts[l] = len(c[:,i][c[:,i] >0])
    return counts
