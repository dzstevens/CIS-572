import classify.util as util
import classify.classifiers as classifiers

def main(classifier, test_file, model_file):
    classifier.dump(model_file)
    count = correct = 0
    for row in util.get_rows(test_file):
        is_spam = row['spam']
        prob = classifier.get_prob(row)
        print(round(prob, 4))
        if round(prob) == is_spam:
            correct += 1
        count += 1
    print('\nAccuracy: {:.2%}'.format(correct/count))

if __name__ == '__main__':
    import sys
    if sys.version_info[0] != 3:
        raise RuntimeError('This program is written for Python 3.x. You are using Python {0}.{1}. '
                           'Please switch to Python 3.x and try again!\n'.format(*sys.version_info[:2]))
    args = sys.argv[1:]
    for f_name in args[1:2]:
        util.verify(f_name)
    if len(args) < 4:
        raise RuntimeError('Not enough input parameters!')
    name = args[0]
    train = args[1]
    test = args[2]
    model = args[-1]
    opts = args[3:-1]
    classifier = classifiers.get_classfier(name, train, opts)
    main(classifier, test, model) 
