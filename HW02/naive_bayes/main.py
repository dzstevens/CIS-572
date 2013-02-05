import sys
import naive_bayes.util as util
import naive_bayes.naive_bayes as nb

def main(train, test, beta, model):
    classifier = nb.Classifier(train, beta)
    util.dump_model(classifier, model)
    count = correct = 0
    for row in util.get_rows(test):
        prob, is_spam = classifier.classify(row)
        print(round(prob, 4))
        if round(prob) == is_spam:
            correct += 1
        count += 1
    print('\nAccuracy: {:.2%}'.format(correct/count))

if __name__ == '__main__':
    if sys.version_info[0] != 3:
        sys.stderr.write('This program is written for Python 3.x. You are using Python {0}.{1}. '
                         'Please switch to Python 3.x and try again!\n'.format(*sys.version_info[:2]))
        sys.exit(1)
    args = sys.argv
    if len(args) != 5:
        sys.stderr.write('Proper usage:\n\n\tpython -m naive_bayes.main <train> <test> <beta> <model>\n\n')
        sys.exit(1)
    for f_name in args[1:3]:
        util.verify(f_name)
    try:
        args[3] = float(args[3])
    except TypeError:
        sys.stderr.write('3rd argument (beta) must be a real number!')
        sys.exit(1)
    main(*args[1:]) 
