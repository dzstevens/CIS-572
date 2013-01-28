import sys
import id3
import id3.util as util
import id3.id3_tree


def main(training_data_file, test_data_file, model_file):
    tree = id3.id3_tree.grow_tree(training_data_file)
    tree.dump_model(model_file)
    answers = []
    for row, class_ in util.load_test(test_data_file):
        values = {k:v for k,v in zip(tree.headers, row)}
        answers.append(tree.classify(values) == class_)
    accuracy = sum(answers) / len(answers)
    print('Accuracy: {:.2%}'.format(accuracy))
    
if __name__ == '__main__':
    if sys.version_info[0] != 3:
        sys.stderr.write('This program is written for Python 3.x. You are using Python {0}.{1}. '
                         'Please switch to Python 3.x and try again!\n'.format(*sys.version_info[:2]))
        sys.exit(1)
    args = sys.argv
    if len(args) != 4:
        sys.stderr.write('Proper usage:\n\n\tpython id3.py <training_input_file> <test_input_file> <model_output_file>\n\n')
        sys.exit(1)
    for f_name in args[1:3]:
        util.verify(f_name)
    main(*args[1:]) 
