import csv
import errno
import sys

import tree 

try:
    import numpy
except ImportError:
        sys.stderr.write('This program is written using Numpy. You do not seem to have Numpy installed '
                         'for this version of Python.\nPlease install Numpy and try again!\n'.format(*sys.version_info[:2]))
        sys.exit(1)


def verify(f_name):
    try:
        f = open(f_name)
    except IOError as e:
        if e.errno == errno.ENOENT:
            sys.stderr.write('\'{}\' is not a valid file name!\n'.format(f_name))
            sys.exit(1)
    else:
        f.close()

def get_headers(data_file):
    with open(data_file) as f:
        return f.readline().strip().split(',')

def load_data(data_file):
    return numpy.loadtxt(data_file, skiprows=1, delimiter=',')

def dump_model(model, output_file):
    with open(output_file, 'w') as f:
        f.write('{}\n'.format(model))

def main(training_data_file, test_data_file, model_file):
    headers = get_headers(training_data_file)
    training_data = load_data(training_data_file)
    print(training_data)
    model = tree.grow_tree(training_data)
    dump_model(model, model_file)
    
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
        verify(f_name)
    main(*args[1:]) 
