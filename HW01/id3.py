import sys
import id3
import id3.util as util


def main(training_data_file, test_data_file, model_file):
    id3_tree = id3.grow_tree(training_data)
    id3_tree.dump_model(model_file)
    
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
