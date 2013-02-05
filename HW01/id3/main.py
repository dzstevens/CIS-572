import sys
import id3
import id3.util as util
import id3.id3_tree

p_table = {'0.10':2.706, '0.05':3.841, '0.025':5.024, '0.01':6.635,
           '0.001': 10.828, 'Do Not Use Chi-Squared Test':False}
choices = {'1':'0.10', '2':'0.05', '3':'0.025', '4':'0.01', '5':'0.001',
           'N':'Do Not Use Chi-Squared Test', 'Q':'Quit'}

def main(training_data_file, test_data_file, model_file):
    choice = -1
    print('What p-value would you like for chi-squared test?\n')
    for k in sorted(choices):
        print('{}: {:<5}'.format(k, choices[k]))
    while True:
        choice = input('\nChoice: ').upper()[0]
        if choice == 'Q':
            sys.exit(0)
        if choice not in choices:
            print('Invalid choice!')
        else:
            break
    chi = p_table[choices[choice]]
    tree = id3.id3_tree.grow_tree(training_data_file, chi)
    tree.dump_model(model_file)
    answers = []
    for row, class_ in util.load_test(test_data_file):
        values = {k:v for k,v in zip(tree.headers, row)}
        answers.append(tree.classify(values) == class_)
    accuracy = sum(answers) / len(answers)
    print('\nAccuracy: {:.2%}'.format(accuracy))
    
if __name__ == '__main__':
    if sys.version_info[0] != 3:
        sys.stderr.write('This program is written for Python 3.x. You are using Python {0}.{1}. '
                         'Please switch to Python 3.x and try again!\n'.format(*sys.version_info[:2]))
        sys.exit(1)
    args = sys.argv
    if len(args) != 4:
        sys.stderr.write('Proper usage:\n\n\tpython -m id3.main <training_input_file> <test_input_file> <model_output_file>\n\n')
        sys.exit(1)
    for f_name in args[1:3]:
        util.verify(f_name)
    main(*args[1:]) 
