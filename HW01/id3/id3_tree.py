import sys

import id3.tree as tree
import id3.util as util

try:
    import numpy as np
except ImportError:
    sys.stderr.write('This program is written using Numpy. You do not seem to have Numpy installed '
                     'for this version of Python.\nPlease install Numpy and try again!\n'.format(*sys.version_info[:2]))
    sys.exit(1)


def get_counts(row, classes):
    counts = [(0, 0), (0, 0)]
    for val in set(row):
        val = int(val)
        filter_ = np.array([i == val for i in row])
        split_ = classes[filter_]
        p = sum(split_)
        n = len(split_) - p
        counts[val] = (n, p)
    return counts

def expected_info(n,p):
    sum_ = n + p
    if sum_ == 0:
        return 0
    n_ratio = n / sum_
    n_term = n_ratio * np.log2(n_ratio) if n else 0
    p_ratio = p / sum_
    p_term = p_ratio * np.log2(p_ratio) if p else 0
    return -1 * (n_term + p_term)

def p(depth=0,**a):
    for k,v in a.items():
        print('depth:{}\n{}:\n{}\n'.format(depth,k,v))
def mutual_info(row, classes):
    counts = get_counts(row, classes)
    return sum([(n + p) * expected_info(n, p)
                for n, p in counts]) / len(row)

def best_split(data):
    data = data
    return np.argmin([mutual_info(row, data[-1]) for row in data[:-1]])

def split(data, index):
    filter_ = np.array([i != index for i in range(len(data[...,0]))])
    left = data[...,data[index]==0][filter_]
    right = data[...,data[index]==1][filter_]
    return left, right

def grow_tree(data_file):
    util.verify(data_file)
    data = np.loadtxt(data_file, skiprows=1, delimiter=',').T
    with open(data_file) as f:
        headers = f.readline().strip().split(',')
    return ID3Tree(data, headers)

def mode(array):
    if isinstance(array, np.integer):
        return int(array)
    return int(np.argmax(np.bincount(array.astype(int))))

class ID3Tree(tree.Tree):
    
    def __init__(self, data, headers, depth=0):
        self.headers = headers
        best_attr = best_split(data)
        super().__init__(headers[best_attr], depth)
        left_data, right_data = split(data, best_attr)
        new_headers = self.headers[:best_attr] + self.headers[best_attr+1:]
        if left_data.size != 0:
            if left_data.ndim == 1 or left_data.shape[0] == 1:
                self.left = mode(left_data[-1])
            else:
                self.left = ID3Tree(left_data, new_headers, self.depth+1)
        if right_data.size != 0:
            if right_data.ndim == 1 or right_data.shape[0] == 1:
                self.right = mode(right_data[-1])
            else:
                self.right = ID3Tree(right_data, new_headers, self.depth+1)
    
    def classify(self, values):
        if self.left is None and self.right is None:
            return self.name
        elif self.right is None:
            next_child = self.left
        elif self.left is None:
            next_child = self.right
        else:    
            next_child = self.right if values[self.name] else self.left
        return next_child.name if isinstance(next_child.name, int) \
                               else next_child.classify(values)
