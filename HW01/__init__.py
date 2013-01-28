import errno
import sys
try:
    import numpy
except ImportError:
        sys.stderr.write('This program is written using Numpy. You do not seem to have Numpy installed '
                         'for this version of Python.\nPlease install Numpy and try again!\n'.format(*sys.version_info[:2]))
        sys.exit(1)

def get_counts(column, classes):
    counts = {}
    for let in set(column):
        filter_ = np.array([i == let for i in column])
        split = classes[filter_]
        p = sum(split)
        n = len(split) - p
        counts[let] = (p, n)
    return counts

def expected_info(p,n):
    sum_ = p + n
    p_ratio = p / sum_
    n_ratio = n / sum_
    return -1 * (p_ratio * np.log2(p_ratio) + 
                 n_ratio * np.log2(n_ratio))

def mutual_info(column, classes):
    counts = get_counts(column, classes).values()
    return sum([(p+n) * expected_info(p, n)
                for p,n in counts]) / len(column)

def best_split(data):
    data = data.T
    return np.argmin([mutual_info(row, data[-1]) for row in data[:-1]])

def split(data, index):
    filter_ = np.array([i != index for i in range(len(data[0]))])
    left = data[data[...,index]==0][...,filter_]
    right = data[data[...,index]==1][...,filter_]
    return left, right

def grow_tree(data_file):
    util.verify(data_file)
    data = numpy.loadtxt(data_file, skiprows=1, delimiter=',')
    with open(data_file) as f:
        headers = f.readline().strip().split(',')
    return tree.ID3Tree(data, headers)

class ID3Tree(tree.Tree):
    
    def __init__(self, data, headers, depth=0):
        self.headers = headers
        best_attr = id3.best_split(data)
        super().__init__(headers[best_attr], depth)
        left_data, right_data = id3.split()
        self.left = split_value(left_data)
        self.right = split_value(right_data)

    def split_value(self, data):
        if len(data) == 0:
            return None
        elif len(left_data.T) == 2:
            return argmax(bincount(data.T[-1]))
        else:
            new_headers = self.headers[:best_attr] + self.headers[best_attr+1:]
            return ID3Tree(data, new_headers, self.depth+1)

    def classify(self, values):
        if self.right is None:
            return self.left
        if self.left is None:
            return self.right
        try:
            next_child = self.right if values[self.name] else self.left
        except LookupError:
            # instance doesn't have this attribute labeled, so always choose left
            next_child = self.left
        return next_child if isinstance(next_child, int) \
                          else next_child.classify(values)
