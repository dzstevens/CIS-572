import numpy as np
from collections import Sequence

class Tree:
    def __init__(self, name, flip=False, depth=0):
        self.name = name
        self._left =  self._zero_side = 0 ^ flip
        self._right = 1 ^ flip 
        self.depth = depth

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        if not isinstance(value, tuple):
            value = (value, False)
        self._left = Tree(*value, depth=self.depth+1)

    @left.deleter
    def left(self):
        self._left = 0 ^ self._zero_side

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        if not isinstance(value, tuple):
            value = (value, False)
        self._right = Tree(*value, depth=self.depth+1)
    
    @right.deleter
    def right(self):
        self._right = 1 ^ self._zero_side
    
    def __str__(self):
        str_ = '' if self.depth is 0 else '\n'
        str_ +=  '{}{} = 0 : {}'.format('| '*self.depth, 
                                         self.name, self.left)
        str_ +=  '\n{}{} = 1 : {}'.format('| '*self.depth, 
                                          self.name, self.right)
        return str_

    def classify(self, values):
        try:
            next_child = self.right if values[self.name] else self.left
        except LookupError:
            # instance doesn't have this attribute labeled, so always choose left
            next_child = self.left
        return next_child if isinstance(next_child, int) \
                          else next_child.classify(values)


def grow_tree(data_set, headers=None):
    return Tree('foo')

def split(data, index):
    filter_ = np.np.array([False if i==index else True for i in range(len(data[0]))])
    left = data[data[:,2]==0][:,data]
    right = data[data[:,2]==1][:,data]
    return left, right
