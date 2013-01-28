import numpy as np
from collections import Sequence

import id3
import id3.util as util

class Tree:
    def __init__(self, name, depth=0):
        self.name = name
        self._left =  0
        self._right = 1
        self.depth = depth

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        if isinstance(value, Tree):
            self._left = value
        self._left = Tree(value, depth=self.depth+1)

    @left.deleter
    def left(self):
        self._left = 0

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        if isinstance(value, Tree):
            self._right = value
        self._right = Tree(value, depth=self.depth+1)
    
    @right.deleter
    def right(self):
        self._right = 1 
    
    def __str__(self):
        str_ = '' if self.depth is 0 else '\n'
        str_ +=  '{}{} = 0 : {}'.format('| '*self.depth, 
                                         self.name, self.left)
        str_ +=  '\n{}{} = 1 : {}'.format('| '*self.depth, 
                                          self.name, self.right)
        return str_

    def dump_model(self, output_file):
        with open(output_file, 'w') as f:
            f.write('{}\n'.format(self))
