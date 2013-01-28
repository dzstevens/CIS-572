import numpy as np
from collections import Sequence

import id3
import id3.util as util

class Tree:
    def __init__(self, name, depth=0):
        self.name = name
        self._left =  None
        self._right = None
        self.depth = depth

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        if isinstance(value, Tree):
            self._left = value
            return
        self._left = Tree(value, depth=self.depth+1)

    @left.deleter
    def left(self):
        self._left = None

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        if isinstance(value, Tree):
            self._right = value
            return
        self._right = Tree(value, depth=self.depth+1)
    
    @right.deleter
    def right(self):
        self._right = None
    
    def __str__(self):
        str_ = ''
        if self.left is not None:
            str_ +=  '\n{}{} = 0 : {}'.format('| '*self.depth, 
                                              self.name, self.left)
        if self.right is not None:
            str_ +=  '\n{}{} = 1 : {}'.format('| '*self.depth, 
                                              self.name, self.right)
        if self.right is None and self.left is None:
            str_ += str(self.name)
        return str_

    def dump_model(self, output_file):
        with open(output_file, 'w') as f:
            f.write('{}\n'.format(self))
