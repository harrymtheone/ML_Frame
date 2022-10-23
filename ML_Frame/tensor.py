import numpy as np


class Tensor:
    def __init__(self, data, parent_node=None, parent_tensors=None):
        self.data = data
        self.parent_node = parent_node
        self.parent_tensors = parent_tensors

    def backward(self):
        self.parent_node.backward(self)
