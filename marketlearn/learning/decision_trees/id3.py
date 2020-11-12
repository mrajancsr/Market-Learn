"""Implementation of ID3(Iterative Dichotomiser 3) Algorithm
based on Mitchell (1997)

Author: Rajan Subramanian
Date: 11/10/2020

Notes:
    - todo Also include the c4.5 algorithm of Quinian
"""
from marketlearn.algorithms.Trees.general_tree import GTree
from typing import Any, Union
import numpy as np


class ID3(GTree):
    """ID3 algorithm built from General Trees using linked lists"""

    def __init__(self,
                 criterion: str = 'entropy',
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: Union[int, float] = 1,
                 prune: bool = False):
        """Default Constructor used to initialize decision tree

        :param criterion: the function to measure the quality of split
         Supported criteria is "entropy" for information gain
         defaults to 'entropy'
        :type criterion: string, optional
        :param max_depth: maximum depth of tree
         If None, then nodes are expanded until leaves are pure
         or until all leaves contain less than min_samples_split samples
         defaults to None
        :type max_depth: int, optional
        :param min_samples_split: the minimum number of samples
         required to split internal node
         defaults to 2
        :type min_samples_split: int, optional
        :param min_samples_leaf: minimum number of samples required to split
         an internal node
         defaults to 1
        :type min_samples_leaf: Union[int, float], optional
        :param prune: if True, prune the tree
         defaults to False
        :type prune: bool, optional
        """
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.prune = prune

    def entropy(self, target: np.ndarray):
        """calculates entropy of a system"""
        unique, counts = np.unique(target, return_counts=True)
        prob = (unique * counts) / target.sum()
        return -(prob * np.log(prob)).sum()

    def information_gain(self, data: np.ndarray, p):
        node = self._validate(p)
        pass


id3 = ID3()
print(id3)

