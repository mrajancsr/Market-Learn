"""Implementation of ID3(Iterative Dichotomiser 3) Algorithm
based on Mitchell (1997)

Author: Rajan Subramanian
Date: 11/10/2020

Notes:
    - todo Also include the c4.5 algorithm of Quinian
"""

from marketlearn.algorithms.trees.general_tree import GTree
from typing import Any, Union
import numpy as np


class ID3(GTree):
    """ID3 algorithm built from General Trees using linked lists

    The class inherits from GTree class which is a general tree

    Parameters
        ----------
        criterion : str, optional, default='entropy'
            measures quality of split.  Supported criteria
            is 'entropy' for information gain
        max_depth : int, optional, default=None
            maximum depth of tree.  If None, then nodes are expanded
            until leaves are pure or until all leaves contain less
            than min_samples_split samples
        min_samples_split : int, optional, default=2
            minimum number of samples required to split internal node
        min_samples_leaf : int, optional, default=1
            the minimum number of samples required to split internal node
        prune : bool, optional
            if True, prue tree, by default False
    """

    def __init__(self,
                 criterion: str = 'entropy',
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 prune: bool = False):
        """Default Constructor used to initialize decision tree"""
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.prune = prune

    def entropy(self, system: np.ndarray) -> float:
        """calculates entropy of system

        Parameters
        ----------
        system : np.ndarray
            the system of interest

        Returns
        -------
        float
            entropy of system
        """
        _, counts = np.unique(system, return_counts=True)
        prob = counts / system.sum()
        return -(prob * np.log(prob)).sum()

    def information_gain(self, data: np.ndarray, p):
        pass
