"""
    File that includes sklearn Transformer classes for transforming the input data.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from functools import reduce


class AddGroupsToData(BaseEstimator, TransformerMixin):
    """
        For transfer active learning with groups, we need to add all of the (positive) group data to the training
        data.
    """
    def __init__(self, groups):
        self.groups = groups
        self.included_groups = None

    def fit(self, idx):
        self.included_groups = np.unique(self.groups[idx])
        return self

    def transform(self, idx):
        added_idx = [idx]
        for i, group in enumerate(self.included_groups):
            added_idx.append(np.where(self.groups == group)[0])
        idx = reduce(np.union1d, added_idx)

        return idx



