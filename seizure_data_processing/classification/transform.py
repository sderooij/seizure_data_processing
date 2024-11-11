"""
    File that includes sklearn Transformer classes for transforming the input data.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from functools import reduce


class AddSeizureGroupData(BaseEstimator, TransformerMixin):
    """
        For transfer active learning with seizure groups, we need to add the full seizure data to the training
        data.
    """
    def __init__(self, seizure_groups):
        self.seizure_groups = seizure_groups
        self.included_seizure_groups = None

    def fit(self, idx):
        self.included_seizure_groups = np.unique(self.seizure_groups[idx])
        return self

    def transform(self, idx):
        added_idx = [idx]
        for i, group in enumerate(self.included_seizure_groups):
            added_idx.append(np.where(self.seizure_groups == group)[0])
        idx = reduce(np.union1d, added_idx)

        return idx



