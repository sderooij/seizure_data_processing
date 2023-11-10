"""
	Helper functions for all datasets (mainly for internal use).
"""

import numpy as np
import pandas as pd


def ann_to_dataframe(seizures):

    seizures = np.array(seizures)
    if len(seizures) == 0:
        seizures = pd.DataFrame(
            columns=["start_time", "stop_time", "annotation", "probability"]
        )
    else:
        seizures = pd.DataFrame(
            seizures,
            columns=["start_time", "stop_time", "annotation", "probability"],
        )
        seizures = seizures.apply(pd.to_numeric, errors="ignore")

    return seizures
