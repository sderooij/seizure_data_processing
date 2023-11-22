"""
    Functions needed to load files from the Seize IT database.
"""

import numpy as np
import pandas as pd

# internal imports
from seizure_data_processing.datasets.helper_functions import ann_to_dataframe


def load_annotations(file: str) -> pd.DataFrame:
    """load annotations and output as a pandas dataframe.

    Args:
        file (str): edf file to annotate

    Returns:
        pd.DataFrame: with columns [start_time, stop_time, annotation, comments]
    """
    if ".edf" in file:
        file = file.replace(".edf", "_a2.tsv")
    try:
        seizures = pd.read_csv(file, sep="\t", header=None, comment="#")
    except pd.errors.EmptyDataError:
        seizures = pd.DataFrame(
            columns=["start_time", "stop_time", "annotation", "comments"]
        )
        return seizures

    seizures.columns = ["start_time", "stop_time", "annotation", "comments"]

    return seizures


if __name__ == "__main__":
    from seizure_data_processing.config import SEIZE_IT_DIR
    tsv_file = SEIZE_IT_DIR + "P_ID30/P_ID30_r1_a2.tsv"

    ann = load_annotations(tsv_file)
