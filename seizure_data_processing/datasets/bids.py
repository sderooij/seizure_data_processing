"""
Functions to load BIDS compatible datasets.
"""

import os
import numpy as np
import pandas as pd
import re

# internal imports
from seizure_data_processing.datasets.helper_functions import ann_to_dataframe



def load_annotations(file: str, *, version='seizeit2', all=False) -> pd.DataFrame:
    """load annotations and output as a pandas dataframe.

    Args:
        file (str): edf file to annotate
        version (str, optional): Version of the annotations. Defaults to 'seizeit2'.
        all (bool, optional): Whether to load all annotations. Defaults to False, which only loads seizures.

    Returns:
        pd.DataFrame: with columns [start_time, stop_time, annotation, +extra comments]
    """
    if ".edf" in file:
        ann_file = file.replace("_eeg.edf", "_events.tsv")
    else:
        ann_file = file

    annotations = pd.read_table(ann_file)
    # except pd.errors.EmptyDataError:
    #     seizures = pd.DataFrame(
    #         columns=["start_time", "stop_time", "annotation", "comments"]
    #     )
    #     return seizures
    if all:
        return annotations
    seizures = annotations.loc[annotations['eventType'].str.contains('sz'),:].copy()
    if seizures.empty:
        seizures = pd.DataFrame(
            columns=["start_time", "stop_time", "annotation", "comments"]
        )
        return seizures

    seizures.rename(columns={'onset': 'start_time', 'eventType': 'annotation'}, inplace=True)
    seizures['stop_time'] = seizures['start_time'] + seizures['duration']

    return seizures