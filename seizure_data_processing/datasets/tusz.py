"""
    Functions needed to load files from the TUH Seizure Corpus (TUSZ).
"""

import numpy as np
import pandas as pd

# internal imports
from seizure_data_processing.datasets.helper_functions import ann_to_dataframe


def load_tse(tse_file: str, dataframe: bool = False):
    """function: loadTSE Load seizure events from a TSE file.

    Args:
      tse_file: TSE event file
      dataframe (bool): output as dataframe. Default to False.

    return:
      seizures: output list of seizures. Each event is tuple of 4 items:
                 (seizure_start [s], seizure_end [s], seizure_type, probability)
    """
    assert ".tse" in tse_file
    VERSION = "version = tse_v1.0.0\n"
    SEIZURES = (
        "seiz",
        "fnsz",
        "gnsz",
        "spsz",
        "cpsz",
        "absz",
        "tnsz",
        "cnsz",
        "tcsz",
        "atsz",
        "mysz",
        "nesz",
    )
    seizures = list()

    # Parse TSE file
    #
    with open(tse_file, "r") as tse:
        firstLine = tse.readline()

        # Check valid TSE
        if firstLine != VERSION:
            raise ValueError(
                'Expected "{}" on first line but read \n {}'.format(VERSION, firstLine)
            )

        # Skip empty second line
        tse.readline()

        # Read all events
        for line in tse.readlines():
            fields = line.split(" ")

            if fields[2] in SEIZURES:
                # Parse fields
                start = float(fields[0])
                end = float(fields[1])
                seizure = fields[2]
                prob = float(fields[3][:-1])

                seizures.append((start, end, seizure, prob))

    if dataframe:
        seizures = ann_to_dataframe(seizures)

    return seizures


def get_duration_tse(tse_file: str):
    """function: loadTSE Load seizure events from a TSE file.

    Args:
      tse: TSE event file

    return:
      stop_time (s), float
    """
    VERSION = "version = tse_v1.0.0\n"

    # Parse TSE file
    #
    with open(tse_file, "r") as tse:
        firstLine = tse.readline()
        # Check valid TSE
        if firstLine != VERSION:
            raise ValueError(
                'Expected "{}" on first line but read \n {}'.format(VERSION, firstLine)
            )

        lines = tse.readlines()
        lastLine = lines[-1]
        fields = lastLine.split(" ")
        stop_time = float(fields[1])

    return stop_time


def load_annotations(file: str) -> pd.DataFrame:
    """load annotations and output as a pandas dataframe.

    Args:
        file (str): edf file to annotate

    Returns:
        pd.DataFrame: with columns [start_time, stop_time, seizure_type, probability]
    """

    if ".edf" in file:
        file = file.replace(".edf", ".tse")

    ann_df = load_tse(file, dataframe=True)

    return ann_df
