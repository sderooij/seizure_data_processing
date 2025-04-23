"""
    Functions needed to load files from the MIT-CHB dataset.
"""
import os

import numpy as np
import pandas as pd
import re

# internal imports
from seizure_data_processing.datasets.helper_functions import ann_to_dataframe
from seizure_data_processing.datasets.tusz import load_tse


def parse_annotations(summary_file, edf_file, *, dataframe=False):
    """parse the seizure annotation from the summary file for the given edf_file.

    Args:
        summary_file (str): absolute path to the summary file
        edf_file (str): absolute path to the edf file
        dataframe (bool, optional): Output as a dataframe. Defaults to False.

    Raises:
        Exception: If edf file is not found in the summary file.

    Returns:
        tuple or DataFrame: of shape (start_time, stop_time, annotation, probability)
    """

    ext = os.path.splitext(summary_file)[1]
    if not ext == ".txt":
        summary_file = summary_file + ".txt"

    edf_file = os.path.basename(edf_file)
    # seiz_start = "Seizure Start Time:"
    # seiz_end = "Seizure End Time:"
    # initialize
    seizures = []

    with open(summary_file, "r") as f:
        lines = f.readlines()
        try:
            edf_index = [i for i, s in enumerate(lines) if edf_file in s][0]
        except:
            if "chb24" in edf_file:
                if dataframe:
                    seizures = pd.DataFrame(
                        columns=["start_time", "stop_time", "annotation", "probability"]
                    )
                return seizures
            else:
                raise Exception("edf file not in annotations")

        if "chb24" in edf_file:
            num_seiz = [int(s) for s in re.findall(r"\b\d+\b", lines[edf_index + 1])][0]
            start_index = edf_index + 2
            stop_index = start_index + 2 * num_seiz
        else:
            num_seiz = [int(s) for s in re.findall(r"\b\d+\b", lines[edf_index + 3])][0]
            start_index = edf_index + 4
            stop_index = start_index + 2 * num_seiz

        for i, line in enumerate(
            lines[start_index : stop_index + 1]
        ):  # +1 because python....
            if ("Seizure" and "Start") in line:
                line = line.split(":")[1]
                start = [float(s) for s in re.findall(r"\b\d+\b", line)][0]
            elif ("Seizure" and "End") in line:
                line = line.split(":")[1]
                stop = [float(s) for s in re.findall(r"\b\d+\b", line)][0]
                seizures.append((start, stop, "seiz", 1))

    assert num_seiz == len(
        seizures
    ), "Number of seizures not equal to amount of extracted annotations."

    if dataframe:
        seizures = ann_to_dataframe(seizures)
    return seizures


def load_annotations(file: str):
    """load annotations for the given edf file.

    Args:
        file (str): edf file to be annotated

    Returns:
        DataFrame: (start_time, stop_time, annotation, probability)
    """

    if os.path.exists(file.replace(".edf", ".tse_bi")):
        return load_tse(file.replace(".edf", ".tse_bi"), dataframe=True)
    elif os.path.exists(file.replace(".edf", ".tse")):
        return load_tse(file.replace(".edf", ".tse"), dataframe=True)

    z = re.search(r"chb\d\d", file)
    folderpath = os.path.split(os.path.abspath(file))[0]
    patient = z.group()
    summary_file = folderpath + "/" + patient + "-summary.txt"

    seizures = parse_annotations(summary_file, file, dataframe=True)
    return seizures
