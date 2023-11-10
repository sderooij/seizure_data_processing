import numpy as np

from seizure_data_processing.config import MIT_CHB_DIR
from seizure_data_processing.datasets import mit_chb as chb


def test_parse_annotations():
    # ======= Test case 1 ==================
    summary_file = MIT_CHB_DIR + "chb01/chb01-summary"
    files = ["chb01_26.edf", "chb01_15.edf", "chb01_11.edf"]
    edf_files = [MIT_CHB_DIR + "/chb09/" + file for file in files]

    lengths = [1, 1, 0]
    start_times = [
        1862.0,
        1732.0,
    ]
    end_times = [
        1963.0,
        1772.0,
    ]
    for i, file in enumerate(edf_files):
        ann = chb.parse_annotations(summary_file, file, dataframe=False)
        assert len(ann) == lengths[i]
        if len(ann) > 0:
            assert ann[0][0] == start_times[i]
            assert ann[0][1] == end_times[i]

    # -============== Test case 2 ===================
    summary_file = MIT_CHB_DIR + "chb09/chb09-summary.txt"
    files = ["chb09_08.edf", "chb09_06.edf", "chb09_15.edf"]
    edf_files = [MIT_CHB_DIR + "/chb09/" + file for file in files]

    lengths = [2, 1, 0]
    start_times = [
        [2951.0, 9196.0],
        [12231.0],
    ]
    end_times = [
        [3030.0, 9267.0],
        [12295.0],
    ]
    for i, file in enumerate(edf_files):
        ann = chb.parse_annotations(summary_file, file, dataframe=True)
        assert len(ann) == lengths[i]
        if len(ann) > 0:
            assert np.all(ann["start_time"].values == start_times[i])
            assert np.all(ann["stop_time"].values == end_times[i])
            assert not ann.empty
        else:
            assert ann.empty

    # =================== Test case 3 =================================
    summary_file = MIT_CHB_DIR + "chb24/chb24-summary.txt"
    files = ["chb24_04.edf", "chb24_02edf"]
    edf_files = [MIT_CHB_DIR + "/chb24/" + file for file in files]

    lengths = [3, 0]
    start_times = [[1088.0, 1411.0, 1745.0]]
    end_times = [[1120.0, 1438.0, 1764.0]]

    for i, file in enumerate(edf_files):
        ann = chb.parse_annotations(summary_file, file, dataframe=True)
        assert len(ann) == lengths[i]
        if len(ann) > 0:
            assert np.all(ann["start_time"].values == start_times[i])
            assert np.all(ann["stop_time"].values == end_times[i])
            assert not ann.empty
        else:
            assert ann.empty


def test_load_annotations():
    files = ["chb09_08.edf", "chb09_06.edf", "chb09_15.edf"]
    edf_files = [MIT_CHB_DIR + "/chb09/" + file for file in files]

    lens = [2, 1, 0]
    start_times = [
        [2951.0, 9196.0],
        [12231.0],
    ]
    end_times = [
        [3030.0, 9267.0],
        [12295.0],
    ]
    for i, file in enumerate(edf_files):
        ann = chb.load_annotations(file)
        assert len(ann) == lens[i]
        if len(ann) > 0:
            assert np.all(ann["start_time"].values == start_times[i])
            assert np.all(ann["stop_time"].values == end_times[i])
            assert not ann.empty
        else:
            assert ann.empty
