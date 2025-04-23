import numpy as np

from seizure_data_processing.config import TUSZ_DIR
from seizure_data_processing.datasets import tusz


def test_load_annotations():

    file = (
        TUSZ_DIR
        + "/edf/train/01_tcp_ar/012/00001204/s002_2004_09_29/00001204_s002_t000.edf"
    )
    ann = tusz.load_annotations(file)
    assert len(ann) == 0

    file = (
        TUSZ_DIR
        + "/edf/train/01_tcp_ar/027/00002707/s001_2006_03_17/00002707_s001_t001.edf"
    )
    ann = tusz.load_annotations(file)
    assert len(ann) == 1
    ann = ann.to_numpy()
    ann_target = np.array([1.0, 21.0606])
    assert np.allclose(ann[:, 0:2].astype(float), ann_target)
    assert ann[:, 2] == "fnsz"
    assert ann[:, 3] == 1.0
