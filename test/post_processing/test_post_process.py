import numpy as np
from seizure_data_processing.post_processing.post_process import *


def test_remove_overlap():

    labels = np.concatenate(
        [
            -1 * np.ones((10, 1)),
            np.ones((20, 1)),
            -1 * np.ones((10, 1)),
            np.ones((20, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    pos_overlap = 0.5
    neg_overlap = 0.0
    desired_overlap = 0.0
    idx = remove_overlap(labels, pos_overlap, neg_overlap, desired_overlap)
    new_labels = labels[idx]
    des_labels = np.concatenate(
        [
            -1 * np.ones((10, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    assert np.array_equal(new_labels, des_labels)

def test_stitch_seizures():
    predicted_labels = np.concatenate(
        [
            1 * np.ones((5, 1)),
            -1 * np.ones((5, 1)),
            np.ones((10, 1)),
            -1 * np.ones((20, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    arp = 6
    sample_duration = 1
    overlap = 0.
    new_labels = stitch_seizures(predicted_labels, arp, sample_duration, overlap)
    des_labels = np.concatenate(
        [
            1 * np.ones((20, 1)),
            -1 * np.ones((20, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
        ]
    ).squeeze()
    assert np.array_equal(new_labels, des_labels)
