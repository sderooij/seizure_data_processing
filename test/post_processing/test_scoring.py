import numpy as np
from seizure_data_processing.post_processing.scoring import *


def test_event_scoring():
    # case 1 --- predicted_labels = test_labels
    predicted_labels = np.concatenate(
        [
            -1 * np.ones((10, 1)),
            np.ones((20, 1)),
            -1 * np.ones((10, 1)),
            np.ones((20, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    test_labels = np.concatenate(
        [
            -1 * np.ones((10, 1)),
            np.ones((20, 1)),
            -1 * np.ones((10, 1)),
            np.ones((20, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    overlap = 0.0
    sample_duration = 2.0
    min_duration = 10.0
    pos_percent = 0.8
    arp = 5
    scores = event_scoring(
        predicted_labels,
        test_labels,
        overlap,
        sample_duration,
        min_duration,
        pos_percent,
        arp,
    )
    assert scores["Recall"] == 1.0
    assert scores["FA/hr"] == 0.0

    # case 2 --- predicted_labels != test_labels
    predicted_labels = np.concatenate(
        [
            -1 * np.ones((10, 1)),
            np.ones((20, 1)),
            -1 * np.ones((10, 1)),
            np.ones((20, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    test_labels = np.concatenate(
        [
            np.ones((9, 1)),
            -1 * np.ones((22, 1)),
            np.ones((8, 1)),
            -1 * np.ones((22, 1)),
            np.ones((9, 1)),
        ]
    )
    scores = event_scoring(
        predicted_labels,
        test_labels,
        overlap,
        sample_duration,
        min_duration,
        pos_percent,
        arp=0,
    )
    assert np.equal(scores["Recall"], 0.0)
    assert scores["FA/hr"] > 0.0

    # case 3 --- predicted_labels != test_labels in some parts
    predicted_labels = np.concatenate(
        [
            -1 * np.ones((10, 1)),
            np.ones((20, 1)),
            -1 * np.ones((10, 1)),
            -1 * np.ones((20, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    test_labels = np.concatenate(
        [
            -1 * np.ones((10, 1)),
            np.ones((20, 1)),
            -1 * np.ones((10, 1)),
            np.ones((20, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    scores = event_scoring(
        predicted_labels,
        test_labels,
        overlap,
        sample_duration,
        min_duration,
        pos_percent,
    )
    assert scores["Recall"] == 0.5
    assert scores["FA/hr"] == 0.0

    # case 4 --- predicted_labels != test_labels in some parts
    predicted_labels = np.concatenate(
        [
            -1 * np.ones((10, 1)),
            np.array([1, -1, 1, -1, 1, 1, 1, 1, 1, 1]).reshape((-1, 1)),
            -1 * np.ones((10, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    test_labels = np.concatenate(
        [
            -1 * np.ones((10, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    scores = event_scoring(
        predicted_labels,
        test_labels,
        overlap,
        sample_duration,
        min_duration,
        pos_percent,
        arp,
    )
    assert np.equal(scores["Recall"], 1.0)
    assert np.equal(scores["FA/hr"], 0.0)

    # case 5 --- 1 false positive
    predicted_labels = np.concatenate(
        [
            1 * np.ones((5, 1)),
            -1 * np.ones((5, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    test_labels = np.concatenate(
        [
            -1 * np.ones((10, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    scores = event_scoring(
        predicted_labels,
        test_labels,
        overlap,
        sample_duration,
        min_duration,
        pos_percent,
        arp=20,
    )
    assert np.equal(scores["Recall"], 1.0)
    assert np.equal(scores["FA/hr"], 0.0)

    # case 6 --- if close not false positi
    predicted_labels = np.concatenate(
        [
            -1 * np.ones((3, 1)),
            1 * np.ones((5, 1)),
            -1 * np.ones((2, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    test_labels = np.concatenate(
        [
            -1 * np.ones((10, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
        ]
    )
    scores = event_scoring(
        predicted_labels,
        test_labels,
        overlap,
        sample_duration,
        min_duration,
        pos_percent,
        arp,
    )
    assert np.equal(scores["Recall"], 1.0)
    assert np.equal(scores["FA/hr"], 0.0)


if __name__ == "__main__":
    test_event_scoring()
    print("scoring.py is correct")