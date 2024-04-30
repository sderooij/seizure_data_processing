"""
    Post processing of the model output.
"""

import numpy as np


def moving_average_filter(data, window_size):
    """
    Apply a moving average filter to the data.

    Parameters
    ----------
    data : np.ndarray
        The data to filter.
    window_size : int
        The size of the window for the moving average filter.

    Returns
    -------
    np.ndarray
        The filtered data.
    """
    # pad data to reduce boundary effects and keep the same length
    data_padded = np.pad(data, (window_size // 2, window_size - 1 - window_size // 2), mode="edge")
    return np.convolve(data_padded, np.ones(window_size) / window_size, mode="valid")


def remove_overlap(labels, pos_overlap=0.9, neg_overlap=0.0, desired_overlap=0.0):
    """
    Remove overlapping windows from test data. Only used for the testing.

    Args:
        labels: (np.array) the true labels to remove the overlap from.
        pos_overlap: (float) the overlap between seizures.
        neg_overlap: (float) the overlap between background.
        desired_overlap: (float) the desired overlap between the windows.

    Returns:
        (np.array) index of the samples to keep.
    """
    # positive labels
    pos_idx = np.where(labels == 1)[0]
    keep_x_idx = int((1 - desired_overlap) / (1 - pos_overlap))
    pos_idx = pos_idx[::keep_x_idx]
    # negative labels
    neg_idx = np.where(labels == -1)[0]
    keep_x_idx = int((1 - desired_overlap) / (1 - neg_overlap))
    neg_idx = neg_idx[::keep_x_idx]
    # merge the index
    idx = np.concatenate((pos_idx, neg_idx))
    # sort the index low to high to get the correct order
    idx = np.sort(idx)

    return idx


def stitch_seizures(predicted_labels, arp, sample_duration, overlap):
    """
    Stitch seizures together that are separated by less than the absolute refractory period for event based scoring.
    To avoid double warnings for the same seizure.

    Args:
        predicted_labels: (np.array) the predicted labels (1D).
        arp: (float) the absolute refractory period, i.e. minimal time between two seizures.
        sample_duration: (float) the duration of a sample in seconds.
        overlap: (float) the overlap between the windows.

    Returns:
        (np.array) the stitched labels.
    """
    # get the window size
    arp_size = int(arp / (sample_duration * (1 - overlap)))
    # get the sliding view
    predicted_labels = np.squeeze(predicted_labels)
    # verify labels are -1 and 1, change if 0 and 1
    unique_labels = np.unique(predicted_labels)
    if np.array_equal(unique_labels, np.array([0, 1])):
        predicted_labels[predicted_labels == 0] = -1
    elif not np.array_equal(unique_labels, np.array([-1, 1])):
        raise ValueError("Labels should be -1 and 1.")

    diff = np.diff(predicted_labels)
    # find the start and end of the seizures
    start_seiz = np.where(diff == 2)[0]+1
    end_seiz = np.where(diff == -2)[0]
    # check first and last value of the predicted labels
    if predicted_labels[0] == 1:
        start_seiz = np.concatenate(([0], start_seiz))
    if predicted_labels[-1] == 1:
        end_seiz = np.concatenate((end_seiz, [len(predicted_labels)]))
    # stitch the seizures together
    stitched_labels = predicted_labels.copy()
    start_seiz = start_seiz[1:]
    end_seiz = end_seiz[:-1]
    for start, end in zip(start_seiz, end_seiz):
        if start - end <= arp_size:
            stitched_labels[end:start] = 1

    return stitched_labels


if __name__ == "__main__":
    # a = np.arange(20)
    # a[10] = 0
    # print(a)
    # am = moving_average_filter(a, 5)
    # print(am)
    # import matplotlib.pyplot as plt
    # plt.plot(a)
    # plt.plot(am)
    # plt.show()
    #
    # b = np.ones(20)
    # b[5] = 0
    # b[10] = 2
    # print(b)
    # bm = moving_average_filter(b, 3)
    # print(bm)

    predicted_labels = np.concatenate(
        [
            1 * np.ones((5, 1)),
            -1 * np.ones((5, 1)),
            np.ones((10, 1)),
            -1 * np.ones((20, 1)),
            np.ones((10, 1)),
            -1 * np.ones((10, 1)),
        ]
    ).squeeze()
    arp = 6
    sample_duration = 1
    overlap = 0.
    new_labels = stitch_seizures(predicted_labels, arp, sample_duration, overlap)
