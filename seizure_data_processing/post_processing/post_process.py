"""
    Post processing of the model output.
"""

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
from scipy.ndimage import uniform_filter1d


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
    # # pad data to reduce boundary effects and keep the same length
    # data_padded = np.pad(
    #     data, (window_size // 2, window_size - 1 - window_size // 2), mode="edge"
    # )
    # return np.convolve(data_padded, np.ones(window_size) / window_size, mode="valid")
    # cumsum = np.cumsum(np.insert(data_padded, 0, 0))
    # return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return uniform_filter1d(data, window_size, mode="nearest", origin=0)


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
    start_seiz = np.where(diff == 2)[0] + 1
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


def optimize_threshold(predictions, labels, metric="roc", *, N=500, min_sensitivity=None):
    """
    Optimize the threshold for the given metric of the predictions given the labels.

    Args:
        predictions: from .decision_function or .predict_proba
        labels: true labels (-1 or 1)
        metric:     - "roc": maximize the Youden's J statistic
                    - "f1": maximize the F1 score
                    - "pr": maximize the precision-recall curve, same as "f1"
                    - "f2": maximize the F2 score
                    - "f3": maximize the F3 score
                    - "sensitivity": maximize sensitivity or maximize precision for a given sensitivity
        N: (optional) number of thresholds to evaluate, default 500, TODO: use this
        min_sensitivity: (optional) minimum sensitivity to maximize precision for, default None

    Returns:
        float: the optimized threshold
    """
    if metric == "roc":
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        return thresholds[np.argmax(tpr - fpr)]  # using Youden's J statistic

    elif metric == "f1" or metric == "f2" or metric == "pr" or metric == "f3":
        precision, recall, thresholds = precision_recall_curve(
            labels, predictions, drop_intermediate=True
        )
        # get indices of zero and nan values
        zero_idx = np.where(precision <= 1e-5)[0]
        nan_idx = np.where(np.isnan(precision))[0]
        # remove zero and nan values
        precision = np.delete(precision, np.concatenate((zero_idx, nan_idx)))
        recall = np.delete(recall, np.concatenate((zero_idx, nan_idx)))
        thresholds = np.delete(thresholds, np.concatenate((zero_idx, nan_idx)))
        if metric == "f1" or metric == "pr":
            fscore = 2 * precision * recall / (precision + recall)
        elif metric == "f2":
            fscore = 5 * precision * recall / (4 * precision + recall)
        elif metric == "f3":
            fscore = 10 * precision * recall / (9 * precision + recall)
        return thresholds[np.argmax(fscore)]

    elif metric == "sensitivity":
        precision, recall, thresholds = precision_recall_curve(
            labels, predictions, drop_intermediate=True
        )
        # get indices of zero and nan values
        zero_idx = np.where(precision <= 1e-5)[0]
        nan_idx = np.where(np.isnan(precision))[0]
        # remove zero and nan values
        precision = np.delete(precision, np.concatenate((zero_idx, nan_idx)))
        recall = np.delete(recall, np.concatenate((zero_idx, nan_idx)))
        thresholds = np.delete(thresholds, np.concatenate((zero_idx, nan_idx)))
        if min_sensitivity is None:
            # maximize sensitivity
            return thresholds[np.argmax(recall)]

        else:
            # maximize precision for a given sensitivity
            idx = np.where(recall >= min_sensitivity)[0]
            if len(idx) == 0:
                return thresholds[np.argmax(recall)]
            else:
                thresholds = thresholds[idx]
                precision = precision[idx]
                return thresholds[np.argmax(precision)]
    else:
        raise ValueError("Invalid metric.")


def optimize_bias(predictions, labels, metric="roc", *, N=500, min_sensitivity=None):
    return -optimize_threshold(predictions, labels, metric, N=N, min_sensitivity=min_sensitivity)


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
    overlap = 0.0
    new_labels = stitch_seizures(predicted_labels, arp, sample_duration, overlap)
