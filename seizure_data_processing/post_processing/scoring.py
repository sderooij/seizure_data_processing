"""

    Scoring functions for seizure detection.

"""

import numpy as np
import collections


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def event_scoring(
    predicted_labels,
    test_labels,
    overlap=0.0,
    sample_duration=2.0,
    min_duration=10.0,
    pos_percent=0.8,
    arp=30.0,
):
    """
    Calculate the scores for the test data using an event based scoring method.
    Args:
        predicted_labels: (np.array) the predicted labels.
        test_labels: (np.array) the true labels.
        overlap: (float) the overlap between the windows. In the range [0, 1]. Defaults to 0.
        sample_duration: (float) the duration of a sample in seconds.
        min_duration: (float) the minimum duration of a seizure in seconds.
        pos_percent: (float) the percentage of positive samples in a window.
        arp: (float) absolute refractory period, i.e. period between two consecutive 'warnings' (in seconds)

    Returns:
        (dict) the scores for the test data. Defaults to Recall and FPR.
    """
    # filter out pos_overlap
    # desired_overlap = bckg_overlap
    # idx = remove_overlap(test_labels, seiz_overlap, bckg_overlap, desired_overlap)
    # test_features = test_features[idx]
    # test_labels = test_labels[idx]
    #
    # # get the predictions
    # predicted_labels = model.predict(test_features)
    # move sliding window over the predictions
    window_size = int(min_duration / (sample_duration*(1-overlap)))
    predicted_labels = np.squeeze(predicted_labels)
    sliding_view_pred = np.lib.stride_tricks.sliding_window_view(
        predicted_labels, window_size
    )
    # sum_pred = np.sum(sliding_view_pred, axis=1)
    min_pos_samp = int(pos_percent * window_size)
    # do the same for the labels
    test_labels = np.squeeze(test_labels)
    sliding_view_labels = np.lib.stride_tricks.sliding_window_view(
        test_labels, window_size
    )
    # sum_labels = np.sum(sliding_view_labels, axis=1)

    num_true_pos = 0
    num_false_pos = 0
    # last_pos = False
    # last_true_pos = False
    n_arp = arp / (sample_duration*(1-overlap))
    arp_counter = 0
    alarm = False
    n_alarm = 0
    alarms = []     # store the alarms  0: false alarm, 1: true alarm
    for idx, window in enumerate(sliding_view_pred):        # TODO: what if false positive becomes true positive?
        if np.sum(window==1) >= min_pos_samp and not alarm:
            alarm = True
            n_alarm += 1
            arp_counter = 0
            if np.sum(sliding_view_labels[idx]==1) >= 1:    # true alarm if at least one positive label in the window
                alarms.append(1)
            else:
                alarms.append(0)
        elif np.sum(window==1) >= min_pos_samp and alarm:
            if np.sum(sliding_view_labels[idx]==1) >= 1 and alarms[n_alarm-1] == 0:  # false to true alarm
                alarms[n_alarm-1] = 1
            arp_counter = 0
        elif alarm and arp_counter <= n_arp:
            arp_counter += 1
        else:
            alarm = False
            arp_counter += 1

        # if np.sum(window) >= min_pos_samp and arp_counter >= n_arp:
        #     if last_pos:
        #         continue
        #     else:
        #         if np.sum(sliding_view_labels[idx]) >= -window_size+1: # at least one positive label in the window
        #             num_true_pos += 1
        #             last_true_pos = True
        #         else:
        #             num_false_pos += 1
        #             last_true_pos = False
        #
        #         arp_counter = 0
        #         last_pos = True
        # # elif (last_pos == True) and (np.sum(sliding_view_labels[idx]) >= -window_size+1):
        # #     continue
        # elif np.sum(window) >= min_pos_samp and arp_counter < n_arp:
        #     arp_counter += 1
        #     if np.sum(sliding_view_labels[idx]) >= -window_size+1 and not last_true_pos:
        #         num_true_pos += 1
        #         num_false_pos -= 1
        #         last_true_pos = True
        #
        # else:
        #     last_pos = False
        #     arp_counter += 1

    # # check for clusters of positive labels
    # for idx, val in enumerate(sum_pred):
    #     if val >= min_pos_samp:
    #         if last_pos:
    #             continue
    #         else:  # new detection
    #             if sum_labels[idx] >= 0:
    #                 num_true_pos += 1
    #             else:
    #                 num_false_pos += 1
    #             last_pos = True
    #     elif (last_pos == True) and (sum_labels[idx] >= 0):
    #         continue
    #     else:
    #         last_pos = False
    #
    # # count the number of true positives
    counter = collections.Counter(alarms)
    num_false_pos = counter[0]
    num_true_pos = counter[1]
    # count total number of seizure events
    num_seiz = np.sum(np.diff(test_labels) == 2)  # count transitions from -1 to 1
    assert (
        num_true_pos <= num_seiz
    ), f"number of true positives {num_true_pos} is greater than number of seizures {num_seiz}"
    # get the labels
    recall = num_true_pos / num_seiz
    total_time = (len(test_labels) * sample_duration) * (1 - overlap)  # in seconds
    # get false alarm rate per 24 hours
    fpr = num_false_pos / len(test_labels)  # per sample
    far24 = (num_false_pos / total_time) * 3600 * 24

    scores = {"Recall": recall, "FA/24hr": far24, "FPR": fpr}

    return scores


if __name__ == "__main__":
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
            np.ones((10, 1)),
            -1 * np.ones((20, 1)),
            np.ones((10, 1)),
            -1 * np.ones((20, 1)),
            np.ones((10, 1)),
        ]
    )
    overlap = 0.0
    sample_duration = 2.0
    min_duration = 10.0
    pos_percent = 0.8
    arp = 0.
    scores = event_scoring(
        predicted_labels,
        test_labels,
        overlap,
        sample_duration,
        min_duration,
        pos_percent,
        arp,
    )
    print(scores)
