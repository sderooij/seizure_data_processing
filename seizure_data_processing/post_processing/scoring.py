"""

    Scoring functions for seizure detection.

"""

import numpy as np
import collections
from sklearn.metrics import get_scorer
import pandas as pd

class SeizureScorer:
    def __init__(self, predictions, metrics, feature_file, *, columns=None):
        self.metrics = metrics
        self.feature_file = feature_file
        self.prediction_df = None
        self._load_feature_info(columns=columns)
        self.prediction_df['predictions'] = predictions

    def _load_feature_info(self, *, columns=None):
        if columns is None:
            columns = ['start_time', 'stop_time', 'filename', 'annotation', 'neurologist_annotation']
        self.prediction_df = pd.read_parquet(self.feature_file, columns=columns)
        return self

    def save_to_events(self, save_folder, *, arp=10, pos_percent=0.8, min_duration=10.0, sample_duration=2.0,
                       overlap=0.5):
        return self


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def get_scores(output, true_labels, metrics: dict):
    predicted_labels = np.sign(output).astype(int)
    # check for zeros in predicted labels
    scores = {}
    for key, value in metrics.items():
        if value == "roc_auc" or value == "average_precision":
            scores[key] = get_scorer(value)._score_func(true_labels, output)
        else:
            scores[key] = get_scorer(value)._score_func(true_labels, predicted_labels)

    return scores


def event_scoring(
    predicted_labels,
    test_labels,
    overlap=0.0,
    sample_duration=2.0,
    min_duration=10.0,
    pos_percent=0.8,
    arp=30.0,
    total_duration=None,
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

    window_size = int(min_duration / (sample_duration * (1 - overlap)))
    predicted_labels = np.squeeze(predicted_labels)
    sliding_view_pred = np.lib.stride_tricks.sliding_window_view(
        predicted_labels, window_size
    )

    min_pos_samp = int(pos_percent * window_size)

    test_labels = np.squeeze(test_labels)
    sliding_view_labels = np.lib.stride_tricks.sliding_window_view(
        test_labels, window_size
    )

    n_arp = arp / (sample_duration * (1 - overlap))
    arp_counter = 0
    alarm = False
    n_alarm = 0
    alarms = []  # store the alarms  0: false alarm, 1: true alarm
    seiz_counter = 0
    true_alarm = False
    num_true_pos = 0
    num_false_pos = 0
    last_pos = False
    for idx, window in enumerate(sliding_view_pred):
        # check if there is a seizure in the window
        if np.sum(sliding_view_labels[idx] == 1) >= 1 and not true_alarm:
            true_alarm = True
            seiz_counter += 1
        elif np.sum(sliding_view_labels[idx] == 1) == 0 and true_alarm:
            true_alarm = False

        if np.sum(window == 1) >= min_pos_samp and not alarm:  # trigger alarm
            alarm = True
            # n_alarm += 1
            arp_counter = 0  # reset the absolute refractory period counter
            if true_alarm:  # true positive
                num_true_pos += 1
                last_pos = True
            else:
                num_false_pos += 1
                last_pos = False
        elif np.sum(window == 1) >= min_pos_samp and alarm:  # alarm already triggered
            if true_alarm and not last_pos:
                num_true_pos += 1
                num_false_pos -= 1
                last_pos = True
            arp_counter = 0
        elif alarm and true_alarm and np.sum(window == 1) > 0:
            arp_counter = 0
        elif (
            alarm and arp_counter <= n_arp
        ):  # count time since alarm (for absolute refractory period)
            arp_counter += 1
        else:
            alarm = False  # reset alarm
            arp_counter += 1

    # # count the number of true positives
    # counter = collections.Counter(alarms)
    # num_false_pos = counter[0]
    # num_true_pos = counter[1]
    # count total number of seizure events
    num_seiz = np.sum(np.diff(test_labels) == 2)  # count transitions from -1 to 1
    assert (
        num_true_pos <= num_seiz
    ), f"number of true positives {num_true_pos} is greater than number of seizures {num_seiz}"
    # get the labels
    recall = num_true_pos / num_seiz
    if total_duration is None:
        total_duration = (len(test_labels) * sample_duration) * (
            1 - overlap
        )  # in seconds
    # get false alarm rate per 24 hours
    fpr = num_false_pos / len(test_labels)  # per sample
    far24 = (num_false_pos / total_duration) * 3600 * 24
    far1 = (num_false_pos / total_duration) * 3600

    scores = {"Recall": recall, "FA/24hr": far24, "FA/hr": far1}

    return scores


def labels_to_events(
    predicted_labels,
    time,
    arp=10.0,
    pos_percent=0.8,
    min_duration=10.0,
    sample_duration=2.0,
    overlap=0.5,
    to_file=False,
    file_name=None,
):
    """
    Convert the model output to seizure events with start and end times.
    Args:
        predicted_labels: np.ndarray, The model output.
        time: pd.DataFrame with start and end times of the samples
        arp: Absolute refractory period, minimal time between two consecutive seizures.
        pos_percent: Required percentage of positive samples in a window to trigger an alarm.
        min_duration: Minimal duration of a seizure.
        sample_duration: Length of a sample in seconds.
        overlap: float, Overlap between segments.
        to_file: bool, Whether to save the events to a file.
        file_name: str, The name of the file to save the events to, typically ends with '.tsv' or '.csv'.

    Returns:
        pd.DataFrame
    """
    window_size = int(min_duration / (sample_duration * (1 - overlap)))
    min_pos_samp = int(pos_percent * window_size)
    predicted_labels = np.squeeze(predicted_labels)
    # combine prediction, time and true labels
    df = pd.DataFrame(
        {
            "start_time": time["start_time"].to_numpy(),
            "stop_time": time["stop_time"].to_numpy(),
            "predicted_labels": predicted_labels,
        }
    )
    # initialize events dataframe
    events = []
    # create a sliding window view of the data
    alarm = False
    arp_counter = arp
    for window in df.rolling(window=window_size, min_periods=window_size):  # check rolling functionality
        # check if the window contains a seizure
        if (
            np.sum(window["predicted_labels"] == 1) >= min_pos_samp
            and not alarm
            and arp_counter >= arp
        ):
            alarm = True
            start_seiz = window["start_time"].iloc[0]
            end_seiz = window["stop_time"].iloc[-1]
            # add the event to the events dataframe
            events.append({"start": start_seiz, "end": end_seiz, "annotation": "seiz"}
            )
            arp_counter = 0
        elif np.sum(window["predicted_labels"] == 1) >= min_pos_samp and alarm:
            arp_counter = 0
            # modify end time of the last event
            end_seiz = window["stop_time"].iloc[-1]
            events[-1]["end"] = end_seiz
        elif np.sum(window["predicted_labels"] == 1) <= 1 and alarm:
            arp_counter += 1
            end_seiz = window["start_time"].iloc[0]
            events[-1]["end"] = end_seiz
            alarm = False
        else:
            arp_counter += 1
    if len(events)==0:
        events = pd.DataFrame(columns=["start", "end", "annotation"])
    else:
        events = pd.DataFrame(events)
    if to_file:
        if file_name.endswith(".tsv"):
            events.to_csv(file_name, sep="\t", index=False)
        else:
            events.to_csv(file_name, index=False)

    return events


def any_ovlp(hyp, ref):
    """
    Any overlap event scoring. For now all seizure types are permissible.
    Args:
        hyp: hypothesis annotations (start, stop, annotation)
        ref: reference annotations (start, stop, annotation)

    Returns:
        tuple (hits, misses, false_alarms)
    """

    hits = 0
    misses = 0
    false_alarms = 0

    for i, event in enumerate(ref):
        hyp_events = get_events(hyp, event["start"], event["stop"])
        if len(hyp_events) == 0 or np.all(hyp_events["annotation"] == "bckg"):
            misses += 1
        else:
            hits += 1

    for i, event in enumerate(hyp):
        ref_events = get_events(ref, event["start"], event["stop"])
        if len(ref_events) == 0 or np.all(ref_events["annotation"] == "bckg"):
            false_alarms += 1

    return hits, misses, false_alarms


def get_events(events, start, stop, *, mode="any-ovlp"):
    """
    Get the events that overlap with the start and stop times.
    Args:
        events: pd.DataFrame with columns ['start_time', 'stop_time', 'annotation', (opt) 'comments',
        (opt) 'probability']
        start: float, start time
        stop: float, stop time
        mode: str, the mode of operation. Defaults to "any-ovlp". (TODO: implement other modes)

    Returns:
        pd.DataFrame with the events that overlap with the start and stop times.
    """
    events = events[(events["stop_time"] >= start) & (events["start_time"] <= stop)]
    return events




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
    arp = 0.0
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
