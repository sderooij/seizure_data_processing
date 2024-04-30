import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from pywt import wavedec
from sklearn.preprocessing import StandardScaler
from functools import partial
from scipy.fft import fft, fftfreq
import scipy
from scipy.stats import skew, kurtosis, entropy
import antropy as ant

from scipy import signal
from scipy.signal import welch, argrelmin, argrelmax
from scipy.integrate import trapezoid, simps

from seizure_data_processing.pre_processing.features import (
    bandpass_filter,
    notch_filter,
    highpass_filter,
    rms,
    number_min,
    number_max,
    number_zero_crossings,
    line_length,
    sample_entropy,
)


def initialize_features(cols):
    """initialize_feature(cols): initializes a dictionary with (empty) dataframes
            for all the features. Names of the channels are given by cols.

    Args:
        cols (list): list of strings containing the names of the channels (columns of the dataframe)

    Returns:
        dict: dictionary of dataframe objects.
    """
    features = {
        "min": pd.DataFrame(columns=cols),
        "max": pd.DataFrame(columns=cols),
        "nzc": pd.DataFrame(columns=cols),
        "skewness": pd.DataFrame(columns=cols),
        "kurtosis": pd.DataFrame(columns=cols),
        "RMS_amplitude": pd.DataFrame(columns=cols),
        "total_power": pd.DataFrame(columns=cols),
        "peak_freq": pd.DataFrame(columns=cols),
        "mean_power_delta": pd.DataFrame(columns=cols),
        "mean_power_theta": pd.DataFrame(columns=cols),
        "mean_power_alpha": pd.DataFrame(columns=cols),
        "mean_power_beta": pd.DataFrame(columns=cols),
        "mean_power_HF": pd.DataFrame(columns=cols),
        "spectral_entropy": pd.DataFrame(columns=cols),
        "line_length": pd.DataFrame(columns=cols),
        "sample_entropy": pd.DataFrame(columns=cols),
    }

    return features


def extract_features(
    eeg,
    filter_param,
    window_time=2,
    seiz_overlap=0.9,
    bckg_overlap=0,
    epoch_remove=True,
    min_amplitude=11,
    max_amplitude=150,
):

    window_length = int(window_time * eeg.Fs)
    seiz_overlap = int(seiz_overlap * window_length)
    bckg_overlap = int(bckg_overlap * window_length)

    filtered_signals = bandpass_filter(
        eeg.data,
        eeg.Fs,
        filter_param["min_freq"],
        filter_param["max_freq"],
        axis=1,
        order=4,
    )
    filtered_signals = notch_filter(
        filtered_signals, eeg.Fs, filter_param["notch_freq"], axis=1
    ).T

    orig_signals = highpass_filter(
        eeg.data, eeg.Fs, filter_param["min_freq"], axis=1, order=4
    ).T

    time = eeg.get_time()
    labels = eeg.get_labels()
    features = initialize_features(eeg.channels)

    i_start = 0
    i_end = window_length
    last = False
    i_feat = 0
    annotations = []
    feat_start_time = []
    feat_stop_time = []
    while not last:
        # ----------- label and time ------------
        window_label = 2 * (np.sum(labels[i_start : i_end + 1]) > 0) - 1
        # ----------- epoch ------------
        filtered_epoch = np.asarray(
            filtered_signals[i_start : i_end + 1, :], dtype=float
        )
        orig_epoch = np.asarray(orig_signals[i_start : i_end + 1, :], dtype=float)
        # ---------- remove bad epochs ------------
        # Check minimum and maximum RMS amplitude, remove bad epochs
        if epoch_remove:
            rms_window = rms(filtered_epoch, axis=None)
            if np.any(rms_window < min_amplitude) or np.any(rms_window > max_amplitude):
                if window_label == 1:
                    i_start, i_end = update_index(
                        i_start, i_end, seiz_overlap, window_length
                    )
                else:
                    i_start, i_end = update_index(
                        i_start, i_end, bckg_overlap, window_length
                    )
                if i_end + 1 > len(time):
                    last = True
                continue

        # ------------------ Feature calculation ----------------------------
        # Time domain features
        features["min"].loc[i_feat] = number_min(filtered_epoch)
        features["max"].loc[i_feat] = number_max(filtered_epoch)
        features["nzc"].loc[i_feat] = number_zero_crossings(filtered_epoch)
        features["skewness"].loc[i_feat] = skew(filtered_epoch, axis=0)
        features["kurtosis"].loc[i_feat] = kurtosis(
            filtered_epoch, axis=0, nan_policy="raise"
        )
        features["RMS_amplitude"].loc[i_feat] = rms(filtered_epoch, axis=0)
        features["line_length"].loc[i_feat] = line_length(filtered_epoch, axis=0)

        # Frequency domain features
        # power spectral density
        freq, psd = welch(
            orig_epoch,
            fs=eeg.Fs,
            nperseg=(1 / window_time) * window_length,
            axis=0,
            scaling="density",
        )
        # total power
        features["total_power"].loc[i_feat] = np.sum(psd, axis=0)
        # peak frequency
        features["peak_freq"].loc[i_feat] = freq[np.argmax(psd, axis=0)]
        # mean power in high frequency band
        features["mean_power_HF"].loc[i_feat] = np.mean(
            psd[(freq >= 40) & (freq < 80), :], axis=0
        )
        # psd filtered in frequency bands
        freq, psd = welch(
            filtered_epoch,
            fs=eeg.Fs,
            nperseg=(1 / window_time) * window_length,
            axis=0,
            scaling="density",
        )
        # mean power in frequency bands
        features["mean_power_delta"].loc[i_feat] = np.mean(
            psd[(freq >= 0.5) & (freq < 4), :], axis=0
        )
        features["mean_power_theta"].loc[i_feat] = np.mean(
            psd[(freq >= 4) & (freq < 8), :], axis=0
        )
        features["mean_power_alpha"].loc[i_feat] = np.mean(
            psd[(freq >= 8) & (freq < 13), :], axis=0
        )
        features["mean_power_beta"].loc[i_feat] = np.mean(
            psd[(freq >= 13) & (freq < 20), :], axis=0
        )
        # ------------------- Entropy Features -------------------------------
        # entropy of the power spectral density
        features["spectral_entropy"].loc[i_feat] = ant.spectral_entropy(
            filtered_epoch,
            eeg.Fs,
            method="welch",
            nperseg=(1 / window_time) * window_length,
            normalize=True,
            axis=0,
        )
        # sample entropy
        features["sample_entropy"].loc[i_feat] = sample_entropy(filtered_epoch, axis=0)
        # # shannon entropy
        # psd_norm = psd / psd.sum(axis=0, keepdims=True)
        #

        # ------------------ Annotations ----------------------------
        annotations.append(window_label)
        feat_start_time.append(time[i_start])
        feat_stop_time.append(time[i_end])

        # -------------------- update index -------------------------------
        if window_label == 1:
            i_start, i_end = update_index(i_start, i_end, seiz_overlap, window_length)
        else:
            i_start, i_end = update_index(i_start, i_end, bckg_overlap, window_length)
        if i_end + 1 > len(time):
            last = True
        i_feat += 1

    # convert to numpy array
    annotations = np.array(annotations)
    feat_start_time = np.array(feat_start_time)
    feat_stop_time = np.array(feat_stop_time)
    assert len(annotations) == len(
        features["min"]
    ), "Length of annotations should be equal to number of features."

    # Combine everything into 1 dataframe
    df = pd.concat(features.values(), axis=1, keys=features.keys())
    df.columns = ["|".join([str(val) for val in col]) for col in df.columns.values]
    add_columns = {
        "epoch": df.index,
        "annotation": annotations,
        "start_time": feat_start_time,
        "stop_time": feat_stop_time,
    }
    df = pd.concat((pd.DataFrame(add_columns, index=df.index), df), axis=1)
    df["filename"] = eeg.filename

    return df


def update_index(i_start, i_end, overlap, window_size):
    i_start = i_start + window_size - overlap
    i_end = i_end + window_size - overlap
    return i_start, i_end
