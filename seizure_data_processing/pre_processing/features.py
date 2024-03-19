"""
    Functions for feature extraction.
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from pywt import wavedec
from sklearn.preprocessing import StandardScaler
from functools import partial

# import tusz_data_processing.data_sampling as ds
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
import antropy as ant
from scipy.integrate import trapezoid, simps


def chunker(arr, size, overlap):
    """chunker (with overlap) for numpy array"""
    for pos in range(0, len(arr), size - overlap):
        yield arr[pos : pos + size + 1]  # TODO: check if correct!!!


def df_chunker(seq, size, overlap):
    """chunker with overlap for dataframe"""
    for pos in range(0, len(seq), size - overlap):
        yield seq.iloc[pos : pos + size]


def notch_filter(x, fsamp, freq, axis=-1):
    b, a = signal.iirnotch(freq, 30, fsamp)
    return signal.lfilter(b, a, x, axis=axis)


def bandpass_filter(x, fsamp, min_freq, max_freq, axis=-1, order=4):
    """filters the given signal x using a Butterworth bandpass filter

    Parameters
    ----------
    x : ndarray
        signal to filter
    fsamp : float
        sampling frequency
    min_freq : float
        cut-off frequency lower bound
    max_freq : float
        cut-off frequency upper bound
    axis : int, optional
        axis to filter, by default -1  (0 for filter along the row, 1 for along the columns)
    order : int, optional
        order of Butterworth filter, by default 4

    Returns
    -------
    ndarray
        filtered_signal
    """
    sos = signal.butter(
        order, [min_freq, max_freq], btype="bandpass", output='sos', fs=fsamp, analog=False
    )

    return signal.sosfiltfilt(sos, x, axis=axis)


def highpass_filter(x, fsamp, min_freq, axis=-1, order=4):
    """filters the given signal x using a Butterworth bandpass filter

    Parameters
    ----------
    x : ndarray
        signal to filter
    fsamp : float
        sampling frequency
    min_freq : float
        cut-off frequency lower bound
    axis : int, optional
        axis to filter, by default -1  (0 for filter along the row, 1 for along the columns)
    order : int, optional
        order of Butterworth filter, by default 4

    Returns
    -------
    ndarray
        filtered_signal
    """
    sos = signal.butter(
        order, min_freq, btype="highpass", fs=fsamp, output="sos", analog=False
    )

    return signal.sosfiltfilt(sos, x, axis=axis)


def lowpass_filter(x, fsamp, max_freq, axis=-1, order=4):
    """filters the given signal x using a Butterworth bandpass filter

    Parameters
    ----------
    x : ndarray
        signal to filter
    fsamp : float
        sampling frequency
    max_freq : float
        cut-off frequency lower bound
    axis : int, optional
        axis to filter, by default -1  (0 for filter along the row, 1 for along the columns)
    order : int, optional
        order of Butterworth filter, by default 4

    Returns
    -------
    ndarray
        filtered_signal
    """
    sos = signal.butter(
        order, max_freq, btype="lowpass", fs=fsamp, output="sos", analog=False
    )

    return signal.sosfiltfilt(sos, x, axis=axis)


def number_zero_crossings(x):
    # use this to calculate the number of zero crossings per column
    # of an I x J array or a I dimensional list
    if np.ndim(x) == 1:
        x = np.c_[x]
    return np.sum(x[:-1, :] * x[1:, :] < 0, axis=0) + np.sum(x == 0, axis=0)


def df_zero_crossings(df, col=None):
    """
    number of zero crossings of the columns of a dataframe object
    :param df:
    :param col:
    :return: numpy array
    """
    if not col:
        nzc = number_zero_crossings(df.to_numpy())
    else:
        nzc = number_zero_crossings(df.to_numpy())
    return nzc


def number_min(x):
    # use this to calculate the number of minima per column
    # of an I x J array or a I dimensional list
    if x.ndim == 1:
        return len(signal.argrelmin(x, axis=0)[0])
    else:
        num_mins = np.zeros(x.shape[1])
    for i in range(0, x.shape[1]):
        num_mins[i] = len(signal.argrelmin(x[:, i], axis=0)[0])
    return num_mins


def number_max(x):
    # use this to calculate the number of maxima per column
    # of an I x J array or a I dimensional list
    if x.ndim == 1:
        return len(signal.argrelmax(x, axis=0)[0])
    else:
        num_max = np.zeros(x.shape[1])
    for i in range(0, x.shape[1]):
        num_max[i] = len(signal.argrelmax(x[:, i], axis=0)[0])
    return num_max


def rms(x, axis=None):
    """
    Calculate the root mean square of the given array x
    Args:
        x (ndarray): array to calculate the rms of
        axis (int): axis to calculate the rms along

    Returns:
        ndarray: rms of x
    """
    if axis is None:  # axis=None
        return np.sqrt(np.sum(x * x, axis=None) / x.size)
    elif axis == 0 or axis == 1:
        return np.sqrt(np.sum(x * x, axis=axis) / x.shape[axis])
    else:
        raise Exception("rms(x) not defined for axis = %d", axis)


def mean_power(f, Pxx_den, min_freq, max_freq):
    ind = np.where((f >= min_freq) & (f <= max_freq))[0]
    return simps(Pxx_den[ind, :], dx=f[1] - f[0], axis=0)


def dwt_transform(data, wavelet, level=4, axis=0):
    """
    Discrete Time Wavelet transform of the bandpass-filtered data (0. - 50 Hz)
    Args:
        data (ndarray):
        wavelet:
        level:
        axis:

    Returns:

    """
    coef = wavedec(data, wavelet, mode="symmetric", level=level, axis=axis)

    return coef


def dwt_relative_power(
    data, epoch_size, overlap, l=0.99923, N=120, wavelet="db4", level=4, axis=0
):
    """Calculate the relative power feature based on the DWT.

    Args:
        data (ndarray): EEG data (1 channel only)
        epoch_size (int): length of epoch
        overlap (int): length of overlap epochs
        l (float, optional): lambda, forgetting factor. Defaults to 0.99923.
        N (int, optional): Memory index. Defaults to 120.
        wavelet (str, optional): Mother wavelet. Defaults to 'db4'.
        level (int, optional): Number of wavelet transform levels. Defaults to 4.
        axis (int, optional): axis on which to perform DWT. Defaults to 0.

    Raises:
        Exception: If number of channels > 1

    Returns:
        ndarray: N_epochs x N_coef array
    """

    n_data, n_chan = data.shape
    if n_chan > 1:
        raise Exception("relative_power not yet implemented for more than 1 channel")

    n_feature = int(n_data / (epoch_size - overlap))

    rp = np.zeros((n_feature, level + 1))
    FG = np.zeros((n_feature, level + 1))
    BG = np.zeros((1, level + 1))  # initialize background power
    for i, chunk in enumerate(chunker(data, epoch_size, overlap)):
        coef = wavedec(chunk, wavelet=wavelet, mode="symmetric", level=level, axis=axis)
        FG[i, :] = [np.median(c**2) for c in coef]  # median(D_i^2)

        if i > N:
            BG = (l - 1) * np.median(
                FG[i - N : i, :], axis=0
            ) + l * BG  # med(FG(e-1)..FG(e-N))
        else:
            BG = (l - 1) * np.median(
                FG[0:i, :], axis=0
            ) + l * BG  # med(FG(e-1)..FG(e-N))

        rp[i, :] = FG[i, :] / BG

    return rp


def line_length(x, axis=0):
    """Calculate the line length feature.

    Args:
        x (ndarray): Data epoch (N, N_chan)
        axis (int, optional): axis along which to calculate the feature. Defaults to 0.

    Returns:
        ndarray: (N_chan,) array with the line length(s)
    """
    diff = np.abs(x[0:-2, :] - x[1:-1, :])

    return np.sum(diff, axis=axis)


# def shannon_entropy(x, axis=0):
#     """Calculate the Shannon entropy of the data. Using scipy.stats.entropy.
#
#     Args:
#         x (ndarray): Data epoch (N, N_chan)
#         axis (int, optional): axis along which to calculate the feature. Defaults to 0.
#
#     Returns:
#         ndarray: (N_chan,) array with the Shannon entropy
#     """
#


def sample_entropy(x, axis=0):
    """Calculate the sample entropy of the data.

    Args:
        x (ndarray): Data epoch (N, N_chan)
        axis (int, optional): axis along which to calculate the feature. Defaults to 0.

    Returns:
        ndarray: (N_chan,) array with the sample entropy
    """

    return np.apply_along_axis(
        ant.sample_entropy, axis=axis, arr=x, order=2, metric="chebyshev"
    )


def normalize_feature(feature, method="standard", epoch_time=2, buffer=120, labda=0.92):
    # input: np array (cols: features, rows:epochs), epoch length (s), buffer (s)
    # median decaying memory method or standard scaler

    if method == "median-decay":
        z = np.zeros((1, feature.shape[1]))
        memory_epochs = int(buffer / epoch_time)  # num epochs for buffer (s)

        norm_features = np.zeros(feature.shape)
        norm_features[0, :] = feature[0, :]
        for i in range(1, feature.shape[0]):
            old_z = z
            trans = i > memory_epochs  # past max transient duration (buffer)
            index_memory = 0 + (i - memory_epochs + 1) * trans
            z = (1 - labda) * np.median(
                feature[index_memory:i, :], axis=0
            ) + labda * old_z
            norm_features[i, :] = feature[i, :] / z

        scaler = []

    elif method == "standard":
        scaler = StandardScaler()
        scaler.fit(feature)
        norm_features = scaler.transform(feature)

    else:
        raise Exception("this normalization method is not valid.")

    return norm_features, scaler


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
    """
    Extract features from the EEG data.
    Args:
        eeg: EEG object
        filter_param: filter parameters, dictionary with keys: min_freq, max_freq, notch_freq
        window_time: duration of the windows in seconds
        seiz_overlap: overlap of the seizure windows
        bckg_overlap: overlap of the background windows
        epoch_remove: remove bad epochs based on amplitude
        min_amplitude: if epoch_remove, minimum amplitude to keep
        max_amplitude: if epoch_remove, maximum amplitude to keep

    Returns:
        dataframe: dataframe with the features, annotations, start/stop time and filename
    """

    window_length = int(window_time * eeg.Fs)
    seiz_overlap = int(seiz_overlap * window_length)
    bckg_overlap = int(bckg_overlap * window_length)

    orig_signals = highpass_filter(
        eeg.data, eeg.Fs, filter_param["min_freq"], axis=1, order=4
    ).T

    filtered_signals = lowpass_filter(
        orig_signals,
        eeg.Fs,
        filter_param["max_freq"],
        axis=0,
        order=4,
    )

    if filter_param["notch_freq"] is not None:
        filtered_signals = notch_filter(
            filtered_signals, eeg.Fs, filter_param["notch_freq"], axis=0
        )

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
        window_label = 2 * (np.sum(labels[i_start : i_end + 1]) > 0) - 1 # positive if more than 50% of the window is labeled as seizure
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
                    break
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
        features["total_power"].loc[i_feat] = simps(psd, dx=freq[1] - freq[0], axis=0)
        # peak frequency
        features["peak_freq"].loc[i_feat] = freq[np.argmax(psd, axis=0)]
        # mean power in high frequency band
        features["mean_power_HF"].loc[i_feat] = mean_power(freq, psd, 40, 80)
        # psd filtered in frequency bands
        freq, psd = welch(
            filtered_epoch,
            fs=eeg.Fs,
            nperseg=(1 / window_time) * window_length,
            axis=0,
            scaling="density",
        )
        # mean power in frequency bands
        features["mean_power_delta"].loc[i_feat] = mean_power(freq, psd, 0.5, 3)
        features["mean_power_theta"].loc[i_feat] = mean_power(freq, psd, 4, 8)
        features["mean_power_alpha"].loc[i_feat] = mean_power(freq, psd, 9, 13)
        features["mean_power_beta"].loc[i_feat] = mean_power(freq, psd, 14, 20)
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
