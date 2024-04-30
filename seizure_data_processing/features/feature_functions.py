"""
    Functions for feature extraction.
"""
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


def number_zero_crossings(x):
    # use this to calculate the number of zero crossings per column
    # of an I x J array or a I dimensional list
    if x.ndim == 1:
        return np.sum(x[:-1] * x[1:] < 0) + np.sum(x == 0)
    return np.sum(x[:-1, :] * x[1:, :] < 0, axis=0) + np.sum(x == 0, axis=0)


def number_min(x):
    # use this to calculate the number of minima per column
    # of an I x J array or a I dimensional list
    if x.ndim == 1:
        return len(argrelmin(x, axis=0)[0])
    else:
        num_mins = np.zeros(x.shape[1])
    for i in range(0, x.shape[1]):
        num_mins[i] = len(argrelmin(x[:, i], axis=0)[0])
    return num_mins


def number_max(x):
    # use this to calculate the number of maxima per column
    # of an I x J array or a I dimensional list
    if x.ndim == 1:
        return len(argrelmax(x, axis=0)[0])
    else:
        num_max = np.zeros(x.shape[1])
    for i in range(0, x.shape[1]):
        num_max[i] = len(scipy.signal.argrelmax(x[:, i], axis=0)[0])
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


# @vectorize([float64])
# def sample_entropy(x, axis=0):
#     """Calculate the sample entropy of the data.
#
#     Args:
#         x (ndarray): Data epoch (N, N_chan)
#         axis (int, optional): axis along which to calculate the feature. Defaults to 0.
#
#     Returns:
#         ndarray: (N_chan,) array with the sample entropy
#     """
#
#     fun = vectorize(ant.sample_entropy, signature="(n)->()")
#
#     return ant.sample_entropy(axis=axis, arr=x, order=2, metric="chebyshev")


def mean_power(f, Pxx_den, min_freq, max_freq):
    idx_band = np.logical_and(f >= min_freq, f <= max_freq)
    return simps(Pxx_den[idx_band], dx=f[1] - f[0], axis=0)


# def bandpower(data, sf, band, window_sec=None, relative=False):
#     """Compute the average power of the signal x in a specific frequency band.
#
#     Parameters
#     ----------
#     data : 1d-array
#         Input signal in the time-domain.
#     sf : float
#         Sampling frequency of the data.
#     band : list
#         Lower and upper frequencies of the band of interest.
#     window_sec : float
#         Length of each window in seconds.
#         If None, window_sec = (1 / min(band)) * 2
#     relative : boolean
#         If True, return the relative power (= divided by the total power of the signal).
#         If False (default), return the absolute power.
#
#     Return
#     ------
#     bp : float
#         Absolute or relative band power.
#     """
#     from scipy.signal import welch
#     from scipy.integrate import simps
#     band = np.asarray(band)
#     low, high = band
#
#     # Define window length
#     if window_sec is not None:
#         nperseg = window_sec * sf
#     else:
#         nperseg = (2 / low) * sf
#
#     # Compute the modified periodogram (Welch)
#     freqs, psd = welch(data, sf, nperseg=nperseg)
#
#     # Frequency resolution
#     freq_res = freqs[1] - freqs[0]
#
#     # Find closest indices of band in frequency vector
#     idx_band = np.logical_and(freqs >= low, freqs <= high)
#
#     # Integral approximation of the spectrum using Simpson's rule.
#     bp = simps(psd[idx_band], dx=freq_res)
#
#     if relative:
#         bp /= simps(psd, dx=freq_res)
#     return bp


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
