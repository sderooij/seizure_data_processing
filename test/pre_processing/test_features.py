import numpy as np
from seizure_data_processing.pre_processing import features as ff

# def test_butter_bandpass_filter():
#     # Test the butter_bandpass_filter function with a known input and output
#     fs = 1000
#     t = np.arange(0, 1, 1/fs)
#     f1 = 10
#     f2 = 100
#     y = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)
#     y_filtered = ff.bandpass_filter(y, fs, 20, 120, order=6)
#     expected_output = np.sin(2*np.pi*f1*t)
#     assert np.allclose(y_filtered, expected_output, atol=1e-3)


def test_number_min():
    """
    test number of minima in time series
    """
    x = np.array([-1, -2, -1.5, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0])
    assert ff.number_min(x) == 3


def test_number_max():
    x = np.array([-1, -2, -1.5, -1, 0, 1, 2, -1, 0, 1, 2, -1])
    assert ff.number_max(x) == 2


def test_number_zero_crossing():
    x = np.array([-1, -2, -1.5, -1, 0, 1, 2, -1, 0, 1, 2, -1])
    assert ff.number_zero_crossings(x) == 4


def test_rms():
    t = np.arange(0, 10, 1 / 1000)
    x = 2 * np.sin(2 * np.pi * 10 * t)
    assert np.isclose(ff.rms(x), 2 / np.sqrt(2), atol=1e-3)
