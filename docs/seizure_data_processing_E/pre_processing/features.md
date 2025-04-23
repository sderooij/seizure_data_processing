# Features

[Seizure_data_processing Index](../../README.md#seizure_data_processing-index) /
[Seizure Data Processing](../index.md#seizure-data-processing) /
[Pre Processing](./index.md#pre-processing) /
Features

> Auto-generated documentation for [seizure_data_processing.pre_processing.features](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py) module.

- [Features](#features)
  - [bandpass_filter](#bandpass_filter)
  - [chunker](#chunker)
  - [df_chunker](#df_chunker)
  - [df_zero_crossings](#df_zero_crossings)
  - [dwt_relative_power](#dwt_relative_power)
  - [dwt_transform](#dwt_transform)
  - [highpass_filter](#highpass_filter)
  - [line_length](#line_length)
  - [mean_power](#mean_power)
  - [normalize_feature](#normalize_feature)
  - [number_max](#number_max)
  - [number_min](#number_min)
  - [number_zero_crossings](#number_zero_crossings)
  - [rms](#rms)

## bandpass_filter

[Show source in features.py:29](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L29)

filters the given signal x using a Butterworth bandpass filter

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

#### Signature

```python
def bandpass_filter(x, fsamp, min_freq, max_freq, axis=-1, order=4):
    ...
```



## chunker

[Show source in features.py:17](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L17)

chunker (with overlap) for numpy array

#### Signature

```python
def chunker(arr, size, overlap):
    ...
```



## df_chunker

[Show source in features.py:23](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L23)

chunker with overlap for dataframe

#### Signature

```python
def df_chunker(seq, size, overlap):
    ...
```



## df_zero_crossings

[Show source in features.py:100](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L100)

number of zero crossings of the columns of a dataframe object

#### Arguments

- `df`
- `col`

#### Returns

numpy array

#### Signature

```python
def df_zero_crossings(df, col=None):
    ...
```



## dwt_relative_power

[Show source in features.py:166](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L166)

Calculate the relative power feature based on the DWT.

#### Arguments

- `data` *ndarray* - EEG data (1 channel only)
- `epoch_size` *int* - length of epoch
- `overlap` *int* - length of overlap epochs
- `l` *float, optional* - lambda, forgetting factor. Defaults to 0.99923.
- `N` *int, optional* - Memory index. Defaults to 120.
- `wavelet` *str, optional* - Mother wavelet. Defaults to 'db4'.
- `level` *int, optional* - Number of wavelet transform levels. Defaults to 4.
- `axis` *int, optional* - axis on which to perform DWT. Defaults to 0.

#### Raises

- `Exception` - If number of channels > 1

#### Returns

- `ndarray` - N_epochs x N_coef array

#### Signature

```python
def dwt_relative_power(
    data, epoch_size, overlap, l=0.99923, N=120, wavelet="db4", level=4, axis=0
):
    ...
```



## dwt_transform

[Show source in features.py:149](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L149)

Discrete Time Wavelet transform of the bandpass-filtered data (0. - 50 Hz)

#### Arguments

data (ndarray):
wavelet:
level:
axis:

#### Signature

```python
def dwt_transform(data, wavelet, level=4, axis=0):
    ...
```



## highpass_filter

[Show source in features.py:64](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L64)

filters the given signal x using a Butterworth bandpass filter

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

#### Signature

```python
def highpass_filter(x, fsamp, min_freq, axis=-1, order=4):
    ...
```



## line_length

[Show source in features.py:215](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L215)

Calculate the line length feature.

#### Arguments

- `x` *ndarray* - Data epoch (N, N_chan)
- `axis` *int, optional* - axis along which to calculate the feature. Defaults to 0.

#### Returns

- `ndarray` - (N_chan,) array with the line length(s)

#### Signature

```python
def line_length(x, axis=0):
    ...
```



## mean_power

[Show source in features.py:144](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L144)

#### Signature

```python
def mean_power(f, Pxx_den, min_freq, max_freq):
    ...
```



## normalize_feature

[Show source in features.py:230](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L230)

#### Signature

```python
def normalize_feature(feature, method="standard", epoch_time=2, buffer=120, labda=0.92):
    ...
```



## number_max

[Show source in features.py:123](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L123)

#### Signature

```python
def number_max(x):
    ...
```



## number_min

[Show source in features.py:114](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L114)

#### Signature

```python
def number_min(x):
    ...
```



## number_zero_crossings

[Show source in features.py:92](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L92)

#### Signature

```python
def number_zero_crossings(x):
    ...
```



## rms

[Show source in features.py:132](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/features.py#L132)

#### Signature

```python
def rms(x, axis=None):
    ...
```