# EEG

[Seizure_data_processing Index](../../README.md#seizure_data_processing-index) /
[Seizure Data Processing](../index.md#seizure-data-processing) /
[Pre Processing](./index.md#pre-processing) /
EEG

> Auto-generated documentation for [seizure_data_processing.pre_processing.eeg](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py) module.

#### Attributes

- `file` - file = (
  config.TUSZ_DIR
  + r"edf\train\01_tcp_ar\000\00000077\s003_2010_01_21\00000077_s003_t000.edf"
  )
  eeg_file = EEG(file): `MIT_CHB_DIR + 'chb01/chb01_03.edf'`


- [EEG](#eeg)
  - [EEG](#eeg-1)
    - [EEG().annotate](#eeg()annotate)
    - [EEG().apply_montage](#eeg()apply_montage)
    - [EEG().bandpass_filter](#eeg()bandpass_filter)
    - [EEG().get_labels](#eeg()get_labels)
    - [EEG().get_time](#eeg()get_time)
    - [EEG().load](#eeg()load)
    - [EEG().plot](#eeg()plot)
    - [EEG().resample](#eeg()resample)
    - [EEG().save](#eeg()save)
    - [EEG().show](#eeg()show)
  - [get_pos_edf](#get_pos_edf)

## EEG

[Show source in eeg.py:20](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L20)

#### Signature

```python
class EEG:
    def __init__(self, filename: str, channels: list[str] = None, dataset=""):
        ...
```

### EEG().annotate

[Show source in eeg.py:158](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L158)

Annotate the EEG data

#### Raises

- `Exception` - If dataset other than mit-chb or tusz is supplied

#### Returns

- [EEG](#eeg) - annotated EEG object

#### Signature

```python
def annotate(self):
    ...
```

### EEG().apply_montage

[Show source in eeg.py:177](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L177)

apply a montage to the EEG signals

#### Arguments

- `self` *EEG* - EEG object
- `montage` *list[str]* - list of montage

#### Returns

self

#### Signature

```python
def apply_montage(self, montage: list[str]):
    ...
```

### EEG().bandpass_filter

[Show source in eeg.py:198](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L198)

Apply a bandpass filter to the data. Uses a Butterworth filter.

#### Arguments

- `min_freq` *float* - cut-off frequency high-pass filter
- `max_freq` *float* - cut-off frequency low-pass filter
- `order` *int, optional* - Order of the butterworth filter. Defaults to 4.

#### Returns

- `self` - filter EEG object

#### Signature

```python
def bandpass_filter(self, min_freq, max_freq, order=4):
    ...
```

### EEG().get_labels

[Show source in eeg.py:295](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L295)

Get the labels corresponding to the annotations. If annotation is "bckg" then label is -1. If annotation is any type of seizure then label is 1.

#### Returns

- `list` - list of the labels for each time point.

#### Signature

```python
def get_labels(self):
    ...
```

### EEG().get_time

[Show source in eeg.py:283](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L283)

Get a time vector for the EEG data

#### Returns

- `ndarray` - Array with time (in s) for each datapoint starting from 0 s.

#### Signature

```python
def get_time(self):
    ...
```

### EEG().load

[Show source in eeg.py:52](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L52)

load EEG signals from an EDF file

#### Arguments

- `self` *str* - EEG object

#### Returns

EEG object

#### Signature

```python
def load(self):
    ...
```

### EEG().plot

[Show source in eeg.py:258](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L258)

Plot the EEG data.

#### Raises

- `Exception` - if channels are anything other than a list or numpy array.

#### Signature

```python
def plot(self):
    ...
```

### EEG().resample

[Show source in eeg.py:123](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L123)

Resample edf data to new sampling frequency.

#### Arguments

self
- `new_fs` *int* - new (desired) sampling frequency

#### Returns

- `ndarray` - array with resampled data

#### Signature

```python
def resample(self, new_fs):
    ...
```

### EEG().save

[Show source in eeg.py:216](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L216)

Save EEG object to a specific file type

#### Arguments

- `filename` *str* - name of file to save to.
- `saveas` *str, optional* - type of file to save to ".hdf5", ".h5" or ".mat". Defaults to ".mat".
- `eeg_file` *str, optional* - name of original EEG file. If None, the end of the filename of the EEG object is used. Defaults to None.

#### Signature

```python
def save(self, filename, saveas=".mat", eeg_file=None):
    ...
```

### EEG().show

[Show source in eeg.py:276](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L276)

same as plot

#### Signature

```python
def show(self):
    ...
```



## get_pos_edf

[Show source in eeg.py:319](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/pre_processing/eeg.py#L319)

Get position of channels in edf file.

#### Arguments

- `label_list` *list* - List of labels/channels
- `target_labels` *list* - List of target labels (labels to extract)

#### Raises

- `Exception` - Failed to find label

#### Returns

indices

#### Signature

```python
def get_pos_edf(label_list: list, target_labels: list):
    ...
```