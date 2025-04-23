# Tusz

[Seizure_data_processing Index](../../README.md#seizure_data_processing-index) /
[Seizure Data Processing](../index.md#seizure-data-processing) /
[Datasets](./index.md#datasets) /
Tusz

> Auto-generated documentation for [seizure_data_processing.datasets.tusz](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/datasets/tusz.py) module.

- [Tusz](#tusz)
  - [get_duration_tse](#get_duration_tse)
  - [load_annotations](#load_annotations)
  - [load_tse](#load_tse)

## get_duration_tse

[Show source in tusz.py:74](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/datasets/tusz.py#L74)

function: loadTSE Load seizure events from a TSE file.

#### Arguments

- `tse` - TSE event file

return:
  stop_time (s), float

#### Signature

```python
def get_duration_tse(tse_file: str):
    ...
```



## load_annotations

[Show source in tusz.py:103](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/datasets/tusz.py#L103)

load annotations and output as a pandas dataframe.

#### Arguments

- `file` *str* - edf file to annotate

#### Returns

- `pd.DataFrame` - with columns [start_time, stop_time, seizure_type, probability]

#### Signature

```python
def load_annotations(file: str) -> pd.DataFrame:
    ...
```



## load_tse

[Show source in tusz.py:12](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/datasets/tusz.py#L12)

function: loadTSE Load seizure events from a TSE file.

#### Arguments

- `tse_file` - TSE event file
- `dataframe` *bool* - output as dataframe. Default to False.

return:
  - `seizures` - output list of seizures. Each event is tuple of 4 items:
             (seizure_start [s], seizure_end [s], seizure_type, probability)

#### Signature

```python
def load_tse(tse_file: str, dataframe: bool = False):
    ...
```