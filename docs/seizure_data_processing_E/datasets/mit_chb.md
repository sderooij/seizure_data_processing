# Mit Chb

[Seizure_data_processing Index](../../README.md#seizure_data_processing-index) /
[Seizure Data Processing](../index.md#seizure-data-processing) /
[Datasets](./index.md#datasets) /
Mit Chb

> Auto-generated documentation for [seizure_data_processing.datasets.mit_chb](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/datasets/mit_chb.py) module.

- [Mit Chb](#mit-chb)
  - [load_annotations](#load_annotations)
  - [parse_annotations](#parse_annotations)

## load_annotations

[Show source in mit_chb.py:83](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/datasets/mit_chb.py#L83)

load annotations for the given edf file.

#### Arguments

- `file` *str* - edf file to be annotated

#### Returns

- `DataFrame` - (start_time, stop_time, seizure_type, probability)

#### Signature

```python
def load_annotations(file: str):
    ...
```



## parse_annotations

[Show source in mit_chb.py:15](https://github.com/sderooij/seizure_data_processing/blob/main/seizure_data_processing/datasets/mit_chb.py#L15)

parse the seizure annotation from the summary file for the given edf_file.

#### Arguments

- `summary_file` *str* - absolute path to the summary file
- `edf_file` *str* - absolute path to the edf file
- `dataframe` *bool, optional* - Output as a dataframe. Defaults to False.

#### Raises

- `Exception` - If edf file is not found in the summary file.

#### Returns

tuple or DataFrame: of shape (start_time, stop_time, seizure_type, probability)

#### Signature

```python
def parse_annotations(summary_file, edf_file, dataframe=False):
    ...
```