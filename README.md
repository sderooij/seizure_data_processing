# Seizure data processing
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![coverage](coverage.svg)

This repository contains (python) code that is used to process (EEG) data that is used for seizure detection. 

### Note on the use of TensorLibrary
The ``classification`` module uses the ``TensorLibrary`` package. This package is not available on PyPi, but can be found at
`github.com/sderooij/tensorlibrary`.
```bash
pip install git+https://github.com/sderooij/tensorlibrary@7729dd2230a610bd982f71de7440d829ca806751
```

