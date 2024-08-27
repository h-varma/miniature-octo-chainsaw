# Welcome to My Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/h-varma/miniature-octo-chainsaw/ci.yml?branch=main)](https://github.com/h-varma/miniature-octo-chainsaw/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/miniature-octo-chainsaw/badge/)](https://miniature-octo-chainsaw.readthedocs.io/)
[![codecov](https://codecov.io/gh/h-varma/miniature-octo-chainsaw/branch/main/graph/badge.svg)](https://codecov.io/gh/h-varma/miniature-octo-chainsaw)

## Installation

The Python package `miniature_octo_chainsaw` can be installed from PyPI:

```
python -m pip install miniature_octo_chainsaw
```

## Development installation

If you want to contribute to the development of `miniature_octo_chainsaw`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/h-varma/miniature-octo-chainsaw.git
cd miniature-octo-chainsaw
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
