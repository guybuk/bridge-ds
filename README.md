[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/guybuk/bridge-ds/python-package.yml)

[TITLE IMAGE PLACEHOLDER]

_Bridge_ is a lightweight Python framework designed to provide a
unified interface to deep learning datasets from different
modalities: Perform global operations, aggregations and queries
with a Pandas-like
experience, and handle individual samples and raw data using a
class-based, tab-completion-ey interface.

**Key Features**

* **Dataset as a table:**
    * Give your deep learning dataset a DataFrame
      interface; making
      cumbersome operations such as selections, sorting and
      aggregations - easy.
* **Explore data in your notebook:**
    * Browse your data directly in your notebook, without
      intermediary web-apps.
* **Agnostic to Deep Learning Engines:**
    * Convert into a training-ready dataset
      in your DL framework of choice.
* **Transform and Debug:**
    * Maintain full visibility into your
      preprocessing/augmentation pipeline. See exactly which
      inputs enter your model.
* **Work with Arbitrary Sources**:
    * Work with remote and local data together, seamlessly.
* **Keep your data to yourself**:
    * No need to upload your data to third parties.

# Contents

<!-- TOC -->
* [Contents](#contents)
* [Installation](#installation)
  * [Install From Source](#install-from-source)
  * [Documentation](#documentation)
<!-- TOC -->

# Installation

```
pip install bridge-ds
```

## Documentation

For high-level demos to show off Bridge's capabilities, consider
browsing the following notebooks:
1. [Quick and easy data exploration](docs/source/notebooks/vision/fundamentals/coco_eda_demo.ipynb)
2. [From sources, through augmentations, to Pytorch](docs/source/notebooks/vision/processing_data/source2tensors_demo.ipynb)

For a deeper understanding of Bridge, and to connect your custom datasets and data types,
 proceed to our [notebooks](notebooks) folder.
