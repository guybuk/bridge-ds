[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/guybuk/bridge-ds/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/bridge-ds/badge/?version=latest)](https://bridge-ds.readthedocs.io/en/latest/?badge=latest)


```
    __         _     __                     __    
   / /_  _____(_)___/ /___ ____        ____/ /____
  / __ \/ ___/ / __  / __ `/ _ \______/ __  / ___/
 / /_/ / /  / / /_/ / /_/ /  __/_____/ /_/ (__  ) 
/_.___/_/  /_/\__,_/\__, /\___/      \__,_/____/  
                   /____/                         
```

_bridge-ds_ is a lightweight Python framework designed to provide a
unified interface to deep learning datasets from different
modalities: Perform global operations, aggregations and queries
with a Pandas-like
experience, and handle individual samples and raw data using a
class-based, tab-completion-ey interface.



# Contents

<!-- TOC -->
* [Contents](#contents)
* [Key Features](#key-features)
* [Installation](#installation)
* [Documentation](#documentation)
<!-- TOC -->

# Key Features

**Browse**

Browse through your datasets with ease using an intuitive interface.

![Browse Datasets](docs/gifs/browse.gif)

**Work with tables**

View your data as tables.

![Table Interface](docs/gifs/tables.gif)

**Plot your data**

Visualize your data quickly and effectively with the exposed Pandas Plotting API.

![Plotting](docs/gifs/plot.gif)

**Assign, sort and filter**

Perform common data operations like assigning new columns, sorting, and filtering with Pandas-like syntax.

![Table Operations](docs/gifs/do_stuff.gif)

**Augment**

Apply and visualize data augmentations directly within your workflow.

![Transforms](docs/gifs/transform.gif)

# Installation

You can install the latest version of Bridge's from PyPI. It comes in a few flavors:

*Core*: The core package includes the basic functionality of Bridge.

```console
$ pip install bridge-ds
```
*Vision*: The vision package includes the core package and additional (opinionated) functionality for working with image datasets.

```console
$ pip install bridge-ds[vision]
```

* _NOTE_: to run the demo notebooks locally, you'll need the `vision` package.

# Documentation

To learn more about bridge-ds, please visit the [official documentation](https://bridge-ds.readthedocs.io/).

# Development

## Setup
```console
$ git clone https://github.com/guybuk/bridge-ds.git
$ cd bridge-ds
$ pip install -e ".[dev]"

# Testing
$ pytest tests/core

# Building the docs
$ sudo apt install pandoc
$ cd docs
$ make html
```

## Roadmap

bridge-ds is under active development, currently in a pre-alpha stage.

The following is a rough roadmap of the planned features:

- Video Support
    - [ ] DataIO for video
    - [ ] DisplayEngine (video player)
    - [ ] DatasetProviders (for popular video datasets)
    - [ ] Transforms (clipping, sampling, augmentation)
- Text
    - [ ] DatasetProviders
    - [ ] DisplayEngine (adapt existing engine to work with classic text tasks: translation, Q&A, etc.)
- Core
    - [ ] DualDatasets (for tasks with two main elements e.g. image-image, image-text,text-text)
    - [ ] Stress testing (currently have no capacity to test huge datasets)