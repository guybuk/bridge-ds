Getting Started
===============
Installation
------------

You can install the latest version of Bridge's from PyPI. It comes in a few flavors:

*Core*: The core package includes the basic functionality of Bridge.

.. code-block:: console

    $ pip install bridge-ds

*Vision*: The vision package includes the core package and additional (opinionated) functionality for working with image datasets.

.. code-block:: console

    $ pip install bridge-ds[vision]


Key Concepts
------------

In this section you will learn the basics of Bridge. Start by
reading about the key concepts, and then proceed to
the guides below.

Element
^^^^^^^

An Element is the basic unit of data in a Dataset, from raw data
objects such as images, text, audio, to various annotations such
as class labels, bounding boxes, and segmentation maps. In
essence,
anything that constitutes a piece of information within the
dataset can be an Element.

Sample
^^^^^^

A Sample is a collection of Elements. It is our representation of
a typical item-within-a-dataset.
For example, an image with
object detections constitutes a Sample, comprising a single image
Element and multiple bounding box Elements.

Dataset
^^^^^^^

A Dataset is a collection of Samples. It exposes the Table and
Sample APIs.

Table API
^^^^^^^^^

A general term for the set of functions and operators exposed by
the
Dataset which allows users to perform
high-level operations with a user experience similar to Pandas -
assign, query, sort, map, etc. In short, an API that
allows users to treat any dataset as a DataFrame, where **every
row is an element.**

Sample API
^^^^^^^^^^

A general term for the set of functions and operators exposed by
the Dataset which allows users to work on
individual examples in the dataset in a meaningful manner.
If the Table API is meant for high-level
dataset management, then the Sample API is used for low-level
operations
like loading,
caching, and transforming raw data (e.g. pixels, strings).
