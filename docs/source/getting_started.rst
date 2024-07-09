Getting started
===============



Installation
------------

You can install the latest version of Bridge's from PyPI. It comes in a few flavors:

*Core*: The core package includes the basic functionality of Bridge.

.. code-block:: console

    $ pip install bridge-ds

*Vision*: The vision package includes the core package and additional functionality for working with image datasets.

.. code-block:: console

    $ pip install bridge-ds[vision]

*Dev*: The dev package includes the core package and additional tools for development.

.. code-block:: console

    $ pip install bridge-ds[dev]


Demos
-----

For high-level demos to show off Bridge's capabilities, consider
browsing the following notebooks:

#. :doc:`Quick and easy data exploration <notebooks/vision/fundamentals/coco_eda_demo>`
#. :doc:`From sources, through augmentations, to Pytorch <notebooks/vision/processing_data/source2tensors_demo>`

For a deeper understanding of Bridge, and to connect your custom datasets and data types,
 proceed to the :doc:`user_guide` section.