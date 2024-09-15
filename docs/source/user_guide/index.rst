User Guide
==========
Demos
-----

For high-level demos that showcase Bridge's capabilities, consider
browsing the following notebooks:

Exploratory Data Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`COCO Dataset EDA <notebooks/vision/fundamentals/coco_eda_demo>`

Full Pipeline
^^^^^^^^^^^^^
* :doc:`COCO Dataset → Augmentations → Pytorch <notebooks/vision/processing_data/source2tensors_demo>`

Guides
------

While the demos provide a glimpse into Bridge's capabilities, the following guides will introduce you to the core concepts and
design of Bridge, which will allow you to tailor it to your own needs.

Fundamentals
^^^^^^^^^^^^

Start here to learn the basics of Bridge, namely how a Bridge Dataset is designed.

* :doc:`The Sample API <notebooks/vision/fundamentals/sample_api>`
* :doc:`The Table API <notebooks/vision/fundamentals/table_api>`

Custom datasets
^^^^^^^^^^^^^^^

Learn how to create custom Bridge Datasets.

* :doc:`Connect raw data to Bridge using Load Mechanisms <notebooks/vision/custom_data/load_mechanism>`
* :doc:`Create a custom dataset with Dataset Providers <notebooks/vision/custom_data/dataset_provider>`
* :doc:`Interact with your data using Display Engines <notebooks/vision/custom_data/display_engine>`

Processing data
^^^^^^^^^^^^^^^

* :doc:`Augment data using Sample Transforms <notebooks/vision/processing_data/sample_transform>`
* :doc:`Save intermediate data with Cache Mechanisms <notebooks/vision/processing_data/cache_mechanism>`


.. toctree::
   :maxdepth: 2
   :hidden:
   
   notebooks/vision/fundamentals/coco_eda_demo
   notebooks/vision/processing_data/source2tensors_demo
   notebooks/vision/fundamentals/sample_api
   notebooks/vision/fundamentals/table_api
   notebooks/vision/custom_data/load_mechanism
   notebooks/vision/custom_data/dataset_provider
   notebooks/vision/custom_data/display_engine
   notebooks/vision/processing_data/sample_transform
   notebooks/vision/processing_data/cache_mechanism