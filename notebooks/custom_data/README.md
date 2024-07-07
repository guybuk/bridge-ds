# Custom Data

While BridgeDS supports common benchmarks like IN1K or MS-COCO, its primary focus lies in handling the complexity of real-world datasets. These datasets often combine multiple types of annotations for a single sample and may involve merging data from diverse sources. As demonstrated in basics.ipynb, Bridge allows datasets to adapt dynamically, depending on the scenario.

In this section, we you will learn how to use your own custom datasets with Bridge.

Start by learning how raw data is actually loaded into Bridge Elements in the [LoadMechanism](load_mechanism.ipynb) tutorial, then proceed to create a [DatasetProvider](dataset_provider.ipynb), and finally learn how to use your own visualization tools in the [DisplayEngine](display_engine.ipynb) demo.