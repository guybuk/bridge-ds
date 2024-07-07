# Processing Data
In the previous tutorials, we learned how to use Bridge Datasets. However, there's a clear caveat: We can't train a model directly on a Bridge Dataset.

There are a couple of reasons for this:
1. Bridge Datasets don't inherit from DL Engine dataset objects (e.g. PyTorch or Keras Datasets).
2. Models don't accept raw data as input. Vision datasets go through augmentations, text datasets go through tokenization and embedding, etc.

The notebooks in this section are: 
- The [source2tensors demo](source2tensors.ipynb), in which we show a **full e2e example** of how we take a Bridge Dataset, augment it, visualize the results, and once we're satisfied with our augmentation paramters we proceed to transform the Bridge Dataset into a PyTorch dataset with tensors.

- The [Sample Transform](sample_transform.ipynb) notebook, in which we go into detail about the mechanism which is used to transform Bridge Datasets' raw data
- The [Cache Mechanism](cache_mechanism.ipynb) notebook, in which we explain how to control where intermediate data (created during the process pipeline) is saved.