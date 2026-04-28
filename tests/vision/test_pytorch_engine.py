"""Unit tests for the PyTorch DataLoader bridge."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bridge.engines.pytorch import PytorchEngineDataset


def test_pytorch_dataset_len_matches_source(synthetic_classification_dataset):
    pyt = PytorchEngineDataset(synthetic_classification_dataset)
    assert len(pyt) == len(synthetic_classification_dataset)


def test_pytorch_dataset_getitem_returns_dict_keyed_by_etype(synthetic_classification_dataset):
    pyt = PytorchEngineDataset(synthetic_classification_dataset)
    item = pyt[0]
    assert isinstance(item, dict)
    assert "image" in item
    assert "class_label" in item


def test_pytorch_dataset_image_data_is_array(synthetic_classification_dataset):
    pyt = PytorchEngineDataset(synthetic_classification_dataset)
    item = pyt[0]
    assert isinstance(item["image"], list)
    assert len(item["image"]) == 1
    img = item["image"][0]
    assert isinstance(img, np.ndarray)
    assert img.shape == (64, 64, 3)


def test_pytorch_dataset_iteration(synthetic_classification_dataset):
    pyt = PytorchEngineDataset(synthetic_classification_dataset)
    items = [pyt[i] for i in range(len(pyt))]
    assert len(items) == 2
    assert all("image" in it and "class_label" in it for it in items)


def test_pytorch_dataset_with_torch_dataloader(synthetic_classification_dataset):
    """End-to-end: wraps in torch.utils.data.DataLoader without crashing.

    We use batch_size=1 and a no-op collate to avoid stacking issues from
    the dict-of-lists shape.
    """
    from torch.utils.data import DataLoader

    pyt = PytorchEngineDataset(synthetic_classification_dataset)
    loader = DataLoader(pyt, batch_size=1, collate_fn=lambda batch: batch)
    batches = list(loader)
    assert len(batches) == 2
    assert isinstance(batches[0], list)
    assert "image" in batches[0][0]


def test_pytorch_dataset_detection_shape(synthetic_detection_dataset):
    pyt = PytorchEngineDataset(synthetic_detection_dataset)
    item = pyt[0]
    assert "image" in item
    assert "bbox" in item
    assert len(item["bbox"]) == 2  # two bboxes per sample in fixture
