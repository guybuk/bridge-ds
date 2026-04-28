"""Shared fixtures for vision tests.

Uses synthetic in-memory data so unit tests run without network or large downloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from bridge.primitives.dataset import SingularDataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.utils.data_objects import BoundingBox, ClassLabel


@pytest.fixture(scope="session", autouse=True)
def _holoviews_bokeh_backend():
    """Panel display engine asserts the holoviews backend is bokeh; notebooks
    set this via hv.extension('bokeh'). Match that here so tests don't need
    to repeat the dance."""
    import holoviews as hv

    hv.extension("bokeh")


@pytest.fixture
def synthetic_image() -> np.ndarray:
    return np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_classification_dataset(synthetic_image) -> SingularDataset:
    """Two-sample image classification dataset, fully in-memory."""
    images = []
    labels = []
    for i in range(2):
        images.append(
            Element(
                element_id=f"img_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism(synthetic_image.copy(), category="image"),
            )
        )
        labels.append(
            Element(
                element_id=f"lbl_{i}",
                sample_id=i,
                etype="class_label",
                load_mechanism=LoadMechanism(ClassLabel(class_idx=i, class_name=f"class_{i}"), category="obj"),
            )
        )
    return SingularDataset.from_lists(images, labels)


@pytest.fixture
def synthetic_detection_dataset(synthetic_image) -> SingularDataset:
    """Two-sample detection dataset with bboxes, fully in-memory."""
    images = []
    bboxes = []
    for i in range(2):
        images.append(
            Element(
                element_id=f"img_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism(synthetic_image.copy(), category="image"),
            )
        )
        for j in range(2):
            bbox = BoundingBox(
                coords=np.array([j * 10.0, j * 10.0, (j + 1) * 20.0, (j + 1) * 20.0]),
                class_label=ClassLabel(class_idx=j, class_name=f"class_{j}"),
            )
            bboxes.append(
                Element(
                    element_id=f"bbox_{i}_{j}",
                    sample_id=i,
                    etype="bbox",
                    load_mechanism=LoadMechanism(bbox, category="obj"),
                )
            )
    return SingularDataset.from_lists(images, bboxes)
