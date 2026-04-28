"""Unit tests for vision SampleTransforms."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
v2 = pytest.importorskip("torchvision.transforms.v2")

from bridge.primitives.sample.transform.vision import TorchvisionV2Transform


def test_horizontal_flip_preserves_image_shape(synthetic_detection_dataset):
    sample = synthetic_detection_dataset.iget(0)
    original_shape = sample.element.data.shape

    transform = TorchvisionV2Transform([v2.RandomHorizontalFlip(p=1.0)], bbox_format="XYXY")
    transformed = sample.transform(transform)

    out = transformed.element.data
    if isinstance(out, torch.Tensor):
        # tensor format CHW
        assert out.shape[-2:] == original_shape[:2]
    else:
        assert out.shape[:2] == original_shape[:2]


def test_transform_preserves_bbox_count_for_geometric_only(synthetic_detection_dataset):
    sample = synthetic_detection_dataset.iget(0)
    n_bboxes_before = len(sample.annotations["bbox"])

    transform = TorchvisionV2Transform([v2.RandomHorizontalFlip(p=1.0)], bbox_format="XYXY")
    transformed = sample.transform(transform)

    assert len(transformed.elements["bbox"]) == n_bboxes_before


def test_transform_returns_singular_sample(synthetic_detection_dataset):
    """SingularSample.transform should return a SingularSample, not the base Sample."""
    from bridge.primitives.sample.singular_sample import SingularSample

    sample = synthetic_detection_dataset.iget(0)
    transform = TorchvisionV2Transform([v2.RandomHorizontalFlip(p=1.0)], bbox_format="XYXY")
    transformed = sample.transform(transform)

    assert isinstance(transformed, SingularSample)


def test_transform_raises_without_image(synthetic_classification_dataset, mocker):
    sample = synthetic_classification_dataset.iget(0)
    # SingularSample.elements always contains the image; simulate the bad case
    # where image is missing by removing it from the dict.
    mocker.patch.object(sample, "_elements", {})
    transform = TorchvisionV2Transform([v2.Identity()])

    with pytest.raises(ValueError, match="image"):
        transform(sample, cache_mechanisms={}, display_engine=None)


def test_transform_changes_encoding_to_torch(synthetic_detection_dataset):
    """When ToImage + ToDtype are applied, output encoding should swap to 'pt'."""
    sample = synthetic_detection_dataset.iget(0)
    transform = TorchvisionV2Transform(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)],
        bbox_format="XYXY",
    )
    transformed = sample.transform(transform)
    assert transformed.element.encoding == "pt"
    assert isinstance(transformed.element.data, torch.Tensor)


def test_transform_resize_changes_image_shape(synthetic_detection_dataset):
    sample = synthetic_detection_dataset.iget(0)
    transform = TorchvisionV2Transform(
        [v2.ToImage(), v2.Resize((32, 32))],
        bbox_format="XYXY",
    )
    transformed = sample.transform(transform)
    out = transformed.element.data
    # tensor in CHW format
    assert out.shape[-2:] == (32, 32)


def test_chained_transforms(synthetic_detection_dataset):
    sample = synthetic_detection_dataset.iget(0)
    flip = TorchvisionV2Transform([v2.RandomHorizontalFlip(p=1.0)], bbox_format="XYXY")
    resize = TorchvisionV2Transform(
        [v2.ToImage(), v2.Resize((32, 32))],
        bbox_format="XYXY",
    )

    flipped = sample.transform(flip)
    resized = flipped.transform(resize)

    assert resized.element.data.shape[-2:] == (32, 32)
