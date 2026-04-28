"""Unit tests for DatasetProviders that operate on local data.

Skips providers requiring real downloads (Coco, IMDb in download mode, CIFAR).
Uses tmp_path to fabricate a tiny on-disk dataset matching each provider's
expected directory layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bridge.providers.text import LargeMovieReviewDataset
from bridge.providers.vision import ImageFolder


def _write_jpeg(path: Path, h: int = 32, w: int = 32) -> None:
    arr = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG")


@pytest.fixture
def imagefolder_root(tmp_path) -> Path:
    """Build a 2-class ImageFolder layout with 3 images per class."""
    for class_name in ("cat", "dog"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(3):
            _write_jpeg(class_dir / f"{i}.jpg")
    return tmp_path


@pytest.fixture
def imdb_root(tmp_path) -> Path:
    """Build a tiny IMDb-like layout: aclImdb/train/{pos,neg}/*.txt.

    Real IMDb filenames are <id>_<rating>.txt with disjoint ratings across
    classes (1-4 neg, 7-10 pos) so stems never collide. Mirror that: use
    different id ranges per class.
    """
    train = tmp_path / "aclImdb" / "train"
    for class_name, base in (("neg", 0), ("pos", 100)):
        d = train / class_name
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"{base + i}_1.txt").write_text(f"sample review {i} for {class_name}")
    return tmp_path


def test_imagefolder_builds_dataset(imagefolder_root):
    provider = ImageFolder(imagefolder_root)
    ds = provider.build_dataset()
    assert len(ds) == 6  # 2 classes × 3 images


def test_imagefolder_has_image_and_label_per_sample(imagefolder_root):
    provider = ImageFolder(imagefolder_root)
    ds = provider.build_dataset()

    for sample in ds:
        assert sample.element.etype == "image"
        assert "class_label" in sample.annotations
        assert len(sample.annotations["class_label"]) == 1


def test_imagefolder_class_labels_distinct(imagefolder_root):
    provider = ImageFolder(imagefolder_root)
    ds = provider.build_dataset()

    class_indices = set()
    for sample in ds:
        cl = sample.annotations["class_label"][0].data
        class_indices.add(cl.class_idx)

    assert class_indices == {0, 1}


def test_imagefolder_image_data_loadable(imagefolder_root):
    provider = ImageFolder(imagefolder_root)
    ds = provider.build_dataset()

    sample = ds.iget(0)
    img = sample.element.data
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.uint8


def test_imdb_provider_builds_dataset(imdb_root):
    provider = LargeMovieReviewDataset(root=imdb_root, split="train", download=False)
    ds = provider.build_dataset()
    assert len(ds) == 6  # 2 classes × 3 reviews


def test_imdb_provider_has_text_and_label(imdb_root):
    provider = LargeMovieReviewDataset(root=imdb_root, split="train", download=False)
    ds = provider.build_dataset()

    for sample in ds:
        assert sample.element.etype == "text"
        assert "class_label" in sample.annotations


def test_imdb_provider_text_loadable(imdb_root):
    provider = LargeMovieReviewDataset(root=imdb_root, split="train", download=False)
    ds = provider.build_dataset()

    sample = ds.iget(0)
    text = sample.element.data
    assert isinstance(text, str)
    assert "sample review" in text
