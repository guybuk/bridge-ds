"""Tests for multi-role ImagePairs provider."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bridge.providers.vision import ImagePairs


def _write_jpeg(path: Path, h: int = 16, w: int = 16) -> None:
    arr = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG")


@pytest.fixture
def pair_root(tmp_path) -> Path:
    """Build root/source/{a,b,c}.jpg + root/target/{a,b,c}.jpg."""
    for sub in ("source", "target"):
        d = tmp_path / sub
        d.mkdir()
        for name in ("a", "b", "c"):
            _write_jpeg(d / f"{name}.jpg")
    return tmp_path


def test_image_pairs_builds_dataset(pair_root):
    ds = ImagePairs(pair_root).build_dataset()
    assert len(ds) == 3


def test_image_pairs_has_two_image_roles(pair_root):
    ds = ImagePairs(pair_root).build_dataset()
    sample = ds.iget(0)
    assert "source" in sample.elements
    assert "target" in sample.elements
    assert sample.elements["source"][0].etype == "image"
    assert sample.elements["target"][0].etype == "image"


def test_image_pairs_data_loadable(pair_root):
    ds = ImagePairs(pair_root).build_dataset()
    sample = ds.iget(0)
    src = sample.one("source").data
    tgt = sample.one("target").data
    assert src.shape == (16, 16, 3)
    assert tgt.shape == (16, 16, 3)


def test_image_pairs_missing_pair_raises(tmp_path):
    """If target/ has fewer files than source/, the provider should refuse to build."""
    (tmp_path / "source").mkdir()
    (tmp_path / "target").mkdir()
    _write_jpeg(tmp_path / "source" / "a.jpg")
    _write_jpeg(tmp_path / "source" / "b.jpg")
    _write_jpeg(tmp_path / "target" / "a.jpg")
    with pytest.raises((ValueError, AssertionError)):
        ImagePairs(tmp_path).build_dataset()
