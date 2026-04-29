"""Smoke tests for ImagePairsPanelEngine on synthetic pair data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bridge.display.vision import ImagePairsPanelEngine
from bridge.providers.vision import ImagePairs


def _write_jpeg(path: Path, h: int = 16, w: int = 16) -> None:
    arr = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG")


@pytest.fixture
def pair_dataset(tmp_path):
    for sub in ("source", "target"):
        d = tmp_path / sub
        d.mkdir()
        for name in ("a", "b"):
            _write_jpeg(d / f"{name}.jpg")
    return ImagePairs(tmp_path).build_dataset()


def test_image_pairs_engine_show_sample(pair_dataset):
    engine = ImagePairsPanelEngine()
    out = engine.show_sample(pair_dataset.iget(0))
    assert out is not None


def test_image_pairs_engine_show_element_image(pair_dataset):
    engine = ImagePairsPanelEngine()
    sample = pair_dataset.iget(0)
    out = engine.show_element(sample.one("source"))
    assert out is not None


def test_image_pairs_engine_show_dataset(pair_dataset):
    engine = ImagePairsPanelEngine()
    out = engine.show_dataset(pair_dataset)
    assert out is not None


def test_image_pairs_provider_default_engine_is_image_pairs(pair_dataset):
    """The provider should default to the matching shape engine."""
    assert isinstance(pair_dataset._display_engine, ImagePairsPanelEngine)
