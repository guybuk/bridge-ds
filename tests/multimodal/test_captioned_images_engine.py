"""Smoke tests for CaptionedImagesPanelEngine."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bridge.display.multimodal import CaptionedImagesPanelEngine
from bridge.providers.multimodal import CaptionedImages


def _write_jpeg(path: Path, h: int = 16, w: int = 16) -> None:
    arr = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG")


@pytest.fixture
def captioned_dataset(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    captions = {}
    for name in ("a", "b"):
        _write_jpeg(img_dir / f"{name}.jpg")
        captions[name] = f"Caption for {name}"
    (tmp_path / "captions.json").write_text(json.dumps(captions))
    return CaptionedImages(tmp_path).build_dataset()


def test_captioned_engine_show_sample(captioned_dataset):
    engine = CaptionedImagesPanelEngine()
    out = engine.show_sample(captioned_dataset.iget(0))
    assert out is not None


def test_captioned_engine_show_element_image(captioned_dataset):
    engine = CaptionedImagesPanelEngine()
    sample = captioned_dataset.iget(0)
    out = engine.show_element(sample.one("image"))
    assert out is not None


def test_captioned_engine_show_element_caption(captioned_dataset):
    engine = CaptionedImagesPanelEngine()
    sample = captioned_dataset.iget(0)
    out = engine.show_element(sample.one("caption"))
    assert out is not None


def test_captioned_engine_show_dataset(captioned_dataset):
    engine = CaptionedImagesPanelEngine()
    out = engine.show_dataset(captioned_dataset)
    assert out is not None


def test_captioned_provider_default_engine_is_captioned(captioned_dataset):
    """Provider should default to the matching shape engine."""
    assert isinstance(captioned_dataset._display_engine, CaptionedImagesPanelEngine)
