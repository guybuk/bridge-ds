"""Tests for CaptionedImages multi-role provider."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bridge.providers.multimodal import CaptionedImages


def _write_jpeg(path: Path, h: int = 16, w: int = 16) -> None:
    arr = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG")


@pytest.fixture
def captioned_root(tmp_path) -> Path:
    """root/images/{a,b}.jpg + root/captions.json with both stems."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    captions = {}
    for name in ("a", "b"):
        _write_jpeg(img_dir / f"{name}.jpg")
        captions[name] = f"A caption for {name}"
    (tmp_path / "captions.json").write_text(json.dumps(captions))
    return tmp_path


def test_captioned_images_builds_dataset(captioned_root):
    ds = CaptionedImages(captioned_root).build_dataset()
    assert len(ds) == 2


def test_captioned_images_has_image_and_caption_roles(captioned_root):
    ds = CaptionedImages(captioned_root).build_dataset()
    sample = ds.iget(0)
    assert "image" in sample.elements
    assert "caption" in sample.elements
    assert sample.elements["image"][0].etype == "image"
    assert sample.elements["caption"][0].etype == "text"


def test_captioned_images_caption_data_loadable(captioned_root):
    ds = CaptionedImages(captioned_root).build_dataset()
    for sample in ds:
        cap = sample.one("caption").data
        assert isinstance(cap, str)
        assert cap.startswith("A caption for ")


def test_captioned_images_missing_caption_raises(tmp_path):
    """Image with no caption entry → provider refuses to build."""
    (tmp_path / "images").mkdir()
    _write_jpeg(tmp_path / "images" / "a.jpg")
    _write_jpeg(tmp_path / "images" / "b.jpg")
    (tmp_path / "captions.json").write_text(json.dumps({"a": "only one"}))
    with pytest.raises(ValueError, match="b"):
        CaptionedImages(tmp_path).build_dataset()


def test_captioned_images_share_sample_id(captioned_root):
    """Image and caption elements for the same sample share sample_id."""
    ds = CaptionedImages(captioned_root).build_dataset()
    for sample in ds:
        assert sample.one("image").sample_id == sample.one("caption").sample_id
