"""Unit tests for vision DisplayEngines.

Exercises show_element / show_sample / show_dataset on tiny synthetic datasets.
We assert that calls return non-None outputs of plausible types; we do not render.
"""

from __future__ import annotations

import pytest

from bridge.display.basic import SimplePrints
from bridge.display.vision import Panel


@pytest.fixture
def panel_engine() -> Panel:
    return Panel(bbox_format="xyxy")


@pytest.fixture
def panel_engine_xywh() -> Panel:
    return Panel(bbox_format="xywh")


def test_panel_show_element_image(panel_engine, synthetic_classification_dataset):
    sample = synthetic_classification_dataset.iget(0)
    out = panel_engine.show_element(sample.one("image"))
    assert out is not None


def test_panel_show_element_bbox(panel_engine, synthetic_detection_dataset):
    sample = synthetic_detection_dataset.iget(0)
    bbox_element = sample.elements["bbox"][0]
    out = panel_engine.show_element(bbox_element)
    assert out is not None


def test_panel_show_element_unknown_etype_raises(panel_engine, synthetic_classification_dataset):
    sample = synthetic_classification_dataset.iget(0)
    label_element = sample.elements["class_label"][0]
    with pytest.raises(NotImplementedError):
        panel_engine.show_element(label_element)


def test_panel_show_sample_classification(panel_engine, synthetic_classification_dataset):
    sample = synthetic_classification_dataset.iget(0)
    out = panel_engine.show_sample(sample)
    assert out is not None


def test_panel_show_sample_detection(panel_engine, synthetic_detection_dataset):
    sample = synthetic_detection_dataset.iget(0)
    out = panel_engine.show_sample(sample)
    assert out is not None


def test_panel_show_dataset(panel_engine, synthetic_detection_dataset):
    out = panel_engine.show_dataset(synthetic_detection_dataset)
    assert out is not None


def test_panel_xywh_format_does_not_raise(panel_engine_xywh, synthetic_detection_dataset):
    sample = synthetic_detection_dataset.iget(0)
    out = panel_engine_xywh.show_sample(sample)
    assert out is not None


def test_simple_prints_show_element(synthetic_classification_dataset, capsys):
    engine = SimplePrints()
    sample = synthetic_classification_dataset.iget(0)
    image = sample.one("image")
    engine.show_element(image)
    captured = capsys.readouterr()
    assert image.id in captured.out or str(image.id) in captured.out


def test_simple_prints_show_sample(synthetic_classification_dataset, capsys):
    engine = SimplePrints()
    engine.show_sample(synthetic_classification_dataset.iget(0))
    captured = capsys.readouterr()
    assert "Sample ID" in captured.out


def test_simple_prints_show_dataset(synthetic_detection_dataset, capsys):
    engine = SimplePrints()
    engine.show_dataset(synthetic_detection_dataset)
    captured = capsys.readouterr()
    assert "Sample ID" in captured.out
