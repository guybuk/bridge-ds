from __future__ import annotations

from bridge.utils import StrEnum


class ElementType(StrEnum):
    image = "image"
    class_label = "class_label"
    bbox = "bbox"
    text = "text"
    segmentation = "segmentation"
    keypoint = "keypoint"
