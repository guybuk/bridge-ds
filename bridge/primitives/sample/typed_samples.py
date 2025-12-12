"""
Typed sample subclasses for common multi-role patterns.

These classes provide static typing and IDE autocomplete support for
samples from typed datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from bridge.primitives.sample.multi_role_sample import MultiRoleSample

if TYPE_CHECKING:
    from bridge.primitives.element.element import Element


class ImageLabelSample(MultiRoleSample):
    """Sample from an ImageLabelDataset (image classification)."""

    @property
    def image(self) -> List[Element]:
        """List of image elements."""
        return self.get_role("image")

    @property
    def label(self) -> List[Element]:
        """List of label elements."""
        return self.get_role("label")


class DetectionSample(MultiRoleSample):
    """Sample from a DetectionDataset (object detection)."""

    @property
    def image(self) -> List[Element]:
        """List of image elements."""
        return self.get_role("image")

    @property
    def bbox(self) -> List[Element]:
        """List of bounding box elements."""
        return self.get_role("bbox")


class PairedSample(MultiRoleSample):
    """Sample from a PairedDataset (translation, image-to-image, etc.)."""

    @property
    def source(self) -> List[Element]:
        """List of source elements."""
        return self.get_role("source")

    @property
    def target(self) -> List[Element]:
        """List of target elements."""
        return self.get_role("target")


class TextImageSample(MultiRoleSample):
    """Sample from a TextImageDataset (text-image pairs)."""

    @property
    def text(self) -> List[Element]:
        """List of text elements."""
        return self.get_role("text")

    @property
    def image(self) -> List[Element]:
        """List of image elements."""
        return self.get_role("image")


class TextLabelSample(MultiRoleSample):
    """Sample from a TextLabelDataset (text classification)."""

    @property
    def text(self) -> List[Element]:
        """List of text elements."""
        return self.get_role("text")

    @property
    def label(self) -> List[Element]:
        """List of label elements."""
        return self.get_role("label")
