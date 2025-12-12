"""
Typed dataset subclasses for common multi-role patterns.

These classes provide static typing and IDE autocomplete support for
common dataset patterns like classification, object detection, etc.

For custom role combinations, use MultiRoleDataset directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import pandas as pd

from bridge.primitives.dataset.multi_role_dataset import MultiRoleDataset
from bridge.primitives.sample.typed_samples import (
    DetectionSample,
    ImageLabelSample,
    PairedSample,
    TextImageSample,
    TextLabelSample,
)
from bridge.utils.constants import INDICES

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism
    from bridge.primitives.element.element import Element


class ImageLabelDataset(MultiRoleDataset):
    """
    Dataset for image classification tasks.

    Roles:
        - image: Image elements
        - label: Classification label elements

    Example:
        ds = ImageLabelDataset.from_dict({
            "image": image_elements,
            "label": label_elements,
        })
        ds.image  # DataFrame of image elements
        ds.label  # DataFrame of label elements
        sample = ds.iget(0)
        sample.image  # List[Element] for images
        sample.label  # List[Element] for labels
    """

    _expected_roles = ("image", "label")

    @property
    def image(self) -> pd.DataFrame:
        """DataFrame of image elements."""
        return self.get_role("image")

    @property
    def label(self) -> pd.DataFrame:
        """DataFrame of label elements."""
        return self.get_role("label")

    def iget(self, index: int) -> ImageLabelSample:
        return ImageLabelSample.from_sample(super(MultiRoleDataset, self).iget(index), self._role_names)

    def get(self, sample_id) -> ImageLabelSample:
        return ImageLabelSample.from_sample(super(MultiRoleDataset, self).get(sample_id), self._role_names)

    @classmethod
    def from_dict(
        cls,
        element_lists: Dict[str, List[Element]],
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> "ImageLabelDataset":
        _validate_roles(element_lists, cls._expected_roles, cls.__name__)
        element_groups = _elements_to_dataframes(element_lists)
        return cls(element_groups, display_engine=display_engine, cache_mechanisms=cache_mechanisms)


class DetectionDataset(MultiRoleDataset):
    """
    Dataset for object detection tasks.

    Roles:
        - image: Image elements
        - bbox: Bounding box elements (can have multiple per sample)

    Example:
        ds = DetectionDataset.from_dict({
            "image": image_elements,
            "bbox": bbox_elements,
        })
        ds.image  # DataFrame of image elements
        ds.bbox   # DataFrame of bounding box elements
    """

    _expected_roles = ("image", "bbox")

    @property
    def image(self) -> pd.DataFrame:
        """DataFrame of image elements."""
        return self.get_role("image")

    @property
    def bbox(self) -> pd.DataFrame:
        """DataFrame of bounding box elements."""
        return self.get_role("bbox")

    def iget(self, index: int) -> DetectionSample:
        return DetectionSample.from_sample(super(MultiRoleDataset, self).iget(index), self._role_names)

    def get(self, sample_id) -> DetectionSample:
        return DetectionSample.from_sample(super(MultiRoleDataset, self).get(sample_id), self._role_names)

    @classmethod
    def from_dict(
        cls,
        element_lists: Dict[str, List[Element]],
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> "DetectionDataset":
        _validate_roles(element_lists, cls._expected_roles, cls.__name__)
        element_groups = _elements_to_dataframes(element_lists)
        return cls(element_groups, display_engine=display_engine, cache_mechanisms=cache_mechanisms)


class PairedDataset(MultiRoleDataset):
    """
    Dataset for paired data tasks (translation, image-to-image, etc.).

    Roles:
        - source: Source elements (input)
        - target: Target elements (output)

    Example:
        # Translation
        ds = PairedDataset.from_dict({
            "source": english_elements,
            "target": french_elements,
        })

        # Image-to-image
        ds = PairedDataset.from_dict({
            "source": input_images,
            "target": output_images,
        })

        ds.source  # DataFrame of source elements
        ds.target  # DataFrame of target elements
    """

    _expected_roles = ("source", "target")

    @property
    def source(self) -> pd.DataFrame:
        """DataFrame of source elements."""
        return self.get_role("source")

    @property
    def target(self) -> pd.DataFrame:
        """DataFrame of target elements."""
        return self.get_role("target")

    def iget(self, index: int) -> PairedSample:
        return PairedSample.from_sample(super(MultiRoleDataset, self).iget(index), self._role_names)

    def get(self, sample_id) -> PairedSample:
        return PairedSample.from_sample(super(MultiRoleDataset, self).get(sample_id), self._role_names)

    @classmethod
    def from_dict(
        cls,
        element_lists: Dict[str, List[Element]],
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> "PairedDataset":
        _validate_roles(element_lists, cls._expected_roles, cls.__name__)
        element_groups = _elements_to_dataframes(element_lists)
        return cls(element_groups, display_engine=display_engine, cache_mechanisms=cache_mechanisms)


class TextImageDataset(MultiRoleDataset):
    """
    Dataset for text-image tasks (captioning, text-to-image, etc.).

    Roles:
        - text: Text/caption elements
        - image: Image elements

    Example:
        ds = TextImageDataset.from_dict({
            "text": caption_elements,
            "image": image_elements,
        })
        ds.text   # DataFrame of text elements
        ds.image  # DataFrame of image elements
    """

    _expected_roles = ("text", "image")

    @property
    def text(self) -> pd.DataFrame:
        """DataFrame of text elements."""
        return self.get_role("text")

    @property
    def image(self) -> pd.DataFrame:
        """DataFrame of image elements."""
        return self.get_role("image")

    def iget(self, index: int) -> TextImageSample:
        return TextImageSample.from_sample(super(MultiRoleDataset, self).iget(index), self._role_names)

    def get(self, sample_id) -> TextImageSample:
        return TextImageSample.from_sample(super(MultiRoleDataset, self).get(sample_id), self._role_names)

    @classmethod
    def from_dict(
        cls,
        element_lists: Dict[str, List[Element]],
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> "TextImageDataset":
        _validate_roles(element_lists, cls._expected_roles, cls.__name__)
        element_groups = _elements_to_dataframes(element_lists)
        return cls(element_groups, display_engine=display_engine, cache_mechanisms=cache_mechanisms)


class TextLabelDataset(MultiRoleDataset):
    """
    Dataset for text classification tasks (sentiment analysis, etc.).

    Roles:
        - text: Text elements
        - label: Classification label elements

    Example:
        ds = TextLabelDataset.from_dict({
            "text": text_elements,
            "label": label_elements,
        })
        ds.text   # DataFrame of text elements
        ds.label  # DataFrame of label elements
    """

    _expected_roles = ("text", "label")

    @property
    def text(self) -> pd.DataFrame:
        """DataFrame of text elements."""
        return self.get_role("text")

    @property
    def label(self) -> pd.DataFrame:
        """DataFrame of label elements."""
        return self.get_role("label")

    def iget(self, index: int) -> TextLabelSample:
        return TextLabelSample.from_sample(super(MultiRoleDataset, self).iget(index), self._role_names)

    def get(self, sample_id) -> TextLabelSample:
        return TextLabelSample.from_sample(super(MultiRoleDataset, self).get(sample_id), self._role_names)

    @classmethod
    def from_dict(
        cls,
        element_lists: Dict[str, List[Element]],
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> "TextLabelDataset":
        _validate_roles(element_lists, cls._expected_roles, cls.__name__)
        element_groups = _elements_to_dataframes(element_lists)
        return cls(element_groups, display_engine=display_engine, cache_mechanisms=cache_mechanisms)


def _validate_roles(element_lists: Dict[str, List[Element]], expected_roles: tuple, class_name: str) -> None:
    """Validate that the provided roles match the expected roles."""
    provided_roles = set(element_lists.keys())
    expected_set = set(expected_roles)
    if provided_roles != expected_set:
        raise ValueError(
            f"{class_name} expects roles {expected_roles}, but got {tuple(provided_roles)}. "
            f"Use MultiRoleDataset for custom role combinations."
        )


def _elements_to_dataframes(element_lists: Dict[str, List[Element]]) -> Dict[str, pd.DataFrame]:
    """Convert element lists to DataFrames."""
    element_groups = {}
    for role, elements in element_lists.items():
        records = [e.to_dict() for e in elements]
        element_groups[role] = pd.DataFrame(records).set_index(INDICES)
    return element_groups
