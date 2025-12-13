"""Tests for typed dataset subclasses."""

import numpy as np
import pytest

from bridge.primitives.dataset import (
    DetectionDataset,
    ImageLabelDataset,
    MultiRoleDataset,
    PairedDataset,
    TextImageDataset,
    TextLabelDataset,
)
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import (
    DetectionSample,
    ImageLabelSample,
    PairedSample,
    TextImageSample,
    TextLabelSample,
)


@pytest.fixture
def image_label_elements():
    """Create image classification elements."""
    images = []
    labels = []
    for i in range(10):
        images.append(
            Element(
                element_id=f"image_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism(np.zeros((32, 32, 3), dtype=np.uint8), category="obj"),
            )
        )
        labels.append(
            Element(
                element_id=f"label_{i}",
                sample_id=i,
                etype="label",
                load_mechanism=LoadMechanism({"class": i % 3}, category="obj"),
            )
        )
    return images, labels


@pytest.fixture
def detection_elements():
    """Create object detection elements."""
    images = []
    bboxes = []
    for i in range(10):
        images.append(
            Element(
                element_id=f"image_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism(np.zeros((64, 64, 3), dtype=np.uint8), category="obj"),
            )
        )
        # Multiple bboxes per image
        for j in range(3):
            bboxes.append(
                Element(
                    element_id=f"bbox_{i}_{j}",
                    sample_id=i,
                    etype="bbox",
                    load_mechanism=LoadMechanism({"x": 10, "y": 10, "w": 20, "h": 20}, category="obj"),
                )
            )
    return images, bboxes


@pytest.fixture
def paired_elements():
    """Create paired elements (source/target)."""
    sources = []
    targets = []
    for i in range(10):
        sources.append(
            Element(
                element_id=f"source_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism(np.zeros((32, 32, 3), dtype=np.uint8), category="obj"),
            )
        )
        targets.append(
            Element(
                element_id=f"target_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism(np.ones((32, 32, 3), dtype=np.uint8), category="obj"),
            )
        )
    return sources, targets


@pytest.fixture
def text_image_elements():
    """Create text-image pair elements."""
    texts = []
    images = []
    for i in range(10):
        texts.append(
            Element(
                element_id=f"text_{i}",
                sample_id=i,
                etype="text",
                load_mechanism=LoadMechanism(f"Caption {i}", category="obj"),
            )
        )
        images.append(
            Element(
                element_id=f"image_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism(np.zeros((32, 32, 3), dtype=np.uint8), category="obj"),
            )
        )
    return texts, images


@pytest.fixture
def text_label_elements():
    """Create text classification elements."""
    texts = []
    labels = []
    for i in range(10):
        texts.append(
            Element(
                element_id=f"text_{i}",
                sample_id=i,
                etype="text",
                load_mechanism=LoadMechanism(f"Sample text {i}", category="obj"),
            )
        )
        labels.append(
            Element(
                element_id=f"label_{i}",
                sample_id=i,
                etype="label",
                load_mechanism=LoadMechanism({"sentiment": "positive" if i % 2 == 0 else "negative"}, category="obj"),
            )
        )
    return texts, labels


class TestImageLabelDataset:
    def test_creation(self, image_label_elements):
        images, labels = image_label_elements
        ds = ImageLabelDataset.from_dict({"image": images, "label": labels})
        assert len(ds) == 10
        assert isinstance(ds, ImageLabelDataset)
        assert isinstance(ds, MultiRoleDataset)

    def test_typed_properties(self, image_label_elements):
        images, labels = image_label_elements
        ds = ImageLabelDataset.from_dict({"image": images, "label": labels})
        # Static type checker knows these are pd.DataFrame
        assert len(ds.image) == 10
        assert len(ds.label) == 10

    def test_iget_returns_typed_sample(self, image_label_elements):
        images, labels = image_label_elements
        ds = ImageLabelDataset.from_dict({"image": images, "label": labels})
        sample = ds.iget(0)
        assert isinstance(sample, ImageLabelSample)
        # Static type checker knows these are List[Element]
        assert len(sample.image) == 1
        assert len(sample.label) == 1

    def test_wrong_roles_raises(self, image_label_elements):
        images, labels = image_label_elements
        with pytest.raises(ValueError, match="expects roles"):
            ImageLabelDataset.from_dict({"img": images, "lbl": labels})


class TestDetectionDataset:
    def test_creation(self, detection_elements):
        images, bboxes = detection_elements
        ds = DetectionDataset.from_dict({"image": images, "bbox": bboxes})
        assert len(ds) == 10

    def test_typed_properties(self, detection_elements):
        images, bboxes = detection_elements
        ds = DetectionDataset.from_dict({"image": images, "bbox": bboxes})
        assert len(ds.image) == 10
        assert len(ds.bbox) == 30  # 3 bboxes per sample

    def test_iget_returns_typed_sample(self, detection_elements):
        images, bboxes = detection_elements
        ds = DetectionDataset.from_dict({"image": images, "bbox": bboxes})
        sample = ds.iget(0)
        assert isinstance(sample, DetectionSample)
        assert len(sample.image) == 1
        assert len(sample.bbox) == 3


class TestPairedDataset:
    def test_creation(self, paired_elements):
        sources, targets = paired_elements
        ds = PairedDataset.from_dict({"source": sources, "target": targets})
        assert len(ds) == 10

    def test_typed_properties(self, paired_elements):
        sources, targets = paired_elements
        ds = PairedDataset.from_dict({"source": sources, "target": targets})
        assert len(ds.source) == 10
        assert len(ds.target) == 10

    def test_iget_returns_typed_sample(self, paired_elements):
        sources, targets = paired_elements
        ds = PairedDataset.from_dict({"source": sources, "target": targets})
        sample = ds.iget(0)
        assert isinstance(sample, PairedSample)
        assert len(sample.source) == 1
        assert len(sample.target) == 1


class TestTextImageDataset:
    def test_creation(self, text_image_elements):
        texts, images = text_image_elements
        ds = TextImageDataset.from_dict({"text": texts, "image": images})
        assert len(ds) == 10

    def test_typed_properties(self, text_image_elements):
        texts, images = text_image_elements
        ds = TextImageDataset.from_dict({"text": texts, "image": images})
        assert len(ds.text) == 10
        assert len(ds.image) == 10

    def test_iget_returns_typed_sample(self, text_image_elements):
        texts, images = text_image_elements
        ds = TextImageDataset.from_dict({"text": texts, "image": images})
        sample = ds.iget(0)
        assert isinstance(sample, TextImageSample)
        assert len(sample.text) == 1
        assert len(sample.image) == 1


class TestTextLabelDataset:
    def test_creation(self, text_label_elements):
        texts, labels = text_label_elements
        ds = TextLabelDataset.from_dict({"text": texts, "label": labels})
        assert len(ds) == 10

    def test_typed_properties(self, text_label_elements):
        texts, labels = text_label_elements
        ds = TextLabelDataset.from_dict({"text": texts, "label": labels})
        assert len(ds.text) == 10
        assert len(ds.label) == 10

    def test_iget_returns_typed_sample(self, text_label_elements):
        texts, labels = text_label_elements
        ds = TextLabelDataset.from_dict({"text": texts, "label": labels})
        sample = ds.iget(0)
        assert isinstance(sample, TextLabelSample)
        assert len(sample.text) == 1
        assert len(sample.label) == 1


class TestTypedDatasetsFallbackToMultiRole:
    """Test that users can still use MultiRoleDataset for custom role combinations."""

    def test_custom_roles_with_multi_role_dataset(self, paired_elements):
        sources, targets = paired_elements
        # Custom role names work with MultiRoleDataset
        ds = MultiRoleDataset.from_dict({"english": sources, "french": targets})
        assert ds.role_names == ("english", "french")
        assert len(ds.english) == 10
        assert len(ds.french) == 10


class TestSampleIdValidation:
    """Test sample ID validation for typed datasets."""

    def test_symmetric_validation_mismatched_sample_ids_raises(self):
        """ImageLabelDataset requires 1:1 sample ID correspondence."""
        images = [
            Element(
                element_id="image_0",
                sample_id=0,
                etype="image",
                load_mechanism=LoadMechanism(np.zeros((32, 32, 3), dtype=np.uint8), category="obj"),
            )
        ]
        labels = [
            Element(
                element_id="label_1",
                sample_id=1,  # Different sample_id
                etype="label",
                load_mechanism=LoadMechanism({"class": 0}, category="obj"),
            )
        ]
        with pytest.raises(ValueError, match="requires 1:1 sample ID correspondence"):
            ImageLabelDataset.from_dict({"image": images, "label": labels})

    def test_symmetric_validation_extra_in_one_role_raises(self):
        """ImageLabelDataset requires identical sample IDs in both roles."""
        images = [
            Element(
                element_id="image_0",
                sample_id=0,
                etype="image",
                load_mechanism=LoadMechanism(np.zeros((32, 32, 3), dtype=np.uint8), category="obj"),
            ),
            Element(
                element_id="image_1",
                sample_id=1,
                etype="image",
                load_mechanism=LoadMechanism(np.zeros((32, 32, 3), dtype=np.uint8), category="obj"),
            ),
        ]
        labels = [
            Element(
                element_id="label_0",
                sample_id=0,
                etype="label",
                load_mechanism=LoadMechanism({"class": 0}, category="obj"),
            )
        ]
        with pytest.raises(ValueError, match="requires 1:1 sample ID correspondence"):
            ImageLabelDataset.from_dict({"image": images, "label": labels})

    def test_asymmetric_validation_bbox_not_in_image_raises(self):
        """DetectionDataset requires bbox sample IDs to be subset of image sample IDs."""
        images = [
            Element(
                element_id="image_0",
                sample_id=0,
                etype="image",
                load_mechanism=LoadMechanism(np.zeros((32, 32, 3), dtype=np.uint8), category="obj"),
            )
        ]
        bboxes = [
            Element(
                element_id="bbox_1",
                sample_id=1,  # sample_id not in images
                etype="bbox",
                load_mechanism=LoadMechanism({"coords": [0, 0, 10, 10]}, category="obj"),
            )
        ]
        with pytest.raises(ValueError, match="has sample IDs not present in"):
            DetectionDataset.from_dict({"image": images, "bbox": bboxes})

    def test_asymmetric_validation_images_without_bboxes_allowed(self):
        """DetectionDataset allows images without bboxes."""
        images = [
            Element(
                element_id="image_0",
                sample_id=0,
                etype="image",
                load_mechanism=LoadMechanism(np.zeros((32, 32, 3), dtype=np.uint8), category="obj"),
            ),
            Element(
                element_id="image_1",
                sample_id=1,
                etype="image",
                load_mechanism=LoadMechanism(np.zeros((32, 32, 3), dtype=np.uint8), category="obj"),
            ),
        ]
        bboxes = [
            Element(
                element_id="bbox_0",
                sample_id=0,  # Only sample 0 has bboxes
                etype="bbox",
                load_mechanism=LoadMechanism({"coords": [0, 0, 10, 10]}, category="obj"),
            )
        ]
        # Should NOT raise - images without bboxes are allowed
        ds = DetectionDataset.from_dict({"image": images, "bbox": bboxes})
        assert len(ds) == 2
        assert len(ds.image) == 2
        assert len(ds.bbox) == 1
