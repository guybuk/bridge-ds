import numpy as np
import pytest

from bridge.primitives.dataset import MultiRoleDataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import MultiRoleSample


@pytest.fixture
def dummy_text_elements():
    """Create 100 samples with source/target text elements."""
    source_elements = []
    target_elements = []

    for i in range(100):
        source_elem = Element(
            element_id=f"source_{i}",
            sample_id=i,
            etype="text",
            load_mechanism=LoadMechanism(f"Source text {i}", category="obj"),
        )
        target_elem = Element(
            element_id=f"target_{i}",
            sample_id=i,
            etype="text",
            load_mechanism=LoadMechanism(f"Target text {i}", category="obj"),
        )
        source_elements.append(source_elem)
        target_elements.append(target_elem)

    return source_elements, target_elements


@pytest.fixture
def dummy_multi_role_dataset(dummy_text_elements):
    source_list, target_list = dummy_text_elements
    return MultiRoleDataset.from_dict(
        {"source": source_list, "target": target_list},
    )


@pytest.fixture
def dummy_image_elements():
    """Create 10 samples with input/output image elements."""
    input_elements = []
    output_elements = []

    for i in range(10):
        input_elem = Element(
            element_id=f"input_{i}",
            sample_id=i,
            etype="image",
            load_mechanism=LoadMechanism(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                category="obj",
            ),
        )
        output_elem = Element(
            element_id=f"output_{i}",
            sample_id=i,
            etype="image",
            load_mechanism=LoadMechanism(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                category="obj",
            ),
        )
        input_elements.append(input_elem)
        output_elements.append(output_elem)

    return input_elements, output_elements


@pytest.fixture
def dummy_image_dataset(dummy_image_elements):
    input_list, output_list = dummy_image_elements
    return MultiRoleDataset.from_dict(
        {"input": input_list, "output": output_list},
    )


@pytest.fixture
def dummy_detection_elements():
    """Create 10 samples with image (1 per sample) and bbox (multiple per sample) elements."""
    image_elements = []
    bbox_elements = []

    for i in range(10):
        # One image per sample
        image_elem = Element(
            element_id=f"img_{i}",
            sample_id=i,
            etype="image",
            load_mechanism=LoadMechanism(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                category="obj",
            ),
        )
        image_elements.append(image_elem)

        # Multiple bboxes per sample (2-5)
        num_bboxes = np.random.randint(2, 6)
        for j in range(num_bboxes):
            bbox_elem = Element(
                element_id=f"bbox_{i}_{j}",
                sample_id=i,
                etype="bbox",
                load_mechanism=LoadMechanism(
                    {"x": 10, "y": 10, "w": 20, "h": 20},
                    category="obj",
                ),
            )
            bbox_elements.append(bbox_elem)

    return image_elements, bbox_elements


@pytest.fixture
def dummy_detection_dataset(dummy_detection_elements):
    image_list, bbox_list = dummy_detection_elements
    return MultiRoleDataset.from_dict(
        {"image": image_list, "bbox": bbox_list},
    )


class TestMultiRoleDatasetCreation:
    def test_creation_from_dict(self, dummy_multi_role_dataset):
        assert len(dummy_multi_role_dataset) == 100

    def test_role_names(self, dummy_multi_role_dataset):
        assert dummy_multi_role_dataset.role_names == ("source", "target")

    def test_get_role(self, dummy_multi_role_dataset):
        source = dummy_multi_role_dataset.get_role("source")
        target = dummy_multi_role_dataset.get_role("target")
        assert len(source) == 100
        assert len(target) == 100

    def test_dynamic_attribute_access(self, dummy_multi_role_dataset):
        source = dummy_multi_role_dataset.source
        target = dummy_multi_role_dataset.target
        assert len(source) == 100
        assert len(target) == 100

    def test_repr(self, dummy_multi_role_dataset):
        repr_str = repr(dummy_multi_role_dataset)
        assert "MultiRoleDataset" in repr_str
        assert "n_samples" in repr_str
        assert "100" in repr_str


class TestMultiRoleDatasetValidation:
    def test_single_role_raises(self, dummy_text_elements):
        source_list, _ = dummy_text_elements
        with pytest.raises(AssertionError, match="Must have at least 2 roles"):
            MultiRoleDataset.from_dict({"source": source_list})

    def test_duplicate_element_ids_raises(self):
        first = [
            Element(
                element_id="same_id",
                sample_id=0,
                etype="text",
                load_mechanism=LoadMechanism("text", category="obj"),
            )
        ]
        second = [
            Element(
                element_id="same_id",  # Same element_id
                sample_id=0,
                etype="text",
                load_mechanism=LoadMechanism("text", category="obj"),
            )
        ]
        with pytest.raises(AssertionError, match="Element IDs must be unique"):
            MultiRoleDataset.from_dict({"first": first, "second": second})


class TestMultiRoleDatasetAccess:
    def test_iget_returns_multi_role_sample(self, dummy_multi_role_dataset):
        sample = dummy_multi_role_dataset.iget(0)
        assert isinstance(sample, MultiRoleSample)

    def test_get_returns_multi_role_sample(self, dummy_multi_role_dataset):
        sample = dummy_multi_role_dataset.get(0)
        assert isinstance(sample, MultiRoleSample)

    def test_iteration(self, dummy_multi_role_dataset):
        count = 0
        for sample in dummy_multi_role_dataset:
            assert isinstance(sample, MultiRoleSample)
            count += 1
        assert count == 100

    def test_indexing(self, dummy_multi_role_dataset):
        sample = dummy_multi_role_dataset[0]
        assert isinstance(sample, MultiRoleSample)


class TestMultiRoleDatasetOperations:
    def test_select_by_role(self, dummy_multi_role_dataset):
        ds = dummy_multi_role_dataset.select_by_role("source", lambda source, target: source.index[:50])
        assert len(ds) == 50
        assert isinstance(ds, MultiRoleDataset)
        assert ds.role_names == ("source", "target")

    def test_assign_to_role(self, dummy_multi_role_dataset):
        ds = dummy_multi_role_dataset.assign_to_role("source", new_col=lambda source, target: range(len(source)))
        assert "new_col" in ds.get_role("source").columns
        assert "new_col" not in ds.get_role("target").columns

    def test_sort_by_role(self, dummy_multi_role_dataset):
        ds = dummy_multi_role_dataset.assign_to_role(
            "source", sort_key=lambda source, target: list(reversed(range(len(source))))
        )
        sorted_ds = ds.sort_by_role("source", by="sort_key", ascending=True)
        assert isinstance(sorted_ds, MultiRoleDataset)


class TestMultiRoleSample:
    def test_get_role(self, dummy_multi_role_dataset):
        sample = dummy_multi_role_dataset.iget(0)
        source_elements = sample.get_role("source")
        assert len(source_elements) == 1
        assert source_elements[0].etype == "text"

    def test_get_role_data(self, dummy_multi_role_dataset):
        sample = dummy_multi_role_dataset.iget(0)
        source_data = sample.get_role_data("source")
        assert source_data == ["Source text 0"]

    def test_dynamic_attribute_access(self, dummy_multi_role_dataset):
        sample = dummy_multi_role_dataset.iget(0)
        assert sample.source == sample.get_role("source")
        assert sample.target == sample.get_role("target")

    def test_data_property(self, dummy_multi_role_dataset):
        sample = dummy_multi_role_dataset.iget(0)
        data = sample.data
        assert "source" in data
        assert "target" in data
        assert data["source"] == ["Source text 0"]
        assert data["target"] == ["Target text 0"]

    def test_role_names_property(self, dummy_multi_role_dataset):
        sample = dummy_multi_role_dataset.iget(0)
        assert sample.role_names == ("source", "target")


class TestMultiRoleDatasetImagePairs:
    def test_image_data_access(self, dummy_image_dataset):
        sample = dummy_image_dataset.iget(0)
        input_data = sample.get_role_data("input")
        output_data = sample.get_role_data("output")
        assert input_data[0].shape == (64, 64, 3)
        assert output_data[0].shape == (64, 64, 3)

    def test_image_role_names(self, dummy_image_dataset):
        assert dummy_image_dataset.role_names == ("input", "output")
        sample = dummy_image_dataset.iget(0)
        assert sample.input == sample.get_role("input")
        assert sample.output == sample.get_role("output")


class TestMultiRoleDatasetMultipleElementsPerRole:
    """Test that MultiRoleDataset correctly handles multiple elements per role (e.g., detection datasets)."""

    def test_detection_dataset_creation(self, dummy_detection_dataset):
        assert len(dummy_detection_dataset) == 10

    def test_detection_sample_has_one_image(self, dummy_detection_dataset):
        sample = dummy_detection_dataset.iget(0)
        images = sample.get_role("image")
        assert len(images) == 1

    def test_detection_sample_has_multiple_bboxes(self, dummy_detection_dataset):
        sample = dummy_detection_dataset.iget(0)
        bboxes = sample.get_role("bbox")
        assert len(bboxes) >= 2  # We created 2-5 bboxes per sample

    def test_detection_dynamic_access(self, dummy_detection_dataset):
        sample = dummy_detection_dataset.iget(0)
        assert sample.image == sample.get_role("image")
        assert sample.bbox == sample.get_role("bbox")

    def test_detection_data_property(self, dummy_detection_dataset):
        sample = dummy_detection_dataset.iget(0)
        data = sample.data
        assert "image" in data
        assert "bbox" in data
        assert len(data["image"]) == 1
        assert len(data["bbox"]) >= 2

    def test_detection_role_counts(self, dummy_detection_dataset):
        repr_str = repr(dummy_detection_dataset)
        assert "n_image" in repr_str
        assert "n_bbox" in repr_str
