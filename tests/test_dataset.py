from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bridge.primitives.dataset import Dataset
from bridge.primitives.element.data.cache.cache_mechanism import CacheMechanism
from bridge.primitives.element.data.category import DataCategory
from bridge.primitives.element.data.load.load_mechanism import LoadMechanism
from bridge.primitives.element.data.uri_components import URIComponents
from bridge.primitives.element.element import Element
from bridge.primitives.element.element_type import ElementType
from bridge.utils.constants import ELEMENT_COLS
from bridge.utils.data_objects import ClassLabel


@pytest.fixture
def dummy_elements():
    """
    sample ids: 0-100
    element ids: 0-100
    """
    elements = []
    for i in range(100):
        img_element = Element(
            element_id=i,
            sample_id=i,
            etype=ElementType.image,
            load_mechanism=LoadMechanism(
                url_or_data=Image.fromarray(np.random.randint(0, 255, size=(100, 100, 3)).astype("uint8")),
                category=DataCategory.image,
            ),
        )
        lbl_element = Element(
            element_id=f"label_{i}",
            sample_id=i,
            etype=ElementType.class_label,
            load_mechanism=LoadMechanism(
                url_or_data=ClassLabel(class_idx=np.random.randint(0, 10)), category=DataCategory.obj
            ),
        )
        elements.extend([img_element, lbl_element])
    return elements


@pytest.fixture
def dummy_elements_2():
    """
    sample ids: 50-150 (50 sample id overlap)
    element ids: 100-200 (0 element id overlap)
    """
    elements = []
    for i in range(100):
        img_element = Element(
            element_id=100 + i,
            sample_id=50 + i,
            etype=ElementType.image,
            load_mechanism=LoadMechanism(
                url_or_data=Image.fromarray(np.random.randint(0, 255, size=(100, 100, 3)).astype("uint8")),
                category=DataCategory.image,
            ),
        )
        lbl_element = Element(
            element_id=f"label_{100+i}",
            sample_id=50 + i,  # Adjusting sample_id to create overlap
            etype=ElementType.class_label,
            load_mechanism=LoadMechanism(
                url_or_data=ClassLabel(class_idx=np.random.randint(0, 10)), category=DataCategory.obj
            ),
        )
        elements.extend([img_element, lbl_element])
    return elements


@pytest.fixture
def dummy_dataset(dummy_elements):
    return Dataset.from_elements(dummy_elements)


@pytest.fixture
def dummy_dataset_2(dummy_elements_2):
    return Dataset.from_elements(dummy_elements_2)


@pytest.fixture
def dummy_classification_dataset_local_cache(tmp_path, dummy_elements):
    cache = CacheMechanism(URIComponents.from_str(str(tmp_path)))
    return Dataset.from_elements(elements=dummy_elements, cache_mechanisms={ElementType.image: cache})


def test_repr(dummy_dataset):
    lens_dict = {"n_samples": 100, "n_class_label": 100, "n_image": 100}
    assert "Dataset: " + str(lens_dict) == repr(dummy_dataset)


def test_select_by_etype(dummy_dataset):
    ds = dummy_dataset.select(lambda e: e[ELEMENT_COLS.ETYPE] == ElementType.class_label)
    assert len(ds) == 100
    assert ElementType.image not in ds.elements[ELEMENT_COLS.ETYPE].drop_duplicates().to_list()


def test_select_by_sample_id(dummy_dataset):
    sample_ids = np.random.choice(
        dummy_dataset.elements.index.get_level_values(ELEMENT_COLS.SAMPLE_ID).drop_duplicates(),
        len(dummy_dataset) // 2,
        replace=False,
    )

    ds = dummy_dataset.select(lambda e: sample_ids)
    assert len(ds) == len(dummy_dataset) // 2


def test_data(dummy_dataset):
    images = []
    labels = []
    for sample in dummy_dataset:
        data_dict = sample.data
        images.extend(data_dict[ElementType.image])
        labels.extend(data_dict[ElementType.class_label])
    assert all([isinstance(img, np.ndarray) for img in images])
    assert all([img.shape == (100, 100, 3) for img in images])
    assert all([isinstance(label, ClassLabel) for label in labels])


def test_cache(dummy_classification_dataset_local_cache):
    cache_path = str(dummy_classification_dataset_local_cache._cache_mechanisms[ElementType.image]._root_uri)

    # assert starting conditions: cache dir exists and is empty, all elements have object data rather than URI
    assert Path(cache_path).exists() and Path(cache_path).is_dir()
    assert len(list(Path(cache_path).iterdir())) == 0
    assert dummy_classification_dataset_local_cache.elements.data.apply(
        lambda d: not isinstance(d, URIComponents)
    ).all()

    # `sample.data` will call the cache mechanism for every element in the dataset, if exists
    for sample in dummy_classification_dataset_local_cache:
        _ = sample.data

    image_schemes = (
        dummy_classification_dataset_local_cache.elements.pipe(
            lambda df_: df_.loc[df_[ELEMENT_COLS.ETYPE] == ElementType.image]
        )
        .data.apply(lambda d: d.scheme)
        .drop_duplicates()
        .values
    )
    assert len(list(Path(cache_path).iterdir())) == len(dummy_classification_dataset_local_cache)
    assert len(image_schemes) == 1 and image_schemes[0] == ""


def test_merge_duplicate_elements(dummy_dataset):
    with pytest.raises(AssertionError):
        _ = dummy_dataset.merge(dummy_dataset)


def test_merge_datasets(dummy_dataset, dummy_dataset_2):
    merged_ds = dummy_dataset.merge(dummy_dataset_2)
    assert len(merged_ds) == 150
    assert len(merged_ds.elements) == 400
