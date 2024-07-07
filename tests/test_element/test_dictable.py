import numpy as np
import pytest

from bridge.display import DisplayEngine
from bridge.primitives.element.data.cache.cache_mechanism import CacheMechanism
from bridge.primitives.element.data.category import DataCategory
from bridge.primitives.element.data.load.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.element.element_type import ElementType
from bridge.utils.constants import ELEMENT_COLS
from bridge.utils.data_objects import BoundingBox, ClassLabel, Keypoint


@pytest.fixture(
    params=[
        {
            ELEMENT_COLS.LOAD_MECHANISM.CATEGORY: DataCategory.obj,
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: BoundingBox(np.array([0, 0, 1, 1]), class_label=ClassLabel(0)),
        },
        {
            ELEMENT_COLS.LOAD_MECHANISM.CATEGORY: DataCategory.obj,
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: ClassLabel(class_idx=0, class_name="some_class"),
        },
        {
            ELEMENT_COLS.LOAD_MECHANISM.CATEGORY: DataCategory.image,
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: "http://example.com/image.jpg",
        },
        {
            ELEMENT_COLS.LOAD_MECHANISM.CATEGORY: DataCategory.image,
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: "dummy_path.jpg",
        },
        {
            ELEMENT_COLS.LOAD_MECHANISM.CATEGORY: DataCategory.obj,
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: Keypoint(np.array([0, 0])),
        },
    ],
    ids=[
        f"{DataCategory.obj}_memory_bbox",
        f"{DataCategory.obj}_memory_class_label",
        f"{DataCategory.image}_http",
        f"{DataCategory.image}_file",
        f"{DataCategory.obj}_memory_keypoint",
    ],
)
def load_mechanism_dict(request):
    return request.param


@pytest.fixture(
    params=[
        {
            ELEMENT_COLS.LOAD_MECHANISM.CATEGORY: "dummy",
            "lol": "dummy",
        },
        {
            "lol": "dummy",
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: "dummy",
        },
    ],
    ids=["no_url_or_data", "no_category"],
)
def bad_load_mechanism_dict(request):
    return request.param


@pytest.fixture
def element_dict(load_mechanism_dict):
    return {
        ELEMENT_COLS.ID: "123",
        ELEMENT_COLS.ETYPE: ElementType.image,
        ELEMENT_COLS.SAMPLE_ID: 0,
        **load_mechanism_dict,
    }


def test_load_mechanism_from_dict(load_mechanism_dict):
    lm = LoadMechanism.from_dict(load_mechanism_dict)
    assert load_mechanism_dict == lm.to_dict()


def test_bad_load_mechanism_dict(bad_load_mechanism_dict):
    with pytest.raises(AssertionError):
        LoadMechanism.from_dict(bad_load_mechanism_dict)


def test_element_from_dict(mocker, element_dict):
    de_mock = mocker.Mock(spec=DisplayEngine)
    cm_mock = mocker.Mock(spec=CacheMechanism)
    element = Element.from_dict(element_dict, display_engine=de_mock, cache_mechanism=cm_mock)
    dic = element.to_dict()
    assert dic == element_dict
