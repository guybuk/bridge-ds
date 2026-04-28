import numpy as np
import pytest

from bridge.display import DisplayEngine
from bridge.primitives.element.data.cache_mechanism import CacheMechanism
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.utils.constants import ELEMENT_COLS
from bridge.utils.data_objects import BoundingBox, ClassLabel, Keypoint


@pytest.fixture(
    params=[
        {
            ELEMENT_COLS.LOAD_MECHANISM.ENCODING: "pickle",
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: BoundingBox(np.array([0, 0, 1, 1]), class_label=ClassLabel(0)),
        },
        {
            ELEMENT_COLS.LOAD_MECHANISM.ENCODING: "pickle",
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: ClassLabel(class_idx=0, class_name="some_class"),
        },
        {
            ELEMENT_COLS.LOAD_MECHANISM.ENCODING: "jpeg",
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: "http://example.com/image.jpg",
        },
        {
            ELEMENT_COLS.LOAD_MECHANISM.ENCODING: "jpeg",
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: "dummy_path.jpg",
        },
        {
            ELEMENT_COLS.LOAD_MECHANISM.ENCODING: "pickle",
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: Keypoint(np.array([0, 0])),
        },
    ],
    ids=[
        "pickle_memory_bbox",
        "pickle_memory_class_label",
        "jpeg_http",
        "jpeg_file",
        "pickle_memory_keypoint",
    ],
)
def load_mechanism_dict(request):
    return request.param


@pytest.fixture(
    params=[
        {
            ELEMENT_COLS.LOAD_MECHANISM.ENCODING: "dummy",
            "lol": "dummy",
        },
        {
            "lol": "dummy",
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: "dummy",
        },
    ],
    ids=["no_url_or_data", "no_encoding"],
)
def bad_load_mechanism_dict(request):
    return request.param


@pytest.fixture
def element_dict():
    return {
        ELEMENT_COLS.ID: "123",
        ELEMENT_COLS.ETYPE: "image",
        ELEMENT_COLS.ROLE: "image",
        ELEMENT_COLS.SAMPLE_ID: 0,
    }


def test_load_mechanism_from_dict(load_mechanism_dict):
    lm = LoadMechanism.from_dict(load_mechanism_dict)
    assert load_mechanism_dict == lm.to_dict()


def test_bad_load_mechanism_dict(bad_load_mechanism_dict):
    with pytest.raises(AssertionError):
        LoadMechanism.from_dict(bad_load_mechanism_dict)


def test_element_from_dict(mocker, element_dict, load_mechanism_dict):
    de_mock = mocker.Mock(spec=DisplayEngine)
    cm_mock = mocker.Mock(spec=CacheMechanism)
    load_mechanism = LoadMechanism.from_dict(load_mechanism_dict)
    element = Element.from_dict(
        element_dict, load_mechanism=load_mechanism, display_engine=de_mock, cache_mechanism=cm_mock
    )
    dic = element.to_dict()
    assert dic == element_dict
