import pytest

from bridge.display import DisplayEngine
from bridge.primitives.element.data.cache_mechanism import CacheMechanism
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element


@pytest.fixture
def dummy_element_id():
    return 0


@pytest.fixture
def dummy_sample_id():
    return "0"


@pytest.fixture
def dummy_etype():
    return "segmentation"


@pytest.fixture
def load_mechanism_mock(mocker):
    mock = mocker.Mock(spec=LoadMechanism)
    mock.category.return_value = "some_category"
    mock.load_data.return_value = b"some_data"
    return mock


@pytest.fixture
def display_engine_mock(mocker):
    return mocker.Mock(spec=DisplayEngine)


@pytest.fixture
def cache_mechanism_mock(mocker):
    return mocker.Mock(spec=CacheMechanism)


@pytest.fixture
def dummy_metadata():
    return {"lol": "kek", "foo": 5}


@pytest.fixture
def dummy_element(
    dummy_element_id,
    dummy_sample_id,
    dummy_etype,
    load_mechanism_mock,
    display_engine_mock,
    cache_mechanism_mock,
    dummy_metadata,
):
    return Element(
        element_id=dummy_element_id,
        etype=dummy_etype,
        load_mechanism=load_mechanism_mock,
        sample_id=dummy_sample_id,
        display_engine=display_engine_mock,
        cache_mechanism=cache_mechanism_mock,
        metadata=dummy_metadata,
    )


def test_validate_metadata(load_mechanism_mock, display_engine_mock, cache_mechanism_mock):
    with pytest.raises(AssertionError):
        Element(
            element_id=0,
            sample_id=0,
            etype="segmentation",
            load_mechanism=load_mechanism_mock,
            display_engine=display_engine_mock,
            cache_mechanism=cache_mechanism_mock,
            metadata={"source": "lol", "data": "dsadsa"},
        )


def test_element_static_properties(
    dummy_element, dummy_element_id, dummy_sample_id, dummy_etype, load_mechanism_mock, dummy_metadata
):
    assert dummy_element.id == dummy_element_id
    assert dummy_element.sample_id == dummy_sample_id
    assert dummy_element.etype == dummy_etype
    assert dummy_element.metadata == dummy_metadata
    assert dummy_element.category == load_mechanism_mock.category


def test_element_data(dummy_element, load_mechanism_mock, cache_mechanism_mock):
    assert dummy_element.data == load_mechanism_mock.load_data()
    assert cache_mechanism_mock.store.called


def test_element_copy(load_mechanism_mock):
    # Setup
    element_id = "test_id"
    sample_id = "sample_1"
    etype = "image"
    metadata = {"key": "value"}

    original = Element(
        element_id=element_id, etype=etype, load_mechanism=load_mechanism_mock, sample_id=sample_id, metadata=metadata
    )

    # Test case 1: Copy without changing IDs
    copied = original.copy()
    assert copied.id == original.id
    assert copied.sample_id == original.sample_id
    assert copied.etype == original.etype
    assert copied.metadata == original.metadata
    assert copied._load_mechanism == original._load_mechanism

    # Test case 2: Copy with new element_id
    new_element_id = "new_test_id"
    copied_new_eid = original.copy(new_element_id=new_element_id)
    assert copied_new_eid.id == new_element_id
    assert copied_new_eid.sample_id == original.sample_id

    # Test case 3: Copy with new sample_id
    new_sample_id = "new_sample_id"
    copied_new_sid = original.copy(new_sample_id=new_sample_id)
    assert copied_new_sid.id == original.id
    assert copied_new_sid.sample_id == new_sample_id

    # Test case 4: Copy with both new IDs
    copied_both = original.copy(new_element_id=new_element_id, new_sample_id=new_sample_id)
    assert copied_both.id == new_element_id
    assert copied_both.sample_id == new_sample_id
