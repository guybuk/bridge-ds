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


def test_element_copy(mocker, load_mechanism_mock, cache_mechanism_mock, display_engine_mock):
    # Setup initial element
    element_id = "test_id"
    etype = "test_type"
    sample_id = "sample_1"
    metadata = {"key1": "value1"}

    element = Element(
        element_id=element_id,
        etype=etype,
        load_mechanism=load_mechanism_mock,
        sample_id=sample_id,
        display_engine=display_engine_mock,
        cache_mechanism=cache_mechanism_mock,
        metadata=metadata,
    )

    # Test 1: Copy with no changes
    copied = element.copy()
    assert copied.id == element_id
    assert copied.etype == etype
    assert copied.sample_id == sample_id
    assert copied._load_mechanism == load_mechanism_mock
    assert copied._display_engine == display_engine_mock
    assert copied.metadata == metadata
    # Verify cache mechanism was detached
    cache_mechanism_mock.detached_copy.assert_called_once()

    # Test 2: Copy with modified attributes
    new_id = "new_id"
    new_etype = "new_type"
    new_load_mechanism = mocker.Mock(spec=LoadMechanism)
    new_metadata = {"key2": "value2"}

    modified = element.copy(
        element_id=new_id, etype=new_etype, load_mechanism=new_load_mechanism, metadata=new_metadata
    )

    assert modified.id == new_id
    assert modified.etype == new_etype
    assert modified._load_mechanism == new_load_mechanism
    assert modified.sample_id == sample_id  # Unchanged
    assert modified._display_engine == display_engine_mock  # Unchanged
    assert modified.metadata == new_metadata

    # Test 3: Verify metadata is deeply copied
    copied = element.copy()
    assert copied.metadata is not element.metadata
    copied.metadata["new_key"] = "new_value"
    assert "new_key" not in element.metadata

    # Test 4: Copy with None cache_mechanism
    element_no_cache = Element(
        element_id=element_id,
        etype=etype,
        load_mechanism=load_mechanism_mock,
        sample_id=sample_id,
        cache_mechanism=None,
    )
    copied_no_cache = element_no_cache.copy()
    assert copied_no_cache._cache_mechanism is None
