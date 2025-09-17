from unittest.mock import Mock

import pandas as pd
import pytest

from bridge.primitives.element.data.cache_mechanism import CacheMechanism
from bridge.primitives.element.data.uri_components import URIComponents
from bridge.utils.constants import INDICES


@pytest.fixture
def root_uri():
    return URIComponents(path="/root/path")


@pytest.fixture
def cache_mechanism(root_uri):
    return CacheMechanism(root_uri)


@pytest.fixture
def mock_is_registered(mocker):
    mock_is_registered = mocker.patch("bridge.primitives.element.data.category_registry.is_registered")
    mock_is_registered.return_value = True
    return mock_is_registered


@pytest.fixture
def mock_store(mocker):
    mock_store = mocker.patch("bridge.primitives.element.data.category_registry.store")
    mock_store.return_value = "Mocked stored LoadMechanism"
    return mock_store


@pytest.fixture
def mock_extension(mocker):
    mock_extension = mocker.patch("bridge.primitives.element.data.category_registry.extension")
    mock_extension.return_value = ".ext"
    return mock_extension


@pytest.fixture
def mock_element():
    element = Mock()
    element.id = "test_id"
    element.category = "test_category"
    return element


@pytest.fixture
def mock_data():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


def test_init(root_uri):
    cm = CacheMechanism(root_uri)
    assert cm._root_uri == root_uri
    assert cm._elements is None


def test_set_elements_df(cache_mechanism):
    df = pd.DataFrame({"id": [1, 2, 3]})
    cache_mechanism.set_elements_df(df)
    assert cache_mechanism._elements is df


def test_store(
    mock_is_registered,
    mock_store,
    mock_extension,
    cache_mechanism,
    mock_element,
    mock_data,
):
    result = cache_mechanism.store(mock_element, mock_data)

    expected_uri = URIComponents(path="/root/path/test_id.ext")
    mock_store.assert_called_once_with(mock_data, expected_uri, "test_category")
    assert result == mock_store.return_value


def test_store_update_elements(
    mock_is_registered,
    mock_store,
    mock_extension,
    cache_mechanism,
    mock_element,
    mock_data,
):
    mock_store.return_value = Mock(to_dict=lambda: {"key1": "value1", "key2": "value2"})
    mock_extension.return_value = ".ext"

    elements_df = pd.DataFrame(
        {
            "sample_id": ["test_sample_id"],
            "element_id": ["test_id"],
            "key1": ["old1"],
            "key2": ["old2"],
        }
    )
    elements_df.set_index(INDICES, inplace=True)
    cache_mechanism.set_elements_df(elements_df)

    cache_mechanism.store(mock_element, mock_data, should_update_elements=True)

    pd.testing.assert_frame_equal(
        cache_mechanism._elements,
        pd.DataFrame(
            {"key1": ["value1"], "key2": ["value2"]},
            index=pd.MultiIndex.from_tuples([("test_sample_id", "test_id")], names=INDICES),
        ),
    )


def test_build_uri(mock_is_registered, mock_store, mock_extension, cache_mechanism, mock_element):
    result = cache_mechanism._build_uri(mock_element, "test_category")
    expected = URIComponents(scheme="", path="/root/path/test_id.ext")
    assert result == expected


def test_build_uri_no_root(mock_element):
    cm = CacheMechanism()
    result = cm._build_uri(mock_element, "test_category")
    assert result is None
