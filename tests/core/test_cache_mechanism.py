from unittest.mock import Mock

import pandas as pd
import pytest

from bridge.primitives.element.data.cache_mechanism import CacheMechanism
from bridge.primitives.element.data.element_store import ElementStore
from bridge.primitives.element.data.uri_components import URIComponents


@pytest.fixture
def root_uri():
    return URIComponents(path="/root/path")


@pytest.fixture
def cache_mechanism(root_uri):
    return CacheMechanism(root_uri)


@pytest.fixture
def mock_is_registered(mocker):
    mock_is_registered = mocker.patch("bridge.primitives.element.data.encoding_registry.is_registered")
    mock_is_registered.return_value = True
    return mock_is_registered


@pytest.fixture
def mock_store(mocker):
    mock_store = mocker.patch("bridge.primitives.element.data.encoding_registry.store")
    mock_store.return_value = "Mocked stored LoadMechanism"
    return mock_store


@pytest.fixture
def mock_extension(mocker):
    mock_extension = mocker.patch("bridge.primitives.element.data.encoding_registry.extension")
    mock_extension.return_value = ".ext"
    return mock_extension


@pytest.fixture
def mock_element():
    element = Mock()
    element.id = "test_id"
    element.encoding = "test_category"
    return element


@pytest.fixture
def mock_data():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


def test_init(root_uri):
    cm = CacheMechanism(root_uri)
    assert cm._root_uri == root_uri
    assert cm._store is None


def test_bind_store(cache_mechanism):
    store = ElementStore()
    cache_mechanism.bind_store(store)
    assert cache_mechanism._store is store


def test_store_without_bound_store(
    mock_is_registered,
    mock_store,
    mock_extension,
    cache_mechanism,
    mock_element,
    mock_data,
):
    """When no store is bound, store() returns the new provider but doesn't update anything."""
    result = cache_mechanism.store(mock_element, mock_data)

    expected_uri = URIComponents(path="/root/path/test_id.ext")
    mock_store.assert_called_once_with(mock_data, expected_uri, "test_category")
    assert result == mock_store.return_value


def test_store_updates_bound_store(
    mock_is_registered,
    mock_store,
    mock_extension,
    cache_mechanism,
    mock_element,
    mock_data,
):
    """When a store is bound, store() also writes the new provider into it."""
    new_provider = Mock(name="NewLoadMechanism")
    mock_store.return_value = new_provider

    element_store = ElementStore()
    # seed the store with an existing entry so update() has something to overwrite
    element_store.update("test_id", Mock(name="OldLoadMechanism"))
    cache_mechanism.bind_store(element_store)

    cache_mechanism.store(mock_element, mock_data)

    assert element_store.get("test_id") is new_provider


def test_build_uri(mock_is_registered, mock_store, mock_extension, cache_mechanism, mock_element):
    result = cache_mechanism._build_uri(mock_element, "test_category")
    expected = URIComponents(scheme="", path="/root/path/test_id.ext")
    assert result == expected


def test_build_uri_no_root(mock_element):
    cm = CacheMechanism()
    result = cm._build_uri(mock_element, "test_category")
    assert result is None
