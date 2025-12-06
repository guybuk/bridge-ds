import pytest

from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.data.uri_components import URIComponents
from bridge.utils.constants import ELEMENT_COLS


@pytest.fixture
def mock_is_registered(mocker):
    mock_is_registered = mocker.patch("bridge.primitives.element.data.category_registry.is_registered")
    mock_is_registered.return_value = True
    return mock_is_registered


@pytest.fixture
def mock_load(mocker):
    mock_load = mocker.patch("bridge.primitives.element.data.category_registry.load")
    mock_load.return_value = "Mocked loaded data"
    return mock_load


def test_load_mechanism_init(mock_is_registered, mock_load):
    url_or_data = URIComponents(scheme="http", netloc="example.com", path="/data")
    lm = LoadMechanism(url_or_data, "test_category")
    assert lm.url_or_data == url_or_data
    assert lm.category == "test_category"


def test_load_mechanism_init_invalid_category(mock_is_registered, mock_load):
    mock_is_registered.return_value = False
    with pytest.raises(AssertionError, match="Category invalid_category is not registered."):
        LoadMechanism(URIComponents(), "invalid_category")


def test_load_data(mock_is_registered, mock_load):
    lm = LoadMechanism(URIComponents(), "test_category")
    assert lm.load_data() == "Mocked loaded data"
    mock_load.assert_called_once_with(lm.url_or_data, "test_category")


def test_to_dict(mock_is_registered, mock_load):
    url_or_data = URIComponents(scheme="http", netloc="example.com", path="/data")
    lm = LoadMechanism(url_or_data, "test_category")
    expected_dict = {
        ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: url_or_data,
        ELEMENT_COLS.LOAD_MECHANISM.CATEGORY: "test_category",
    }
    assert lm.to_dict() == expected_dict


def test_from_dict(mock_is_registered, mock_load):
    url_or_data = URIComponents(scheme="http", netloc="example.com", path="/data")
    input_dict = {
        ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: url_or_data,
        ELEMENT_COLS.LOAD_MECHANISM.CATEGORY: "test_category",
    }
    lm = LoadMechanism.from_dict(input_dict)
    assert lm.url_or_data == url_or_data
    assert lm.category == "test_category"


def test_from_url_string(mock_is_registered, mock_load):
    url_string = "http://example.com/data"
    lm = LoadMechanism.from_url_string(url_string, "test_category")
    assert isinstance(lm.url_or_data, URIComponents)
    assert lm.url_or_data.scheme == "http"
    assert lm.url_or_data.netloc == "example.com"
    assert lm.url_or_data.path == "/data"
    assert lm.category == "test_category"


def test_from_dict_missing_keys():
    input_dict = {ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: URIComponents()}
    with pytest.raises(AssertionError):
        LoadMechanism.from_dict(input_dict)
