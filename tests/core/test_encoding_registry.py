import pytest

from bridge.primitives.element.data import encoding_registry


def test_default_registry():
    assert len(encoding_registry.list_registered_encodings()) > 0


@pytest.fixture
def clear_registry(mocker):
    mocker.patch("bridge.primitives.element.data.encoding_registry.REGISTRY", {})
    encoding_registry.REGISTRY.clear()


@pytest.fixture
def mock_io(mocker):
    mock_io = mocker.Mock()
    mock_io.encoding = "mock"
    mock_io.extension = ".mock"
    mock_io.store.return_value = "stored"
    mock_io.load.return_value = "loaded"
    return mock_io


def test_is_registered(clear_registry, mock_io):
    assert not encoding_registry.is_registered("mock")
    encoding_registry.register(mock_io)
    assert encoding_registry.is_registered("mock")


def test_list_registered_encodings(clear_registry, mock_io):
    assert encoding_registry.list_registered_encodings() == []
    encoding_registry.register(mock_io)
    assert encoding_registry.list_registered_encodings() == ["mock"]


def test_register(clear_registry, mock_io):
    encoding_registry.register(mock_io)
    assert "mock" in encoding_registry.list_registered_encodings()
    with pytest.raises(ValueError):
        encoding_registry.register(mock_io)


def test_store(clear_registry, mock_io):
    encoding_registry.register(mock_io)
    result = encoding_registry.store("data", None, "mock")
    assert result == "stored"


def test_load(clear_registry, mock_io):
    encoding_registry.register(mock_io)
    result = encoding_registry.load("data", "mock")
    assert result == "loaded"


def test_extension(clear_registry, mock_io):
    encoding_registry.register(mock_io)
    assert encoding_registry.extension("mock") == ".mock"


def test_unregistered_encoding():
    with pytest.raises(KeyError):
        encoding_registry.store("data", None, "unregistered")

    with pytest.raises(KeyError):
        encoding_registry.load("data", "unregistered")

    with pytest.raises(KeyError):
        encoding_registry.extension("unregistered")
