import pytest

from bridge.primitives.element.data import category_registry


def test_default_registry():
    assert len(category_registry.list_registered_categories()) > 0


@pytest.fixture
def clear_registry(mocker):
    mocker.patch("bridge.primitives.element.data.category_registry.REGISTRY", {})
    category_registry.REGISTRY.clear()


@pytest.fixture
def mock_io(mocker):
    mock_io = mocker.Mock()
    mock_io.category = "mock"
    mock_io.extension = ".mock"
    mock_io.store.return_value = "stored"
    mock_io.load.return_value = "loaded"
    return mock_io


def test_is_registered(clear_registry, mock_io):
    assert not category_registry.is_registered("mock")
    category_registry.register(mock_io)
    assert category_registry.is_registered("mock")


def test_list_registered_categories(clear_registry, mock_io):
    assert category_registry.list_registered_categories() == []
    category_registry.register(mock_io)
    assert category_registry.list_registered_categories() == ["mock"]


def test_register(clear_registry, mock_io):
    category_registry.register(mock_io)
    assert "mock" in category_registry.list_registered_categories()
    with pytest.raises(ValueError):
        category_registry.register(mock_io)


def test_store(clear_registry, mock_io):
    category_registry.register(mock_io)
    result = category_registry.store("data", None, "mock")
    assert result == "stored"


def test_load(clear_registry, mock_io):
    category_registry.register(mock_io)
    result = category_registry.load("data", "mock")
    assert result == "loaded"


def test_extension(clear_registry, mock_io):
    category_registry.register(mock_io)
    assert category_registry.extension("mock") == ".mock"


def test_unregistered_category():
    with pytest.raises(KeyError):
        category_registry.store("data", None, "unregistered")

    with pytest.raises(KeyError):
        category_registry.load("data", "unregistered")

    with pytest.raises(KeyError):
        category_registry.extension("unregistered")
