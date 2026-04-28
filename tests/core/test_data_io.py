"""Round-trip tests for the built-in DataIO implementations.

These exercise actual store/load round-trips against tmp_path. Previously the
suite covered the registry layer but never the concrete IO classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from bridge.primitives.element.data.data_io import ObjDataIO, TextDataIO
from bridge.primitives.element.data.uri_components import URIComponents


@dataclass
class _Sentinel:
    """A non-trivial picklable object for round-trip checks."""

    name: str
    values: list[int]


def _file_uri(path) -> URIComponents:
    return URIComponents(scheme="", path=str(path))


def test_text_data_io_store_then_load_roundtrip(tmp_path):
    payload = "hello bridge\nmultiline content"
    target = tmp_path / "subdir" / "doc.txt"

    lm = TextDataIO.store(payload, _file_uri(target))

    assert target.exists()
    assert target.read_text() == payload
    assert lm.encoding == "text"

    # The returned LoadMechanism should resolve back to the same content.
    reloaded = TextDataIO.load(lm.url_or_data)
    assert reloaded == payload


def test_text_data_io_store_inline_when_url_is_none():
    payload = "inline text"
    lm = TextDataIO.store(payload, None)
    assert lm.url_or_data == payload
    assert lm.encoding == "text"


def test_obj_data_io_roundtrip(tmp_path):
    """The bug this fixes: ObjDataIO.store wrote pickle files, but
    ObjDataIO.load raised NotImplementedError on the URI path. Caching
    a Python object to disk and re-reading it is now a complete
    round-trip."""
    sentinel = _Sentinel(name="bbox", values=[1, 2, 3])
    target = tmp_path / "blob" / "obj.pkl"

    lm = ObjDataIO.store(sentinel, _file_uri(target))

    assert target.exists()
    assert lm.encoding == "obj"

    reloaded = ObjDataIO.load(lm.url_or_data)
    assert isinstance(reloaded, _Sentinel)
    assert reloaded.name == "bbox"
    assert reloaded.values == [1, 2, 3]


def test_obj_data_io_load_inline_passthrough():
    sentinel = _Sentinel(name="x", values=[])
    assert ObjDataIO.load(sentinel) is sentinel


def test_obj_data_io_load_rejects_non_local_scheme():
    with pytest.raises(NotImplementedError):
        ObjDataIO.load(URIComponents(scheme="https", netloc="example.com", path="/x"))


def test_text_data_io_store_rejects_non_local_scheme():
    with pytest.raises(NotImplementedError):
        TextDataIO.store("data", URIComponents(scheme="https", netloc="example.com", path="/x"))


def test_numpy_data_io_no_longer_registered():
    """NumpyDataIO was dead code; deleting it removes 'numpy' from the registry."""
    from bridge.primitives.element.data import encoding_registry

    assert "numpy" not in encoding_registry.list_registered_encodings()
