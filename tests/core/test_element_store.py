"""Tests for the ElementStore — the shared per-lineage location store."""

from __future__ import annotations

import pytest

from bridge.primitives.element.data.element_store import ElementStore
from bridge.primitives.element.data.load_mechanism import LoadMechanism


def _lm(value="x", encoding="pickle") -> LoadMechanism:
    return LoadMechanism(value, encoding=encoding)


def test_empty_store_has_zero_length():
    store = ElementStore()
    assert len(store) == 0


def test_set_and_get():
    store = ElementStore()
    lm = _lm("hello")
    store.set("e0", lm)
    assert store.get("e0") is lm


def test_contains():
    store = ElementStore()
    store.set("e0", _lm())
    assert "e0" in store
    assert "missing" not in store


def test_get_missing_raises_keyerror():
    store = ElementStore()
    with pytest.raises(KeyError):
        store.get("missing")


def test_set_existing_key_raises():
    store = ElementStore()
    store.set("e0", _lm("v1"))
    with pytest.raises(ValueError, match="already has"):
        store.set("e0", _lm("v2"))


def test_update_overwrites():
    store = ElementStore()
    store.set("e0", _lm("v1"))
    store.update("e0", _lm("v2"))
    assert store.get("e0").url_or_data == "v2"


def test_update_creates_if_absent():
    """update() is set-or-overwrite; no error for a missing key."""
    store = ElementStore()
    store.update("e0", _lm("v"))
    assert store.get("e0").url_or_data == "v"


def test_extend_merges_two_disjoint_stores():
    a = ElementStore()
    a.set("e0", _lm("a"))
    b = ElementStore()
    b.set("e1", _lm("b"))
    a.extend(b)
    assert len(a) == 2
    assert a.get("e0").url_or_data == "a"
    assert a.get("e1").url_or_data == "b"


def test_extend_overlap_raises():
    a = ElementStore()
    a.set("e0", _lm("a"))
    b = ElementStore()
    b.set("e0", _lm("b"))
    with pytest.raises(ValueError, match="overlap"):
        a.extend(b)


def test_iteration_returns_element_ids():
    store = ElementStore()
    store.set("e0", _lm())
    store.set("e1", _lm())
    assert set(store) == {"e0", "e1"}
