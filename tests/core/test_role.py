"""Tests for Element.role and the role-as-default-etype convention."""

from __future__ import annotations

from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.utils.constants import ELEMENT_COLS
from bridge.utils.data_objects import ClassLabel


def _make_element(etype: str = "image", role: str | None = None) -> Element:
    return Element(
        element_id="el_0",
        sample_id="s_0",
        etype=etype,
        role=role,
        load_mechanism=LoadMechanism(ClassLabel(class_idx=0), encoding="pickle"),
    )


def test_role_defaults_to_etype_when_unspecified():
    elem = _make_element(etype="image")
    assert elem.role == "image"


def test_role_uses_explicit_value_when_provided():
    elem = _make_element(etype="image", role="reference")
    assert elem.role == "reference"
    assert elem.etype == "image"  # etype unchanged


def test_role_none_explicit_falls_back_to_etype():
    elem = _make_element(etype="text", role=None)
    assert elem.role == "text"


def test_role_empty_string_is_preserved():
    """Empty string is a valid (if unusual) role and must not trigger the etype fallback.

    The fallback is keyed on `is None`, not bare truthiness — this test guards that.
    """
    elem = _make_element(etype="image", role="")
    assert elem.role == ""


def test_role_appears_in_to_dict():
    elem = _make_element(etype="image", role="reference")
    d = elem.to_dict()
    assert d[ELEMENT_COLS.ROLE] == "reference"


def test_to_dict_round_trip_preserves_role():
    elem = _make_element(etype="image", role="target")
    d = elem.to_dict()
    rebuilt = Element.from_dict(d)
    assert rebuilt.role == "target"
    assert rebuilt.etype == "image"


def test_from_dict_tolerates_missing_role_key():
    """Backward compat: dicts written before the role column existed should still load."""
    elem = _make_element(etype="text")
    d = elem.to_dict()
    d.pop(ELEMENT_COLS.ROLE)  # simulate old data
    rebuilt = Element.from_dict(d)
    assert rebuilt.role == "text"  # falls back to etype
    assert rebuilt.etype == "text"
