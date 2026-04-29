"""Tests for Element.role and the role-as-default-etype convention."""

from __future__ import annotations

import pytest

from bridge.primitives.dataset import Dataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import Sample
from bridge.utils.constants import ELEMENT_COLS
from bridge.utils.data_objects import ClassLabel


def _make_element(
    etype: str = "image",
    role: str | None = None,
    element_id: str = "el_0",
    sample_id: str = "s_0",
) -> Element:
    return Element(
        element_id=element_id,
        sample_id=sample_id,
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
    rebuilt = Element.from_dict(d, load_mechanism=elem._load_mechanism)
    assert rebuilt.role == "target"
    assert rebuilt.etype == "image"


def test_from_dict_tolerates_missing_role_key():
    """Backward compat: dicts written before the role column existed should still load."""
    elem = _make_element(etype="text")
    d = elem.to_dict()
    d.pop(ELEMENT_COLS.ROLE)  # simulate old data
    rebuilt = Element.from_dict(d, load_mechanism=elem._load_mechanism)
    assert rebuilt.role == "text"  # falls back to etype
    assert rebuilt.etype == "text"


def test_sample_groups_elements_by_role_when_roles_distinct():
    """Two elements with the same etype but different roles go into different slots."""
    ref = _make_element(role="reference", element_id="e0", sample_id="s0")
    tgt = _make_element(role="target", element_id="e1", sample_id="s0")
    sample = Sample(elements=[ref, tgt])
    assert set(sample.elements.keys()) == {"reference", "target"}
    assert sample.elements["reference"] == [ref]
    assert sample.elements["target"] == [tgt]


def test_sample_grouping_is_unchanged_when_roles_default():
    """Two elements with the same etype and no explicit role group together (role==etype)."""
    a = _make_element(element_id="e0", sample_id="s0")
    b = _make_element(element_id="e1", sample_id="s0")
    sample = Sample(elements=[a, b])
    assert list(sample.elements.keys()) == ["image"]
    assert sample.elements["image"] == [a, b]


def test_by_etype_collects_across_roles():
    ref = _make_element(etype="image", role="reference", element_id="e0", sample_id="s0")
    tgt = _make_element(etype="image", role="target", element_id="e1", sample_id="s0")
    label = _make_element(etype="class_label", role="label", element_id="lbl", sample_id="s0")
    sample = Sample(elements=[ref, tgt, label])

    images = sample.by_etype("image")
    assert {(role, e.id) for role, e in images} == {("reference", "e0"), ("target", "e1")}


def test_by_etype_empty_when_etype_absent():
    ref = _make_element(etype="image", role="reference", element_id="e0", sample_id="s0")
    sample = Sample(elements=[ref])
    assert sample.by_etype("bbox") == []


def test_by_etype_works_when_role_equals_etype():
    """Old-style elements (role==etype default) still find via by_etype."""
    a = _make_element(etype="image", element_id="e0", sample_id="s0")
    sample = Sample(elements=[a])
    images = sample.by_etype("image")
    assert len(images) == 1
    role, elem = images[0]
    assert role == "image"
    assert elem.id == "e0"


def test_one_returns_single_element():
    ref = _make_element(etype="image", role="reference", element_id="e0", sample_id="s0")
    sample = Sample(elements=[ref])
    elem = sample.one("reference")
    assert elem is ref


def test_one_raises_when_role_absent():
    ref = _make_element(etype="image", role="reference", element_id="e0", sample_id="s0")
    sample = Sample(elements=[ref])
    with pytest.raises(ValueError, match="role='target'"):
        sample.one("target")


def test_one_raises_when_role_has_multiple_elements():
    a = _make_element(etype="image", role="image", element_id="e0", sample_id="s0")
    b = _make_element(etype="image", role="image", element_id="e1", sample_id="s0")
    sample = Sample(elements=[a, b])
    with pytest.raises(ValueError, match="exactly one"):
        sample.one("image")


def test_from_role_dict_assigns_roles_from_keys():
    ref0 = _make_element(etype="image", element_id="e0", sample_id=0)
    tgt0 = _make_element(etype="image", element_id="e1", sample_id=0)
    ref1 = _make_element(etype="image", element_id="e2", sample_id=1)
    tgt1 = _make_element(etype="image", element_id="e3", sample_id=1)
    ds = Dataset.from_role_dict({
        "reference": [ref0, ref1],
        "target": [tgt0, tgt1],
    })

    sample = ds.iget(0)
    assert set(sample.elements.keys()) == {"reference", "target"}
    assert sample.one("reference").id == "e0"
    assert sample.one("target").id == "e1"


def test_from_role_dict_overrides_prior_role():
    """If an Element already has a role set, the dict key wins."""
    elem = _make_element(etype="image", role="ignored", element_id="e0", sample_id=0)
    ds = Dataset.from_role_dict({"reference": [elem]})
    assert ds.iget(0).one("reference").id == "e0"


def test_from_role_dict_empty_value_lists_are_ok():
    ref = _make_element(etype="image", element_id="e0", sample_id=0)
    ds = Dataset.from_role_dict({"reference": [ref], "target": []})
    assert "reference" in ds.iget(0).elements
    assert "target" not in ds.iget(0).elements  # no element so no key


def test_from_role_dict_does_not_mutate_input_elements():
    elem = _make_element(etype="image", role="original_role", element_id="e0", sample_id=0)
    Dataset.from_role_dict({"reference": [elem]})
    # Caller's element should still have its original role
    assert elem.role == "original_role"
