"""Regression test: Dataset.merge combines stores from both sides.

The pre-existing `test_merge_datasets` checks row count but not that the
merged store contains both sides' load mechanisms. After ElementStore
landed (sub-branch 3), this is the load-bearing invariant: a merged
Dataset must be able to resolve every element id from either side
through `_join_locations` (which reads the store).
"""
from __future__ import annotations

from bridge.primitives.dataset import Dataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.utils.constants import ELEMENT_COLS


def _make_pickle_element(eid: str, sample_id, data) -> Element:
    return Element(
        element_id=eid,
        sample_id=sample_id,
        etype="generic",
        load_mechanism=LoadMechanism(data, encoding="pickle"),
    )


def test_merge_combines_stores():
    left_elems = [
        _make_pickle_element("a", 0, "left-data-a"),
        _make_pickle_element("b", 1, "left-data-b"),
    ]
    right_elems = [
        _make_pickle_element("c", 2, "right-data-c"),
        _make_pickle_element("d", 3, "right-data-d"),
    ]

    left_ds = Dataset.from_role_dict({"generic": left_elems})
    right_ds = Dataset.from_role_dict({"generic": right_elems})

    merged = left_ds.merge(right_ds)

    assert len(merged) == 4
    # Every element id from both sides must resolve in the merged store.
    for eid in ("a", "b", "c", "d"):
        lm = merged._store.get(eid)
        assert lm is not None
        assert lm.encoding == "pickle"

    # And the user-facing elements view should expose all four element ids.
    user_facing = merged.elements
    eids_in_view = set(user_facing.index.get_level_values(ELEMENT_COLS.ID))
    assert eids_in_view == {"a", "b", "c", "d"}


def test_merge_preserves_url_or_data_per_side():
    """The merged store must preserve each side's url_or_data exactly —
    not pick one side's data for both halves.
    """
    left_elems = [_make_pickle_element("a", 0, "left-data-a")]
    right_elems = [_make_pickle_element("c", 2, "right-data-c")]

    left_ds = Dataset.from_role_dict({"generic": left_elems})
    right_ds = Dataset.from_role_dict({"generic": right_elems})

    merged = left_ds.merge(right_ds)

    assert merged._store.get("a").url_or_data == "left-data-a"
    assert merged._store.get("c").url_or_data == "right-data-c"
