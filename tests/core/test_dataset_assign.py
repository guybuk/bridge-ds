"""Pin Dataset.assign(data=...) no-propagate behavior — discovered during notebook migration.

The notebook migration revealed that `Dataset.assign(data=...)` writes a
column to `_df`, but that column is overridden every time `ds.elements`
is accessed (because `_join_locations` synthesizes `data` and `encoding`
from the store). This is the deliberate post-v0.2 design — the load
mechanism layer is the source of truth — but it's surprising for callers
who'd expect `assign` to behave like `pandas.DataFrame.assign`.

Pin the current behavior so future API changes are forced to be
intentional. If a future PR changes `assign` semantics, these tests will
turn red and the change will be reviewed.
"""
from __future__ import annotations

from bridge.primitives.dataset import Dataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.utils.constants import ELEMENT_COLS


def _trivial_pickle_dataset(n: int = 3) -> Dataset:
    elems = [
        Element(
            element_id=f"e_{i}",
            sample_id=i,
            etype="t",
            load_mechanism=LoadMechanism(f"data-{i}", encoding="pickle"),
        )
        for i in range(n)
    ]
    return Dataset.from_role_dict({"t": elems})


def test_assign_metadata_persists_through_elements_view():
    """Sanity check: arbitrary metadata columns set via `assign` survive
    the elements view. (If this test fails, the elements view is doing
    something weirder than just synthesizing the load-mechanism columns.)
    """
    ds = _trivial_pickle_dataset(3)
    ds_with_meta = ds.assign(custom=lambda df: ["a", "b", "c"])
    assert "custom" in ds_with_meta.elements.columns
    assert list(ds_with_meta.elements["custom"]) == ["a", "b", "c"]


def test_assign_to_url_or_data_column_is_ignored_at_view_layer():
    """assign(data=...) writes to _df but is overridden by the store-join
    in `elements`.

    This pins the current behavior: the load-mechanism layer is the
    source of truth for url_or_data / encoding, and `assign` cannot
    override it. If this test starts failing, the API has changed
    (intentionally or otherwise) and the change should be reviewed.
    """
    ds = _trivial_pickle_dataset(3)
    url_col = ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA
    ds_with_overwrite = ds.assign(**{url_col: lambda df: ["overwritten"] * 3})
    # Even though we wrote to _df, the elements view re-synthesizes from
    # the store, so the original url_or_data values come through.
    elements_view = ds_with_overwrite.elements
    assert list(elements_view[url_col]) == ["data-0", "data-1", "data-2"]


def test_assign_to_encoding_column_is_ignored_at_view_layer():
    """Symmetric to the url_or_data case: the encoding column is also
    re-synthesized from the store and not overridable via `assign`.
    """
    ds = _trivial_pickle_dataset(3)
    enc_col = ELEMENT_COLS.LOAD_MECHANISM.ENCODING
    ds_with_overwrite = ds.assign(**{enc_col: lambda df: ["bogus"] * 3})
    elements_view = ds_with_overwrite.elements
    assert list(elements_view[enc_col]) == ["pickle", "pickle", "pickle"]
