"""Edge-case tests for Dataset constructor and _extract_store_from_df back-compat path.

The constructor has two entry-points:
  * `Dataset(df)` — extracts a fresh ElementStore from the df's url_or_data
    and encoding columns (back-compat for callers that hand-build a
    DataFrame).
  * `Dataset(df, store=...)` — uses the provided store, but still strips
    any leaked url_or_data / encoding columns from the df.

These edge cases were never explicitly pinned during the v0.2 rewrite.
"""
from __future__ import annotations

import pandas as pd

from bridge.primitives.dataset import Dataset
from bridge.primitives.element.data.element_store import ElementStore
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.utils.constants import ELEMENT_COLS, INDICES


def test_extract_store_from_empty_df():
    """Empty DataFrame should construct a Dataset with an empty store."""
    columns = [
        ELEMENT_COLS.SAMPLE_ID,
        ELEMENT_COLS.ID,
        ELEMENT_COLS.ETYPE,
        ELEMENT_COLS.ROLE,
        ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA,
        ELEMENT_COLS.LOAD_MECHANISM.ENCODING,
    ]
    df = pd.DataFrame(columns=columns).set_index(INDICES)
    ds = Dataset(df)
    assert len(ds) == 0
    assert isinstance(ds._store, ElementStore)
    assert len(ds._store) == 0
    # url_or_data and encoding must be stripped from _df even when empty.
    assert ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA not in ds._df.columns
    assert ELEMENT_COLS.LOAD_MECHANISM.ENCODING not in ds._df.columns


def test_missing_url_column_silent_noop_current_behavior():
    """Pin the current silent-bail-out when URL_OR_DATA is missing.

    `_extract_store_from_df` returns the input df unchanged with an empty
    store rather than raising. This is permissive for back-compat, NOT a
    design invariant — if a future PR adds explicit validation, this test
    will turn red and force the change to be intentional.
    """
    df = pd.DataFrame(
        {
            ELEMENT_COLS.ROLE: ["x"],
            ELEMENT_COLS.SAMPLE_ID: [0],
            ELEMENT_COLS.ID: ["e"],
            ELEMENT_COLS.ETYPE: ["t"],
        }
    ).set_index(INDICES)
    ds = Dataset(df)
    # No exception, store is empty.
    assert isinstance(ds._store, ElementStore)
    assert len(ds._store) == 0
    # _df keeps the columns we passed (since the back-compat path bailed early).
    assert ELEMENT_COLS.ROLE in ds._df.columns
    assert ELEMENT_COLS.ETYPE in ds._df.columns


def test_missing_encoding_column_silent_noop_current_behavior():
    """Pin the current silent-bail-out when ENCODING is missing.

    Symmetric to `test_missing_url_column_silent_noop_current_behavior`:
    `_extract_store_from_df` bails without stripping URL_OR_DATA from the
    df when ENCODING is absent. This is permissive back-compat behavior,
    NOT a design invariant — if a future PR adds explicit validation,
    this test will turn red and force the change to be intentional.
    """
    df = pd.DataFrame(
        {
            ELEMENT_COLS.ROLE: ["x"],
            ELEMENT_COLS.SAMPLE_ID: [0],
            ELEMENT_COLS.ID: ["e"],
            ELEMENT_COLS.ETYPE: ["t"],
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: ["data"],
        }
    ).set_index(INDICES)
    ds = Dataset(df)
    assert len(ds._store) == 0
    # The url_or_data column is preserved on _df because the back-compat
    # path bailed without stripping (encoding was missing).
    assert ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA in ds._df.columns


def test_dataset_two_arg_construction_with_existing_store():
    """Dataset(df, store=existing) should not strip the store and should
    drop any leaked url_or_data / encoding columns from _df.
    """
    store = ElementStore()
    store.set("e1", LoadMechanism("data1", encoding="pickle"))
    store.set("e2", LoadMechanism("data2", encoding="pickle"))

    df = pd.DataFrame(
        [
            {
                ELEMENT_COLS.SAMPLE_ID: 0,
                ELEMENT_COLS.ID: "e1",
                ELEMENT_COLS.ETYPE: "t",
                ELEMENT_COLS.ROLE: "t",
            },
            {
                ELEMENT_COLS.SAMPLE_ID: 1,
                ELEMENT_COLS.ID: "e2",
                ELEMENT_COLS.ETYPE: "t",
                ELEMENT_COLS.ROLE: "t",
            },
        ]
    ).set_index(INDICES)

    ds = Dataset(df, store=store)
    assert len(ds) == 2
    # Store unchanged.
    assert ds._store.get("e1").url_or_data == "data1"
    assert ds._store.get("e2").url_or_data == "data2"
    # Constructor strips URL_OR_DATA / ENCODING from _df even though they
    # weren't present here — the invariant is "_df never carries those
    # columns when a store is provided."
    assert ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA not in ds._df.columns
    assert ELEMENT_COLS.LOAD_MECHANISM.ENCODING not in ds._df.columns


def test_dataset_two_arg_construction_strips_leaked_location_columns():
    """If a caller passes a df that has url_or_data / encoding columns
    AND a store, the constructor must strip those columns from _df —
    the store is the source of truth.
    """
    store = ElementStore()
    store.set("e1", LoadMechanism("authoritative-data", encoding="pickle"))

    df = pd.DataFrame(
        [
            {
                ELEMENT_COLS.SAMPLE_ID: 0,
                ELEMENT_COLS.ID: "e1",
                ELEMENT_COLS.ETYPE: "t",
                ELEMENT_COLS.ROLE: "t",
                ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: "stale-data",
                ELEMENT_COLS.LOAD_MECHANISM.ENCODING: "pickle",
            },
        ]
    ).set_index(INDICES)

    ds = Dataset(df, store=store)
    assert ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA not in ds._df.columns
    assert ELEMENT_COLS.LOAD_MECHANISM.ENCODING not in ds._df.columns
    # The elements view re-synthesizes from the store, so the
    # authoritative value comes through.
    assert ds.elements[ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA].iloc[0] == "authoritative-data"
