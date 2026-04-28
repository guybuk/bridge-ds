"""Regression test for the cache-aliasing bug.

Today's behavior: when a Dataset is derived (via select/assign/sort/merge),
all derived Datasets share the same CacheMechanism instance. The cache
holds a reference to a DataFrame that gets reassigned at every Dataset
construction (via _connect_caches → set_elements_df). After two
derivations from the same parent, the cache only knows about the
most-recently-constructed Dataset's DataFrame. Reads on any earlier
Dataset write the URI bookkeeping into the wrong table.

Expected behavior post-fix: all Datasets in a lineage share an
ElementStore. Cache writes update the store; every Dataset sees the
update via its derived `elements` property.
"""

from __future__ import annotations

import numpy as np
import pytest

from bridge.primitives.dataset import Dataset
from bridge.primitives.element.data.cache_mechanism import CacheMechanism
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.data.uri_components import URIComponents
from bridge.primitives.element.element import Element
from bridge.utils.constants import ELEMENT_COLS
from bridge.utils.data_objects import ClassLabel


@pytest.fixture
def tmp_cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def dataset_with_two_derivations(tmp_cache_dir):
    """Build a Dataset of 4 in-memory image elements + a CacheMechanism
    pointing at a temp dir, then derive two non-overlapping subsets.
    """
    elements = []
    for i in range(4):
        img = np.full((4, 4, 3), i, dtype=np.uint8)
        elements.append(
            Element(
                element_id=f"img_{i}",
                sample_id=i,
                etype="image",
                # encoding="pickle" rather than "jpeg" so the test runs in the
                # core matrix (jpeg storage requires skimage from vision extras).
                # The aliasing bug is encoding-agnostic; pickle exercises the same
                # cache code path.
                load_mechanism=LoadMechanism(img, encoding="pickle"),
                metadata={"group": "a" if i < 2 else "b"},
            )
        )
        elements.append(
            Element(
                element_id=f"label_{i}",
                sample_id=i,
                etype="class_label",
                load_mechanism=LoadMechanism(ClassLabel(i), encoding="pickle"),
            )
        )
    cache = CacheMechanism(URIComponents.from_str(str(tmp_cache_dir)))
    ds = Dataset.from_elements(elements, cache_mechanisms={"image": cache})
    ds_a = ds.select(lambda df: df["group"] == "a")
    ds_b = ds.select(lambda df: df["group"] == "b")
    return ds, ds_a, ds_b


def test_aliasing_read_on_a_does_not_corrupt_b(dataset_with_two_derivations):
    """Reading an element on ds_a must not change ds_b's view of any element.

    Today, the cache mechanism's _elements pointer ends up at ds_b after
    ds_b is constructed. A cache fire from ds_a writes the URI into
    ds_b's DataFrame instead of ds_a's. This test catches that.
    """
    ds, ds_a, ds_b = dataset_with_two_derivations

    # Snapshot ds_b's url_or_data column for image elements before reading anything
    before_b = ds_b.elements.loc[
        ds_b.elements[ELEMENT_COLS.ETYPE] == "image",
        ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA,
    ].tolist()

    # Trigger a cache write on ds_a
    sample = ds_a.iget(0)
    _ = sample.one("image").data

    # ds_b's image url_or_data must be exactly what it was before
    after_b = ds_b.elements.loc[
        ds_b.elements[ELEMENT_COLS.ETYPE] == "image",
        ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA,
    ].tolist()

    assert len(before_b) == len(after_b), "row count changed"
    for before_row, after_row in zip(before_b, after_b):
        # An ndarray comparison via `==` returns an array; use np.array_equal
        if isinstance(before_row, np.ndarray) and isinstance(after_row, np.ndarray):
            assert np.array_equal(before_row, after_row), "ds_b's image data changed after read on ds_a"
        else:
            assert type(before_row) is type(after_row), (
                f"ds_b's url_or_data type changed: {type(before_row)} -> {type(after_row)}"
            )


def test_aliasing_read_on_a_updates_a_view(dataset_with_two_derivations):
    """After reading an element on ds_a (cache fires), ds_a's elements
    should show the cached URI. This is the symmetric requirement: the
    cache update must be visible on the Dataset that triggered it.
    """
    ds, ds_a, ds_b = dataset_with_two_derivations

    sample = ds_a.iget(0)
    _ = sample.one("image").data

    # ds_a should now see a URIComponents (the cache location), not the
    # original ndarray, in url_or_data for the image element it just read.
    sample_id = sample.id
    images = ds_a.elements.loc[
        (ds_a.elements[ELEMENT_COLS.ETYPE] == "image"),
    ]
    matching = images.loc[
        images.index.get_level_values(ELEMENT_COLS.SAMPLE_ID) == sample_id
    ]
    assert len(matching) == 1
    new_value = matching[ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA].iloc[0]
    assert isinstance(new_value, URIComponents), (
        f"Expected URIComponents after cache fire, got {type(new_value).__name__}"
    )


def test_aliasing_read_on_a_propagates_to_b_when_element_id_present(dataset_with_two_derivations):
    """If the element exists in both ds_a and ds_b (overlap case), a
    cache fire on ds_a should be visible on ds_b. (In our fixture the
    splits are disjoint so this doesn't apply to images, but we can test
    via the parent ds, which contains all elements.)
    """
    ds, ds_a, ds_b = dataset_with_two_derivations

    sample = ds_a.iget(0)
    sample_id = sample.id
    _ = sample.one("image").data

    # The parent ds contains every element. The element we just cached
    # should now show URIComponents in ds.elements too.
    images = ds.elements.loc[
        (ds.elements[ELEMENT_COLS.ETYPE] == "image"),
    ]
    matching = images.loc[
        images.index.get_level_values(ELEMENT_COLS.SAMPLE_ID) == sample_id
    ]
    assert len(matching) == 1
    new_value = matching[ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA].iloc[0]
    assert isinstance(new_value, URIComponents), (
        f"Expected URIComponents on parent ds after cache fire on derived ds_a, "
        f"got {type(new_value).__name__}"
    )


def test_dataset_df_does_not_carry_location_columns(dataset_with_two_derivations):
    """After ElementStore landed, Dataset._df should never carry url_or_data
    or encoding columns — they live in the store. This invariant must
    hold for derived Datasets too, not just freshly-constructed ones.
    """
    ds, ds_a, ds_b = dataset_with_two_derivations
    url_col = ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA
    enc_col = ELEMENT_COLS.LOAD_MECHANISM.ENCODING
    for d in (ds, ds_a, ds_b):
        assert url_col not in d._df.columns, f"_df has leaked {url_col} on {d!r}"
        assert enc_col not in d._df.columns, f"_df has leaked {enc_col} on {d!r}"


def test_cache_mechanism_rebind_to_different_store_raises(tmp_cache_dir):
    """CacheMechanism is single-lineage. Rebinding to a different store is
    a sign that the cache is being shared across independent Dataset
    lineages, which would reintroduce the aliasing class of bug.
    """
    from bridge.primitives.element.data.element_store import ElementStore
    cache = CacheMechanism(URIComponents.from_str(str(tmp_cache_dir)))
    store_a = ElementStore()
    store_b = ElementStore()
    cache.bind_store(store_a)
    # Rebinding to the same store is fine (idempotent for a lineage's
    # multiple Datasets each calling bind_store).
    cache.bind_store(store_a)
    # Rebinding to a different store must raise.
    with pytest.raises(ValueError, match="different ElementStore"):
        cache.bind_store(store_b)
