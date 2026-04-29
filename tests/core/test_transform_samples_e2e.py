"""End-to-end tests for Dataset.transform_samples through the multi-role store path.

`Dataset.transform_samples` is the heaviest machinery in the multi-role
API. Until now it was only smoke-tested via the slow notebook suite.
This test exercises the round-trip with a trivial pickle-only no-op
transform, so it stays in tests/core/ and runs without the vision extra.
"""
from __future__ import annotations

from typing import Dict

from bridge.primitives.dataset import Dataset
from bridge.primitives.element.data.cache_mechanism import CacheMechanism
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import Sample
from bridge.primitives.sample.transform.sample_transform import SampleTransform
from bridge.utils.constants import ELEMENT_COLS


class _NoOpPickleTransform(SampleTransform):
    """No-op transform that re-emits each element unchanged.

    The element data is loaded once (so the cache-or-not path is
    exercised), then a fresh Element is constructed pointing at the
    in-memory data via a pickle-encoded LoadMechanism. The
    transformed Dataset's elements should compare equal to the
    originals.
    """

    def __call__(
        self,
        sample: Sample,
        cache_mechanisms: Dict[str, CacheMechanism],
        display_engine,
    ) -> Sample:
        new_elements = {}
        for role, elems in sample.elements.items():
            new_elems = []
            for elem in elems:
                # Force a load so cache machinery (if any) has a chance to fire.
                data = elem.data
                new_elems.append(
                    Element(
                        element_id=elem.id,
                        etype=elem.etype,
                        sample_id=elem.sample_id,
                        load_mechanism=LoadMechanism(data, encoding="pickle"),
                        role=elem.role,
                        metadata=elem.metadata,
                    )
                )
            new_elements[role] = new_elems
        return Sample(elements=new_elements, display_engine=display_engine)


def _make_dataset_with_two_roles(n_samples: int = 3) -> Dataset:
    role_a_elems = []
    role_b_elems = []
    for i in range(n_samples):
        role_a_elems.append(
            Element(
                element_id=f"a_{i}",
                sample_id=i,
                etype="t",
                load_mechanism=LoadMechanism(f"a-data-{i}", encoding="pickle"),
            )
        )
        role_b_elems.append(
            Element(
                element_id=f"b_{i}",
                sample_id=i,
                etype="t",
                load_mechanism=LoadMechanism(f"b-data-{i}", encoding="pickle"),
            )
        )
    return Dataset.from_role_dict({"role_a": role_a_elems, "role_b": role_b_elems})


def test_transform_samples_preserves_roles_e2e():
    ds = _make_dataset_with_two_roles(n_samples=3)
    transformed = ds.transform_samples(transform=_NoOpPickleTransform())

    assert len(transformed) == 3
    # Every sample must still have both roles.
    for sample in transformed:
        assert "role_a" in sample.elements
        assert "role_b" in sample.elements
    # Every element id must resolve in the transformed dataset's store.
    eids = transformed._df.index.get_level_values(ELEMENT_COLS.ID)
    for eid in eids:
        assert transformed._store.get(eid) is not None


def test_transform_samples_data_round_trips_through_pickle():
    ds = _make_dataset_with_two_roles(n_samples=2)
    transformed = ds.transform_samples(transform=_NoOpPickleTransform())
    # The no-op transform preserves data byte-for-byte.
    for original_sample, new_sample in zip(ds, transformed):
        for role in ("role_a", "role_b"):
            assert original_sample.one(role).data == new_sample.one(role).data


def test_transform_samples_roundtrips_element_ids():
    """Element ids must survive the round-trip — the no-op transform
    constructs new Elements with the same ids, and Dataset.from_elements
    builds a fresh store keyed by those ids.
    """
    ds = _make_dataset_with_two_roles(n_samples=3)
    transformed = ds.transform_samples(transform=_NoOpPickleTransform())

    original_eids = set(ds._df.index.get_level_values(ELEMENT_COLS.ID))
    new_eids = set(transformed._df.index.get_level_values(ELEMENT_COLS.ID))
    assert original_eids == new_eids
