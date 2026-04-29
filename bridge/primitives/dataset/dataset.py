from __future__ import annotations

import functools
from types import GeneratorType
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, Iterable, Iterator, List, Sequence

import pandas as pd
from typing_extensions import Self

from bridge.primitives.dataset.sample_api import SampleAPI
from bridge.primitives.dataset.table_api import TableAPI
from bridge.primitives.element.data.element_store import ElementStore
from bridge.primitives.sample import Sample
from bridge.utils.constants import ELEMENT_COLS, INDICES
from bridge.utils.helper import Displayable

if TYPE_CHECKING:
    from bridge.display.display_engine import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism
    from bridge.primitives.element.element import Element
    from bridge.primitives.sample.transform import SampleTransform


class Dataset(TableAPI, SampleAPI, Displayable):
    def __init__(
        self,
        elements: pd.DataFrame,
        store: ElementStore | None = None,
        display_engine: DisplayEngine = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ):
        if store is None:
            store, elements = self._extract_store_from_df(elements)
        else:
            # Always strip location columns from _df, even when store is provided —
            # they may have leaked back in via derivations that pass `self.elements`
            # (which synthesizes the columns) into a new Dataset constructor.
            url_col = ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA
            enc_col = ELEMENT_COLS.LOAD_MECHANISM.ENCODING
            cols_to_drop = [c for c in (url_col, enc_col) if c in elements.columns]
            if cols_to_drop:
                elements = elements.drop(columns=cols_to_drop)
        self._df = elements
        self._store = store
        self._display_engine = display_engine
        self._cache_mechanisms = cache_mechanisms or {}
        for cache in self._cache_mechanisms.values():
            if cache is not None:
                cache.bind_store(self._store)

    @staticmethod
    def _extract_store_from_df(df: pd.DataFrame) -> tuple[ElementStore, pd.DataFrame]:
        """Build an ElementStore from a DataFrame's url_or_data + encoding
        columns (back-compat path for callers that hand-build a DataFrame).
        Returns the new store and a DataFrame without those columns.
        """
        store = ElementStore()
        url_col = ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA
        enc_col = ELEMENT_COLS.LOAD_MECHANISM.ENCODING
        if url_col not in df.columns or enc_col not in df.columns:
            return store, df

        # Lazy import to avoid the circular dep noted on the dataset module.
        from bridge.primitives.element.data.load_mechanism import LoadMechanism

        eids = df.index.get_level_values(ELEMENT_COLS.ID).values
        url_vals = df[url_col].values
        enc_vals = df[enc_col].values
        for eid, uod, enc in zip(eids, url_vals, enc_vals):
            store.set(eid, LoadMechanism(uod, encoding=enc))
        return store, df.drop(columns=[url_col, enc_col])

    def _join_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add url_or_data / encoding columns to a copy of df by joining
        with the lineage's ElementStore. Used by `elements` (full table)
        and by SingularDataset's samples/annotations (filtered subsets).
        """
        df = df.copy()
        url_col = ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA
        enc_col = ELEMENT_COLS.LOAD_MECHANISM.ENCODING
        eids = df.index.get_level_values(ELEMENT_COLS.ID)
        lms = [self._store.get(eid) for eid in eids]
        df[url_col] = [lm.url_or_data for lm in lms]
        df[enc_col] = [lm.encoding for lm in lms]
        return df

    @property
    def elements(self) -> pd.DataFrame:
        return self._join_locations(self._df)

    @property
    def sample_ids(self) -> List[Hashable]:
        return self._df.index.get_level_values(ELEMENT_COLS.SAMPLE_ID).drop_duplicates().to_list()

    @property
    def display_engine(self) -> DisplayEngine | None:
        return self._display_engine

    def select(self, selector: Callable):
        # selector receives the user-facing elements (with location columns) for filtering
        selected = selector(self.elements)
        elements = self._df.loc[selected]
        return Dataset(
            elements,
            store=self._store,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def assign(self, **kwargs: Dict[str, Callable[[pd.DataFrame], Sequence]]) -> Self:
        new_df = self._df.assign(**kwargs)
        return Dataset(
            new_df,
            store=self._store,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def sort(self, by: str, ascending: bool = True):
        new_df = self._df.sort_values(by=by, ascending=ascending)
        return Dataset(
            new_df,
            store=self._store,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def merge(
        self,
        other: "Dataset",
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> "Dataset":
        """Merge two Datasets. Element ids must be disjoint.

        Combines stores from both sides via ElementStore.extend. When
        cache_mechanisms is None, merges the two sides' caches with
        `other`'s entries overriding `self`'s on key conflict.
        """
        self_eids = self._df.index.get_level_values(ELEMENT_COLS.ID)
        other_eids = other._df.index.get_level_values(ELEMENT_COLS.ID)
        assert (
            len(self_eids.intersection(other_eids)) == 0
        ), "Cannot merge Datasets with duplicate element ids."
        elements = pd.concat([self._df, other._df])
        merged_store = ElementStore()
        merged_store.extend(self._store)
        merged_store.extend(other._store)
        if display_engine is None:
            display_engine = self._display_engine
        if cache_mechanisms is None:
            cache_mechanisms = {**self._cache_mechanisms, **other._cache_mechanisms}
        return Dataset(
            elements,
            store=merged_store,
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )

    def iget(self, index: int) -> Sample:
        sample_id = self.sample_ids[index]
        return self.get(sample_id)

    def get(self, sample_id: Hashable) -> Sample:
        sample_df = self._df.xs(sample_id, level=ELEMENT_COLS.SAMPLE_ID, drop_level=False)
        return Sample.from_pd_dataframe(
            sample_df,
            store=self._store,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def transform_samples(
        self,
        transform: SampleTransform,
        map_fn=map,
        cache_mechanisms: Dict[str, CacheMechanism] | None = None,
        display_engine: DisplayEngine | None = None,
    ) -> Self:
        fn = functools.partial(
            Sample.transform, transform=transform, cache_mechanisms=cache_mechanisms, display_engine=display_engine
        )
        samples = map_fn(fn, self)
        if isinstance(samples, GeneratorType):
            samples = list(samples)
        elements = [element for sample in samples for e_list in sample.elements.values() for element in e_list]
        return Dataset.from_elements(elements, display_engine=display_engine)

    def map_samples(self, function: Callable[[Sample], Any], map_fn=map):
        outputs = map_fn(function, self)
        if isinstance(outputs, GeneratorType):
            return list(outputs)
        return outputs

    def show(self, **kwargs):
        return self._display_engine.show_dataset(self, **kwargs)

    def __getitem__(self, item):
        return self.iget(item)

    def __iter__(self) -> Iterator[Sample]:
        for sample_id in self.sample_ids:
            yield self.get(sample_id)

    def __len__(self) -> int:
        sample_ids = self.sample_ids
        return len(sample_ids)

    def __repr__(self) -> str:
        lens_dict = {"n_samples": len(self)}
        for etype, group in self._df.groupby(ELEMENT_COLS.ETYPE):
            lens_dict[f"n_{etype}"] = len(group)
        return "Dataset: " + str(lens_dict)

    @classmethod
    def from_elements(
        cls,
        elements: Iterable[Element],
        display_engine: DisplayEngine = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> Self:
        store = ElementStore()
        records = []
        for elem in elements:
            store.set(elem.id, elem._load_mechanism)
            records.append(elem.to_pd_series())
        elements_df = pd.DataFrame(records).set_index(INDICES)
        return cls(
            elements=elements_df,
            store=store,
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )

    @classmethod
    def from_role_dict(
        cls,
        elements_by_role: Dict[str, List[Element]],
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> Self:
        from bridge.primitives.element.element import Element

        store = ElementStore()
        records = []
        for role, elements in elements_by_role.items():
            for elem in elements:
                record = elem.to_dict()
                record[ELEMENT_COLS.ROLE] = role
                records.append(pd.Series(record))
                store.set(elem.id, elem._load_mechanism)
        if not records:
            elements_df = pd.DataFrame(columns=Element.keys).set_index(INDICES)
        else:
            elements_df = pd.DataFrame(records).set_index(INDICES)
        return cls(
            elements=elements_df,
            store=store,
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )
