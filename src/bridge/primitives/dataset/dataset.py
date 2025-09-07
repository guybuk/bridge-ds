from __future__ import annotations

import functools
from types import GeneratorType
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, Iterable, Iterator, List, Sequence

import pandas as pd
from typing_extensions import Self

from bridge.primitives.dataset.sample_api import SampleAPI
from bridge.primitives.dataset.table_api import TableAPI
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
        display_engine: DisplayEngine = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ):
        self._elements = elements
        self._display_engine = display_engine
        self._cache_mechanisms = cache_mechanisms or {}
        self._connect_caches()

    @property
    def elements(self) -> pd.DataFrame:
        return self._elements.copy()

    @property
    def sample_ids(self) -> List[Hashable]:
        return self._elements.index.get_level_values(ELEMENT_COLS.SAMPLE_ID).drop_duplicates().to_list()

    def select(self, selector: Callable):
        selected = selector(self.elements)
        elements = self.elements.loc[selected]
        return Dataset(elements, display_engine=self._display_engine, cache_mechanisms=self._cache_mechanisms)

    def assign(self, **kwargs: Dict[str, Callable[[pd.DataFrame], Sequence]]) -> Self:
        new_elements = self._elements.assign(**kwargs)
        return Dataset(new_elements, display_engine=self._display_engine, cache_mechanisms=self._cache_mechanisms)

    def sort(self, by: str, ascending: bool = True):
        new_elements = self._elements.sort_values(by=by, ascending=ascending)
        return Dataset(new_elements, display_engine=self._display_engine, cache_mechanisms=self._cache_mechanisms)

    def merge(
        self,
        other: "Dataset",
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> "Dataset":
        self_element_ids = self.elements.index.get_level_values(ELEMENT_COLS.ID)
        other_element_ids = other.elements.index.get_level_values(ELEMENT_COLS.ID)
        assert (
            len(self_element_ids.intersection(other_element_ids)) == 0
        ), "Cannot merge Datasets with duplicate element ids."
        elements = pd.concat([self.elements, other.elements])
        if display_engine is None:
            display_engine = self._display_engine
        if cache_mechanisms is None:
            cache_mechanisms = self._cache_mechanisms
        return Dataset(elements, display_engine, cache_mechanisms=cache_mechanisms)

    def iget(self, index: int) -> Sample:
        sample_id = self.sample_ids[index]
        return self.get(sample_id)

    def get(self, sample_id: Hashable) -> Sample:
        sample_df = self._elements.xs(sample_id, level=ELEMENT_COLS.SAMPLE_ID, drop_level=False)
        return Sample.from_pd_dataframe(
            sample_df, display_engine=self._display_engine, cache_mechanisms=self._cache_mechanisms
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
        for etype, group in self._elements.groupby(ELEMENT_COLS.ETYPE):
            lens_dict[f"n_{etype}"] = len(group)
        return "Dataset: " + str(lens_dict)

    @classmethod
    def from_elements(
        cls,
        elements: Iterable[Element],
        display_engine: DisplayEngine = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> Self:
        element_records = [e.to_pd_series() for e in elements]
        elements_df = pd.DataFrame(element_records).set_index(INDICES)
        return cls(elements=elements_df, display_engine=display_engine, cache_mechanisms=cache_mechanisms)

    def _connect_caches(self):
        for cache in self._cache_mechanisms.values():
            if cache is not None:
                cache.set_elements_df(self._elements)
