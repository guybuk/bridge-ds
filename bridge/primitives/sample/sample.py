from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Hashable, List

import pandas as pd

from bridge.primitives.element.data.cache_mechanism import CacheMechanism
from bridge.primitives.element.element import Element
from bridge.utils.constants import ELEMENT_COLS, INDICES
from bridge.utils.helper import Displayable

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE
    from bridge.primitives.sample.transform.sample_transform import SampleTransform


class Sample(Displayable):
    keys = ELEMENT_COLS.list()

    def __init__(
        self,
        elements: List[Element] | Dict[str, List[Element]],
        display_engine: DisplayEngine | None = None,
    ):
        if isinstance(elements, dict):
            self._elements = elements
        else:
            self._elements = self._convert_elements_list_to_dict(elements)

        self._assert_valid_elements(self._elements)

        self._display_engine = display_engine

    @property
    def id(self) -> Hashable:
        for e_list in self._elements.values():
            if len(e_list) > 0:
                return e_list[0].sample_id

    @property
    def elements(self) -> Dict[str, List[Element]]:
        return self._elements

    @property
    def data(self) -> Dict[str, List[ELEMENT_DATA_TYPE]]:
        data_dict = defaultdict(list)
        for etype, elist in self._elements.items():
            data_dict[etype].extend([e.data for e in elist])
        return dict(data_dict)

    def show(self, **kwargs: Any):
        return self._display_engine.show_sample(self, **kwargs)

    def transform(
        self,
        transform: SampleTransform,
        cache_mechanisms: Dict[str, CacheMechanism] | None = None,
        display_engine: DisplayEngine | None = None,
    ) -> "Sample":
        cache_mechanisms = self._get_cache_mechanisms_for_transform(self, cache_mechanisms)
        if display_engine is None:
            return transform(self, cache_mechanisms, self._display_engine)
        else:
            return transform(self, cache_mechanisms, display_engine)

    @classmethod
    def from_pd_dataframe(
        cls,
        elements_df: pd.DataFrame,
        display_engine: DisplayEngine | None,
        cache_mechanisms: Dict[str, CacheMechanism | None],
    ):
        def fast_to_dict_records(df):
            data = df.values.tolist()
            columns = df.columns.tolist()
            index = df.index.names
            index_data = df.index.values.tolist()
            idx_records = [dict(zip(index, data)) for data in index_data]
            records = [dict(zip(columns, datum)) for datum in data]
            [rec.update(idx_rec) for rec, idx_rec in zip(records, idx_records)]
            return records

        elements = []
        for element_row in fast_to_dict_records(elements_df):
            etype = element_row[ELEMENT_COLS.ETYPE]

            element_type = str(etype)
            elements.append(
                Element.from_dict(
                    element_row,
                    display_engine=display_engine,
                    cache_mechanism=cache_mechanisms.get(element_type),
                )
            )
        return cls(elements=elements, display_engine=display_engine)

    def to_pd_dataframe(self):
        records = []
        for e_list in self.elements.values():
            for element in e_list:
                records.append(element.to_pd_series())
        return pd.DataFrame(records).set_index(INDICES)

    def __len__(self) -> int:
        return sum(map(len, self._elements.values()))

    @staticmethod
    def _assert_valid_elements(elements: Dict[str, List[Element]]):
        sample_ids_from_elements = set([e.sample_id for e_list in elements.values() for e in e_list])
        assert len(sample_ids_from_elements) == 1, (
            f"All elements must contain a single sample id,"
            f" got {len(sample_ids_from_elements)}: {sample_ids_from_elements}"
        )

    @staticmethod
    def _convert_elements_list_to_dict(elements: List[Element]) -> Dict[str, List[Element]]:
        elements_by_type: Dict[str, List[Element]] = defaultdict(list)
        for element in elements:
            elements_by_type[element.etype].append(element)
        d = dict(elements_by_type)
        return d

    @staticmethod
    def _get_cache_mechanisms_for_transform(sample: Sample, cache_mechanisms: Dict[str, CacheMechanism] | None):
        if cache_mechanisms is None:
            cache_mechanisms = {}

        sample_etypes = sample.elements.keys()
        if set(sample_etypes) == set(cache_mechanisms.keys()):
            return cache_mechanisms

        default_cache_mechanisms = {etype: CacheMechanism() for etype in sample_etypes}

        missing_keys = set(default_cache_mechanisms.keys()) - set(cache_mechanisms.keys())
        if len(missing_keys) > 0:
            pass
            # warnings.warn(
            #     f"The following keys don't have cache mechanisms and are set to `memory` by default: {missing_keys}"
            # )
        default_cache_mechanisms.update(cache_mechanisms)
        return default_cache_mechanisms
