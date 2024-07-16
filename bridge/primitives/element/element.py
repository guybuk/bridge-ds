from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Hashable

import pandas as pd

from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.utils import validate_metadata
from bridge.utils.constants import ELEMENT_COLS
from bridge.utils.helper import Displayable

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism
    from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE
    from bridge.primitives.element.element_type import ElementType


class Element(Displayable):
    keys = ELEMENT_COLS.list()

    def __init__(
        self,
        element_id: Hashable,
        etype: ElementType,
        load_mechanism: LoadMechanism,
        sample_id: Hashable,
        display_engine: DisplayEngine | None = None,
        cache_mechanism: CacheMechanism | None = None,
        metadata: Dict[str, Any] | None = None,
    ):
        validate_metadata(self.keys, metadata)
        self._element_id = element_id
        self._etype = etype
        self._sample_id = sample_id
        self._load_mechanism = load_mechanism
        self._display_engine = display_engine
        self._cache_mechanism = cache_mechanism
        self._metadata = metadata or {}

    @property
    def id(self) -> Hashable:
        return self._element_id

    @property
    def data(self) -> ELEMENT_DATA_TYPE:
        return self._data_impl()

    def _data_impl(self):
        data = self._load_mechanism.load_data()
        if self._cache_mechanism:
            new_load_mechanism = self._cache_mechanism.store(self, data, should_update_elements=True)
            self._load_mechanism = new_load_mechanism
            return data
        return data

    @property
    def etype(self) -> ElementType:
        return self._etype

    @property
    def category(self) -> str:
        return self._load_mechanism.category

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def sample_id(self) -> Hashable:
        return self._sample_id

    def __str__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            ELEMENT_COLS.ID: self.id,
            ELEMENT_COLS.ETYPE: self.etype,
            ELEMENT_COLS.SAMPLE_ID: self.sample_id,
            **self._load_mechanism.to_dict(),
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, dic: Dict[str, Any], **kwargs):
        assert set(dic.keys()).issuperset(set(cls.keys)), f"Missing keys: {set(cls.keys) - set(dic.keys())}"
        element_id = dic[ELEMENT_COLS.ID]
        sample_id = dic[ELEMENT_COLS.SAMPLE_ID]
        etype = dic[ELEMENT_COLS.ETYPE]
        load_mechanism = LoadMechanism.from_dict({k: v for k, v in dic.items() if k in LoadMechanism.keys})
        metadata = {k: v for k, v in dic.items() if k not in cls.keys}
        return cls(
            element_id=element_id,
            etype=etype,
            load_mechanism=load_mechanism,
            sample_id=sample_id,
            display_engine=kwargs.get("display_engine"),
            cache_mechanism=kwargs.get("cache_mechanism"),
            metadata=metadata,
        )

    def to_pd_series(self) -> pd.Series:
        return pd.Series(self.to_dict())

    @classmethod
    def from_pd_series(
        cls,
        element_series: pd.Series,
        display_engine: DisplayEngine | None = None,
        cache_mechanism: CacheMechanism | None = None,
    ):
        return cls.from_dict(
            {**element_series.to_dict()},
            display_engine=display_engine,
            cache_mechanism=cache_mechanism,
        )

    def show(self, **kwargs):
        return self._display_engine.show_element(self, **kwargs)
