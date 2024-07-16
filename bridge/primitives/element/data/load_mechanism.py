from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from typing_extensions import Self

from bridge.primitives.element.data import data_io
from bridge.primitives.element.data.uri_components import URIComponents
from bridge.utils import Dictable
from bridge.utils.constants import ELEMENT_COLS

if TYPE_CHECKING:
    from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE


class LoadMechanism(Dictable):
    keys = ELEMENT_COLS.LOAD_MECHANISM.list()

    def __init__(self, url_or_data: URIComponents | ELEMENT_DATA_TYPE, category: str) -> None:
        assert data_io.is_registered(category), f"Category {category} is not registered."
        self._url_or_data = url_or_data
        self._category = category

    @property
    def url_or_data(self) -> URIComponents | ELEMENT_DATA_TYPE:
        return self._url_or_data

    @property
    def category(self) -> str:
        return self._category

    def load_data(self) -> Any:
        return data_io.load(self._url_or_data, self._category)

    def to_dict(self) -> Dict[str, Any]:
        return {
            ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA: self.url_or_data,
            ELEMENT_COLS.LOAD_MECHANISM.CATEGORY: self.category,
        }

    @classmethod
    def from_dict(cls, dic: Dict[str, Any], **kwargs) -> Self:
        assert set(dic.keys()).issuperset(set(cls.keys))
        return cls(
            url_or_data=dic[ELEMENT_COLS.LOAD_MECHANISM.URL_OR_DATA],
            category=dic[ELEMENT_COLS.LOAD_MECHANISM.CATEGORY],
        )

    @classmethod
    def from_url_string(cls, url_string: str, category: str) -> Self:
        components = URIComponents.from_str(url_string)
        return cls(components, category)
