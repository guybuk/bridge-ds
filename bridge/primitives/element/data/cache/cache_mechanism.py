from __future__ import annotations

from typing import TYPE_CHECKING, Hashable

import pandas as pd

from bridge.primitives.element.data.cache import cache_methods
from bridge.primitives.element.data.category import DataCategory
from bridge.primitives.element.data.uri_components import URIComponents

if TYPE_CHECKING:
    from bridge.primitives.element.data.load.load_mechanism import LoadMechanism
    from bridge.primitives.element.element import Element
    from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE

CATEGORY_TO_EXTENSION = {
    DataCategory.image: ".jpg",
    DataCategory.torch: ".pt",
    DataCategory.text: ".txt",
    DataCategory.obj: ".pkl",
}


class CacheMechanism:
    def __init__(self, root_uri: URIComponents | None = None):
        self._elements = None
        self._root_uri = root_uri

    def set_elements_df(self, elements: pd.DataFrame):
        self._elements = elements

    def store(
        self,
        element: Element,
        data: ELEMENT_DATA_TYPE,
        as_category: DataCategory | None = None,
        should_update_elements: bool = False,
    ) -> LoadMechanism:
        if as_category is None:
            as_category = element.category
        uri = self._build_uri(element, as_category)
        new_provider = cache_methods.store(data, uri, as_category)
        if should_update_elements and self._elements is not None:
            self._update_samples_with_new_provider(element.id, new_provider)
        return new_provider

    def _build_uri(self, element: Element, category: DataCategory) -> URIComponents | None:
        if self._root_uri is None:
            return None

        uri = URIComponents(
            scheme=self._root_uri.scheme,
            path=self._root_uri.path + f"/{element.id}{CATEGORY_TO_EXTENSION[category]}",
        )
        return uri

    def _update_samples_with_new_provider(self, element_id: Hashable, new_provider: LoadMechanism):
        dic = new_provider.to_dict()
        self._elements.loc[(slice(None), element_id), list(dic.keys())] = dic.values()
