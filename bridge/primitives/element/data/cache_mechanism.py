from __future__ import annotations

from typing import TYPE_CHECKING, Hashable

import pandas as pd

from bridge.primitives.element.data import encoding_registry
from bridge.primitives.element.data.uri_components import URIComponents

if TYPE_CHECKING:
    from bridge.primitives.element.data.load_mechanism import LoadMechanism
    from bridge.primitives.element.element import Element
    from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE


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
        as_encoding: str | None = None,
        should_update_elements: bool = False,
    ) -> LoadMechanism:
        if as_encoding is None:
            as_encoding = element.encoding
        assert encoding_registry.is_registered(as_encoding), f"Encoding {as_encoding} is not registered."
        uri = self._build_uri(element, as_encoding)
        new_provider = encoding_registry.store(data, uri, as_encoding)
        if should_update_elements and self._elements is not None:
            self._update_samples_with_new_provider(element.id, new_provider)
        return new_provider

    def _build_uri(self, element: Element, encoding: str) -> URIComponents | None:
        if self._root_uri is None:
            return None

        uri = URIComponents(
            scheme=self._root_uri.scheme,
            path=self._root_uri.path + f"/{element.id}{encoding_registry.extension(encoding)}",
        )
        return uri

    def _update_samples_with_new_provider(self, element_id: Hashable, new_provider: LoadMechanism):
        dic = new_provider.to_dict()
        self._elements.loc[(slice(None), element_id), list(dic.keys())] = dic.values()
