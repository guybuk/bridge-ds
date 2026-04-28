from __future__ import annotations

from typing import TYPE_CHECKING

from bridge.primitives.element.data import encoding_registry
from bridge.primitives.element.data.uri_components import URIComponents

if TYPE_CHECKING:
    from bridge.primitives.element.data.element_store import ElementStore
    from bridge.primitives.element.data.load_mechanism import LoadMechanism
    from bridge.primitives.element.element import Element
    from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE


class CacheMechanism:
    def __init__(self, root_uri: URIComponents | None = None):
        self._root_uri = root_uri
        self._store: ElementStore | None = None

    def bind_store(self, store: ElementStore) -> None:
        """Attach the lineage's ElementStore. Called by Dataset.__init__.

        Idempotent for the same store. Raises if rebound to a different
        store, since that would mean the cache is being shared across
        lineages — which would reintroduce the aliasing class of bug
        this design is meant to prevent.
        """
        if self._store is not None and self._store is not store:
            raise ValueError(
                "CacheMechanism is already bound to a different ElementStore. "
                "Cache mechanisms are single-lineage; create a new CacheMechanism "
                "for an independent Dataset lineage."
            )
        self._store = store

    def store(
        self,
        element: Element,
        data: ELEMENT_DATA_TYPE,
        as_encoding: str | None = None,
    ) -> LoadMechanism:
        if as_encoding is None:
            as_encoding = element.encoding
        assert encoding_registry.is_registered(as_encoding), f"Encoding {as_encoding} is not registered."
        uri = self._build_uri(element, as_encoding)
        new_provider = encoding_registry.store(data, uri, as_encoding)
        if self._store is not None:
            self._store.update(element.id, new_provider)
        return new_provider

    def _build_uri(self, element: Element, encoding: str) -> URIComponents | None:
        if self._root_uri is None:
            return None

        return URIComponents(
            scheme=self._root_uri.scheme,
            path=self._root_uri.path + f"/{element.id}{encoding_registry.extension(encoding)}",
        )
