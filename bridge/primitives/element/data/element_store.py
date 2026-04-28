"""ElementStore: a shared canonical map of element_id -> LoadMechanism.

The store is shared by reference across all Datasets in a derivation
lineage. Cache writes mutate the store; every Dataset that holds the
same store reference sees the update.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Hashable, Iterator

if TYPE_CHECKING:
    from bridge.primitives.element.data.load_mechanism import LoadMechanism


class ElementStore:
    """Mutable mapping from element_id to LoadMechanism."""

    def __init__(self) -> None:
        self._table: dict[Hashable, "LoadMechanism"] = {}
        # Incremented on every mutation; Datasets read this to detect when
        # cached joined views are stale.
        self._mutation_count: int = 0

    def __contains__(self, element_id: Hashable) -> bool:
        return element_id in self._table

    def __len__(self) -> int:
        return len(self._table)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._table)

    def set(self, element_id: Hashable, load_mechanism: "LoadMechanism") -> None:
        """Set the LoadMechanism for an element_id, raising if already present."""
        if element_id in self._table:
            raise ValueError(
                f"ElementStore already has an entry for {element_id!r}; "
                "use update() to overwrite or extend() to merge stores."
            )
        self._table[element_id] = load_mechanism
        self._mutation_count += 1

    def update(self, element_id: Hashable, load_mechanism: "LoadMechanism") -> None:
        """Set or overwrite the LoadMechanism for an element_id."""
        self._table[element_id] = load_mechanism
        self._mutation_count += 1

    def get(self, element_id: Hashable) -> "LoadMechanism":
        try:
            return self._table[element_id]
        except KeyError:
            raise KeyError(
                f"ElementStore has no entry for element_id={element_id!r}"
            ) from None

    def extend(self, other: "ElementStore") -> None:
        """Merge another store's entries into this one. Element ids must be disjoint."""
        overlap = set(self._table) & set(other._table)
        if overlap:
            raise ValueError(
                f"Cannot extend ElementStore: element_id overlap on {sorted(overlap)!r}"
            )
        self._table.update(other._table)
        self._mutation_count += 1
