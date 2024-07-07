from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Generic

from bridge.display.basic import SimplePrints
from bridge.primitives.types import D, S

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache.cache_mechanism import CacheMechanism
    from bridge.primitives.element.element_type import ElementType


class DatasetProvider(ABC, Generic[D, S]):
    @abstractmethod
    def build_dataset(
        self,
        display_engine: DisplayEngine[D, S] = SimplePrints(),
        cache_mechanisms: Dict[ElementType, CacheMechanism | None] | None = None,
    ) -> D:
        pass
