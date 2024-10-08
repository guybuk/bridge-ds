from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism
    from bridge.primitives.sample import Sample


class SampleTransform(ABC):
    @abstractmethod
    def __call__(
        self,
        sample: Sample,
        cache_mechanisms: Dict[str, CacheMechanism] | None,
        display_engine: DisplayEngine | None,
    ) -> Sample:
        pass
