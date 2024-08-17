from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, Iterable, Iterator, Sequence

from tqdm.contrib import tmap
from typing_extensions import Self

if TYPE_CHECKING:
    from bridge.display.display_engine import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism
    from bridge.primitives.element.element import Element
    from bridge.primitives.sample import Sample
    from bridge.primitives.sample.transform import SampleTransform


class SampleAPI(abc.ABC):
    @abc.abstractmethod
    def iget(self, index: int) -> Sample:
        pass

    @abc.abstractmethod
    def get(self, sample_id: Hashable) -> Sample:
        pass

    @abc.abstractmethod
    def transform_samples(
        self,
        transform: SampleTransform,
        map_fn=tmap,
        cache_mechanisms: Dict[str, CacheMechanism] | None = None,
        display_engine: DisplayEngine | None = None,
    ) -> Self:
        pass

    @abc.abstractmethod
    def map_samples(self, function: Callable[[Sample], Any], map_fn=tmap) -> Sequence[Sample]:
        pass

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @classmethod
    @abc.abstractmethod
    def from_elements(cls, elements: Iterable[Element]) -> Self:
        pass
