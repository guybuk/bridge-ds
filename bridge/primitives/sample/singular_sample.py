from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

from typing_extensions import Self

from bridge.primitives.sample import Sample
from bridge.utils.constants import IS_SAMPLE_COL_NAME

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism
    from bridge.primitives.element.element import Element
    from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE
    from bridge.primitives.sample.transform import SampleTransform


class SingularSample(Sample):
    def __init__(self, elements: List[Element] | Dict[str, List[Element]], display_engine: DisplayEngine = None):
        super().__init__(elements, display_engine)
        self._set_element_and_annotations()

    def _set_element_and_annotations(self):
        element = None
        found_element = False
        annotations = defaultdict(list)
        for etype, e_list in self.elements.items():
            for e in e_list:
                if e.metadata.get(IS_SAMPLE_COL_NAME) is True and found_element:
                    raise RuntimeError()
                if e.metadata.get(IS_SAMPLE_COL_NAME) is True:
                    found_element = True
                    element = e
                else:
                    annotations[etype].append(e)
        if not found_element:
            raise RuntimeError()

        self._element = element
        self._annotations = dict(annotations)

    @property
    def element(self) -> Element:
        return self._element

    @property
    def data(self) -> ELEMENT_DATA_TYPE:
        return self._element.data

    @property
    def annotations(self) -> Dict[str, List[Element]]:
        return self._annotations

    def transform(
        self,
        transform: SampleTransform,
        cache_mechanisms: Dict[str, CacheMechanism] | None = None,
        display_engine: DisplayEngine | None = None,
    ) -> Self:
        transformed_sample = super().transform(transform, cache_mechanisms, display_engine)
        return self.from_sample(transformed_sample)

    @classmethod
    def from_sample(cls, sample: Sample) -> Self:
        return cls(sample._elements, sample._display_engine)
