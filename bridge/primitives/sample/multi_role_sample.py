from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple

from typing_extensions import Self

from bridge.primitives.sample.sample import Sample
from bridge.utils.constants import ROLE_COL_NAME

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism
    from bridge.primitives.element.element import Element
    from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE
    from bridge.primitives.sample.transform import SampleTransform


class MultiRoleSample(Sample):
    """
    A sample with named element roles.

    Each role can contain one or more elements. Roles are accessed by name
    via properties or dynamic attribute access.

    Example:
        sample = MultiRoleSample(elements, role_names=("image", "bboxes"))
        sample.image      # List[Element] with role "image"
        sample.bboxes     # List[Element] with role "bboxes"
        sample.get_role("image")  # Same as above
        sample.data       # {"image": [...], "bboxes": [...]}
    """

    def __init__(
        self,
        elements: List[Element] | Dict[str, List[Element]],
        role_names: Tuple[str, ...],
        display_engine: DisplayEngine | None = None,
    ):
        super().__init__(elements, display_engine)
        self._role_names = role_names
        self._role_elements: Dict[str, List[Element]] = {}
        self._set_role_elements()

    def _set_role_elements(self):
        """Extract elements by their role metadata."""
        role_elements = defaultdict(list)

        for etype, e_list in self.elements.items():
            for e in e_list:
                role = e.metadata.get(ROLE_COL_NAME)
                if role in self._role_names:
                    role_elements[role].append(e)

        # Validate all roles have at least one element
        for role in self._role_names:
            if role not in role_elements or len(role_elements[role]) == 0:
                raise RuntimeError(f"No element found with role '{role}'")

        self._role_elements = dict(role_elements)

    @property
    def role_names(self) -> Tuple[str, ...]:
        """Return the configured role names."""
        return self._role_names

    def get_role(self, role: str) -> List[Element]:
        """Get elements by role name."""
        if role not in self._role_elements:
            raise KeyError(f"Unknown role: {role}. Available: {self._role_names}")
        return self._role_elements[role]

    def get_role_data(self, role: str) -> List[ELEMENT_DATA_TYPE]:
        """Get element data by role name."""
        return [e.data for e in self.get_role(role)]

    @property
    def data(self) -> Dict[str, List[ELEMENT_DATA_TYPE]]:
        """Return data from all elements keyed by role name."""
        return {role: [e.data for e in elems] for role, elems in self._role_elements.items()}

    def __getattr__(self, name: str):
        """Allow dynamic access via role names (e.g., sample.image, sample.bboxes)."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        if "_role_elements" in self.__dict__ and name in self._role_elements:
            return self._role_elements[name]

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def transform(
        self,
        transform: SampleTransform,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
        display_engine: DisplayEngine | None = None,
    ) -> Self:
        transformed_sample = super().transform(transform, cache_mechanisms, display_engine)
        return self.from_sample(transformed_sample, self._role_names)

    @classmethod
    def from_sample(
        cls,
        sample: Sample,
        role_names: Tuple[str, ...],
    ) -> Self:
        return cls(sample._elements, role_names, sample._display_engine)
