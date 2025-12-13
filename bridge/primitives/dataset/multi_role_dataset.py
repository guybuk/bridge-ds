from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Sequence, Tuple

import pandas as pd
from typing_extensions import Self

from bridge.primitives.dataset.dataset import Dataset
from bridge.primitives.sample.multi_role_sample import MultiRoleSample
from bridge.utils.constants import ELEMENT_COLS, INDICES, ROLE_COL_NAME

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism
    from bridge.primitives.element.element import Element
    from bridge.primitives.sample.transform import SampleTransform


class MultiRoleDataset(Dataset):
    """
    A dataset with named element roles per sample.

    Each sample can have any number of elements per role. Roles are accessed
    by name via properties or dynamic attribute access.

    Example:
        # Detection dataset: 1 image, N bboxes per sample
        ds = MultiRoleDataset.from_dict({
            "image": image_elements,
            "bboxes": bbox_elements,
        })
        ds.image           # DataFrame of image elements
        ds.bboxes          # DataFrame of bbox elements
        ds.get_role("image")  # Same as above
        ds.iget(0)         # MultiRoleSample with .image and .bboxes attributes

        # Translation dataset: 1 source, 1 target per sample
        ds = MultiRoleDataset.from_dict({
            "english": english_elements,
            "french": french_elements,
        })
    """

    def __init__(
        self,
        element_groups: Dict[str, pd.DataFrame],
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ):
        role_names = tuple(element_groups.keys())
        assert len(role_names) >= 2, "Must have at least 2 roles"
        assert len(role_names) == len(set(role_names)), "Role names must be unique"

        # Validate no overlapping element IDs across roles
        all_element_ids = []
        for role, df in element_groups.items():
            ids = df.index.get_level_values(ELEMENT_COLS.ID).tolist()
            all_element_ids.extend(ids)
        assert len(all_element_ids) == len(set(all_element_ids)), "Element IDs must be unique across all roles"

        self._role_names = role_names

        # Mark each DataFrame with its role and concatenate
        dfs = []
        for role, df in element_groups.items():
            df = df.copy()
            df[ROLE_COL_NAME] = role
            dfs.append(df)
        elements = pd.concat(dfs)

        super().__init__(elements, display_engine, cache_mechanisms)

    @property
    def role_names(self) -> Tuple[str, ...]:
        """Return the configured role names."""
        return self._role_names

    def get_role(self, role: str) -> pd.DataFrame:
        """Get DataFrame of elements for a specific role."""
        if role not in self._role_names:
            raise KeyError(f"Unknown role: {role}. Available: {self._role_names}")
        return (
            self._elements.loc[self._elements[ROLE_COL_NAME] == role]
            .dropna(axis="columns", how="all")
            .drop(columns=ROLE_COL_NAME)
        )

    def __getattr__(self, name: str) -> pd.DataFrame:
        """Allow dynamic access via role names (e.g., ds.image, ds.bboxes)."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        if "_role_names" in self.__dict__ and name in self._role_names:
            return self.get_role(name)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def iget(self, index: int) -> MultiRoleSample:
        return MultiRoleSample.from_sample(super().iget(index), self._role_names)

    def get(self, sample_id: Hashable) -> MultiRoleSample:
        return MultiRoleSample.from_sample(super().get(sample_id), self._role_names)

    def select_by_role(
        self,
        role: str,
        selector: Callable[..., Sequence],
    ) -> "MultiRoleDataset":
        """
        Select samples based on a condition applied to a specific role.

        The selector receives DataFrames for all roles as keyword arguments.
        """
        role_dfs = {r: self.get_role(r) for r in self._role_names}
        selected = selector(**role_dfs)
        target_df = role_dfs[role].loc[selected]
        selected_sample_ids = target_df.index.get_level_values(ELEMENT_COLS.SAMPLE_ID)

        new_groups = {}
        for r in self._role_names:
            df = role_dfs[r]
            new_groups[r] = df.loc[df.index.get_level_values(ELEMENT_COLS.SAMPLE_ID).isin(selected_sample_ids)]

        return MultiRoleDataset(
            new_groups,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def assign_to_role(
        self,
        role: str,
        **kwargs: Callable[..., Sequence],
    ) -> "MultiRoleDataset":
        """
        Assign new columns to a specific role's elements.

        Each kwarg is a callable that receives DataFrames for all roles as keyword arguments.
        """
        role_dfs = {r: self.get_role(r) for r in self._role_names}
        values_dict = {name: fn(**role_dfs) for name, fn in kwargs.items()}
        new_df = role_dfs[role].assign(**values_dict)

        new_groups = dict(role_dfs)
        new_groups[role] = new_df

        return MultiRoleDataset(
            new_groups,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def sort_by_role(self, role: str, by: str, ascending: bool = True) -> "MultiRoleDataset":
        """Sort dataset by a column in a specific role's elements."""
        role_dfs = {r: self.get_role(r) for r in self._role_names}
        role_dfs[role] = role_dfs[role].sort_values(by=by, ascending=ascending)

        return MultiRoleDataset(
            role_dfs,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def transform_samples(
        self,
        transform: SampleTransform,
        map_fn=map,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
        display_engine: DisplayEngine | None = None,
    ) -> "MultiRoleDataset":
        ds = super().transform_samples(
            transform, map_fn=map_fn, cache_mechanisms=cache_mechanisms, display_engine=display_engine
        )

        new_groups = {}
        for role in self._role_names:
            new_groups[role] = (
                ds.elements.loc[ds.elements[ROLE_COL_NAME] == role]
                .dropna(axis="columns", how="all")
                .drop(columns=ROLE_COL_NAME)
            )

        return MultiRoleDataset(
            new_groups,
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )

    def __repr__(self) -> str:
        lens_dict = {"n_samples": len(self), "roles": list(self._role_names)}
        for role in self._role_names:
            role_df = self._elements[self._elements[ROLE_COL_NAME] == role]
            lens_dict[f"n_{role}"] = len(role_df)
        return "MultiRoleDataset: " + str(lens_dict)

    @classmethod
    def from_dict(
        cls,
        element_lists: Dict[str, List[Element]],
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> Self:
        """
        Create MultiRoleDataset from a dict of element lists.

        Args:
            element_lists: Dict mapping role names to lists of Elements
            display_engine: Display engine for visualization
            cache_mechanisms: Cache mechanisms by element type
        """
        element_groups = {}
        for role, elements in element_lists.items():
            records = [e.to_dict() for e in elements]
            element_groups[role] = pd.DataFrame(records).set_index(INDICES)

        return cls(
            element_groups,
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )
