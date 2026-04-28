from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Sequence

import pandas as pd
from typing_extensions import Self

from bridge.primitives.dataset.dataset import Dataset
from bridge.primitives.element.data.element_store import ElementStore
from bridge.primitives.sample.singular_sample import SingularSample
from bridge.utils.constants import ELEMENT_COLS, INDICES, IS_SAMPLE_COL_NAME

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism
    from bridge.primitives.element.element import Element
    from bridge.primitives.sample.transform import SampleTransform


class SingularDataset(Dataset):
    """
    An annotated dataset is a popular use-case where a dataset is composed of samples (images, text, audio, video)
    and annotations (bboxes, captions, frames, labels)
    This implementation exposes `ds.elements` as two different views: `ds.samples` and `ds.annotations`, and the
    respective `select_<samples/annotations>`, `sort_<examples/annotations>`, `assign_<examples/annotations>` methods.
    """

    def __init__(
        self,
        samples: pd.DataFrame,
        annotations: pd.DataFrame,
        store: ElementStore | None = None,
        display_engine: DisplayEngine = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ):
        assert (
            len(
                samples.index.get_level_values(ELEMENT_COLS.ID).intersection(
                    annotations.index.get_level_values(ELEMENT_COLS.ID)
                )
            )
            == 0
        ), "samples and annotations can't share ids"

        samples[IS_SAMPLE_COL_NAME] = True
        annotations[IS_SAMPLE_COL_NAME] = False
        elements = pd.concat([samples, annotations])
        super().__init__(elements, store=store, display_engine=display_engine, cache_mechanisms=cache_mechanisms)

    @property
    def samples(self) -> pd.DataFrame:
        sub = self._df.loc[self._df[IS_SAMPLE_COL_NAME]]
        return (
            self._join_locations(sub)
            .dropna(axis="columns", how="all")
            .drop(columns=IS_SAMPLE_COL_NAME)
        )

    @property
    def annotations(self) -> pd.DataFrame:
        sub = self._df.loc[~self._df[IS_SAMPLE_COL_NAME]]
        return (
            self._join_locations(sub)
            .dropna(axis="columns", how="all")
            .drop(columns=IS_SAMPLE_COL_NAME)
        )

    def iget(self, index: int) -> SingularSample:
        return SingularSample.from_sample(super().iget(index))

    def get(self, sample_id: Hashable) -> SingularSample:
        return SingularSample.from_sample(super().get(sample_id))

    def select_samples(self, selector: Callable[[pd.DataFrame, pd.DataFrame], Sequence]):
        samples = self.samples
        annotations = self.annotations
        selected = selector(samples, annotations)
        new_samples = samples.loc[selected]
        new_annotations = annotations.loc[
            annotations.index.get_level_values(ELEMENT_COLS.SAMPLE_ID).isin(
                new_samples.index.get_level_values(ELEMENT_COLS.SAMPLE_ID)
            )
        ]
        return SingularDataset(
            new_samples,
            new_annotations,
            store=self._store,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def select_annotations(self, selector: Callable[[pd.DataFrame, pd.DataFrame], Sequence]):
        samples = self.samples
        annotations = self.annotations
        selected = selector(samples, annotations)
        new_annotations = annotations.loc[selected]

        return SingularDataset(
            samples,
            new_annotations,
            store=self._store,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def assign_samples(self, **kwargs: Callable[[pd.DataFrame, pd.DataFrame], Sequence]) -> Self:
        samples = self.samples
        annotations = self.annotations
        values_dict = {name: assign_fn(samples, annotations) for name, assign_fn in kwargs.items()}
        new_samples = samples.assign(**values_dict)
        return SingularDataset(
            new_samples,
            annotations,
            store=self._store,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def assign_annotations(self, **kwargs: Callable[[pd.DataFrame, pd.DataFrame], Sequence]) -> Self:
        samples = self.samples
        annotations = self.annotations
        values_dict = {name: assign_fn(samples, annotations) for name, assign_fn in kwargs.items()}
        new_annotations = annotations.assign(**values_dict)
        return SingularDataset(
            samples,
            new_annotations,
            store=self._store,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def sort_samples(self, by: str, ascending: bool = True):
        samples = self.samples
        annotations = self.annotations
        new_samples = samples.sort_values(by=by, ascending=ascending)
        return SingularDataset(
            new_samples,
            annotations,
            store=self._store,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def sort_annotations(self, by: str, ascending: bool = True):
        samples = self.samples
        annotations = self.annotations
        new_annotations = annotations.sort_values(by=by, ascending=ascending)
        return SingularDataset(
            samples,
            new_annotations,
            store=self._store,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def transform_samples(
        self,
        transform: SampleTransform,
        map_fn=map,
        cache_mechanisms: Dict[str, CacheMechanism] | None = None,
        display_engine: DisplayEngine | None = None,
    ) -> Self:
        ds = super().transform_samples(
            transform, map_fn=map_fn, cache_mechanisms=cache_mechanisms, display_engine=display_engine
        )
        full = ds.elements
        samples = (
            full.loc[full[IS_SAMPLE_COL_NAME]]
            .dropna(axis="columns", how="all")
            .drop(columns=IS_SAMPLE_COL_NAME)
        )
        annotations = (
            full.loc[~full[IS_SAMPLE_COL_NAME]]
            .dropna(axis="columns", how="all")
            .drop(columns=IS_SAMPLE_COL_NAME)
        )
        return SingularDataset(samples, annotations, display_engine=display_engine)

    @classmethod
    def from_lists(
        cls,
        samples_list: List[Element],
        annotations_list: List[Element],
        display_engine: DisplayEngine = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> Self:
        store = ElementStore()
        sample_records = []
        for s in samples_list:
            sample_records.append(s.to_dict())
            store.set(s.id, s._load_mechanism)
        annotation_records = []
        for a in annotations_list:
            annotation_records.append(a.to_dict())
            store.set(a.id, a._load_mechanism)

        samples_df = pd.DataFrame(sample_records).set_index(INDICES)
        annotations_df = pd.DataFrame(annotation_records).set_index(INDICES)

        return cls(samples_df, annotations_df, store=store, display_engine=display_engine, cache_mechanisms=cache_mechanisms)
