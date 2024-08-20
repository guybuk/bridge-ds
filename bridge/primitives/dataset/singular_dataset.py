from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Hashable, List, Sequence

import pandas as pd
from typing_extensions import Self

from bridge.primitives.dataset.dataset import Dataset
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
        super().__init__(elements, display_engine, cache_mechanisms)

    @property
    def samples(self) -> pd.DataFrame:
        return (
            self._elements.loc[self._elements[IS_SAMPLE_COL_NAME]]
            .dropna(axis="columns", how="all")
            .drop(columns=IS_SAMPLE_COL_NAME)
        )

    @property
    def annotations(self) -> pd.DataFrame:
        return (
            self._elements.loc[~self._elements[IS_SAMPLE_COL_NAME]]
            .dropna(axis="columns", how="all")
            .drop(columns=IS_SAMPLE_COL_NAME)
        )

    def iget(self, index: int) -> SingularSample:
        return SingularSample.from_sample(super().iget(index))

    def get(self, sample_id: Hashable) -> SingularSample:
        return SingularSample.from_sample(super().get(sample_id))

    def select_samples(self, selector: Callable[[pd.DataFrame, pd.DataFrame], Sequence]):
        selected = selector(self.samples, self.annotations)
        new_samples = self.samples.loc[selected]
        new_annotations = self.annotations.pipe(
            lambda df_: df_.loc[
                df_.index.get_level_values(ELEMENT_COLS.SAMPLE_ID).isin(
                    new_samples.index.get_level_values(ELEMENT_COLS.SAMPLE_ID)
                )
            ]
        )
        return SingularDataset(
            new_samples,
            new_annotations,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def select_annotations(self, selector: Callable[[pd.DataFrame, pd.DataFrame], Sequence]):
        selected = selector(self.samples, self.annotations)
        new_annotations = self.annotations.loc[selected]

        return SingularDataset(
            self.samples,
            new_annotations,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def assign_samples(self, **kwargs: Callable[[pd.DataFrame, pd.DataFrame], Sequence]) -> Self:
        values_dict = {name: assign_fn(self.samples, self.annotations) for name, assign_fn in kwargs.items()}
        new_samples = self.samples.assign(**values_dict)
        return SingularDataset(
            new_samples,
            self.annotations,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def assign_annotations(self, **kwargs: Callable[[pd.DataFrame, pd.DataFrame], Sequence]) -> Self:
        values_dict = {name: assign_fn(self.samples, self.annotations) for name, assign_fn in kwargs.items()}
        new_annotations = self.annotations.assign(**values_dict)
        return SingularDataset(
            self.samples,
            new_annotations,
            display_engine=self._display_engine,
            cache_mechanisms=self._cache_mechanisms,
        )

    def sort_samples(self, by: str, ascending: bool = True):
        new_samples = self.samples.sort_values(by=by, ascending=ascending)
        return SingularDataset(
            new_samples, self.annotations, display_engine=self._display_engine, cache_mechanisms=self._cache_mechanisms
        )

    def sort_annotations(self, by: str, ascending: bool = True):
        new_annotations = self.annotations.sort_values(by=by, ascending=ascending)
        return SingularDataset(
            self.samples, new_annotations, display_engine=self._display_engine, cache_mechanisms=self._cache_mechanisms
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
        samples = (
            ds.elements.loc[ds.elements[IS_SAMPLE_COL_NAME]]
            .dropna(axis="columns", how="all")
            .drop(columns=IS_SAMPLE_COL_NAME)
        )
        annotations = (
            ds.elements.loc[~ds.elements[IS_SAMPLE_COL_NAME]]
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
        sample_records = [s.to_dict() for s in samples_list]
        annotation_records = [a.to_dict() for a in annotations_list]

        samples_df = pd.DataFrame(sample_records).set_index(INDICES)
        annotations_df = pd.DataFrame(annotation_records).set_index(INDICES)

        return cls(samples_df, annotations_df, display_engine=display_engine, cache_mechanisms=cache_mechanisms)
