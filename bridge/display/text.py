from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import pandas as pd
import panel as pn

from bridge.display import DisplayEngine
from bridge.primitives.dataset import Dataset
from bridge.primitives.sample import Sample

if TYPE_CHECKING:
    from bridge.primitives.element.element import Element


class Panel(DisplayEngine[Dataset, Sample]):
    def show_element(self, element: Element, element_plot_kwargs: Dict[str, Any] | None = None):
        if element.etype == "class_label":
            return pn.pane.Markdown(element.to_pd_series().to_frame().T.to_markdown())
        elif element.etype == "text":
            return pn.pane.Markdown(element.data)
        else:
            raise NotImplementedError()

    def show_sample(
        self,
        sample: Sample,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
    ):
        class_labels = sample.elements.get("class_label", [])
        annotations_md = pd.DataFrame([ann.to_pd_series() for ann in class_labels]).to_markdown()
        text_elements = sample.elements.get("text", [])
        text_data = text_elements[0].data if text_elements else ""
        text_display = pn.pane.Markdown(text_data)
        return pn.Column("# Sample Text:", text_display, "# Annotations Data:", annotations_md)

    def show_dataset(
        self,
        dataset: Dataset,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
        dataset_plot_kwargs: Dict[str, Any] | None = None,
    ):
        sample_ids = dataset.sample_ids
        sample_ids_wig = pn.widgets.DiscreteSlider(name="Sample ID", options=sample_ids, value=sample_ids[0])

        @pn.depends(sample_ids_wig.param.value)
        def plot_sample_by_widget(sample_id):
            return self.show_sample(dataset.get(sample_id), element_plot_kwargs, sample_plot_kwargs)

        return pn.Column(sample_ids_wig, plot_sample_by_widget)
