from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import pandas as pd
import panel as pn

from bridge.display import DisplayEngine
from bridge.primitives.dataset import Dataset
from bridge.primitives.sample import Sample

if TYPE_CHECKING:
    from bridge.primitives.element.element import Element


class TextClassificationPanelEngine(DisplayEngine[Dataset, Sample]):
    """Renders text + class_label samples.

    Expected sample shape:
      - role ``text`` (etype ``text``) — required, displayed as Markdown
      - role ``class_label`` (etype ``class_label``) — optional, rendered as a Markdown table alongside the text
    """

    TEXT_ROLE = "text"
    CLASS_LABEL_ROLE = "class_label"

    def show_element(self, element: Element, element_plot_kwargs: Dict[str, Any] | None = None):
        if element.etype == "class_label":
            return pn.pane.Markdown(element.to_pd_series().to_frame().T.to_markdown())
        elif element.etype == "text":
            return pn.pane.Markdown(element.data)
        else:
            raise NotImplementedError(f"TextClassificationPanelEngine cannot render etype={element.etype}")

    def show_sample(
        self,
        sample: Sample,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
    ):
        if self.TEXT_ROLE not in sample.elements:
            raise ValueError(
                f"TextClassificationPanelEngine requires a '{self.TEXT_ROLE}' role on the sample"
            )

        text_display = pn.pane.Markdown(sample.one(self.TEXT_ROLE).data)
        components: list = ["# Sample Text:", text_display]
        if self.CLASS_LABEL_ROLE in sample.elements:
            label_elements = sample.elements[self.CLASS_LABEL_ROLE]
            annotations_md = pd.DataFrame([e.to_pd_series() for e in label_elements]).to_markdown()
            components.extend(["# Annotations Data:", annotations_md])
        return pn.Column(*components)

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
