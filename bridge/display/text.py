from typing import TYPE_CHECKING, Any, Dict

import pandas as pd
import panel as pn

from bridge.primitives.sample.singular_sample import SingularSample

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.dataset import SingularDataset
    from bridge.primitives.element.element import Element


class TextClassification(DisplayEngine[SingularDataset, SingularSample]):
    def show_element(self, element: Element, element_plot_kwargs: Dict[str, Any] | None = None):
        if element.etype == "class_label":
            return pn.pane.Markdown(element.to_pd_series().to_frame().T.to_markdown())
        elif element.etype == "text":
            return pn.pane.Markdown(element.data)
        else:
            raise NotImplementedError()

    def show_sample(
        self,
        sample: SingularSample,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
    ):
        annotations_md = pd.DataFrame([ann.to_pd_series() for ann in sample.annotations["class_label"]]).to_markdown()
        text_display = pn.pane.Markdown(sample.data)
        return pn.Column("# Sample Text:", text_display, "# Annotations Data:", annotations_md)

    def show_dataset(
        self,
        dataset: SingularDataset,
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


# class Panel(DisplayEngine):
#     def show_element(self, element: Element, element_plot_kwargs: Dict[str, Any] | None = None):
#         if element.etype == "class_label":
#             return self._show_class_label(element, element_plot_kwargs)
#         elif element.etype == "text":
#             return self._show_text(element, element_plot_kwargs)
#         else:
#             raise NotImplementedError()
#
#     def show_sample(
#         self,
#         sample: Sample,
#         element_plot_kwargs: Dict[str, Any] | None = None,
#         sample_plot_kwargs: Dict[str, Any] | None = None,
#     ):
#         annotations_md = pd.DataFrame(
#             [ann.to_pd_series() for ann in sample.elements["class_label"]]
#         ).to_markdown()
#         text_display = self.show_element(sample.elements["text"][0])
#         return pn.Column("# Sample Text:", text_display, "# Annotations Data:", annotations_md)
#
#     def show_dataset(
#         self,
#         dataset: Dataset,
#         element_plot_kwargs: Dict[str, Any] | None = None,
#         sample_plot_kwargs: Dict[str, Any] | None = None,
#         dataset_plot_kwargs: Dict[str, Any] | None = None,
#     ):
#         sample_ids = dataset.sample_ids
#         sample_ids_wig = pn.widgets.DiscreteSlider(name="Sample ID", options=sample_ids, value=sample_ids[0])
#
#         @pn.depends(sample_ids_wig.param.value)
#         def plot_sample_by_widget(sample_id):
#             return self.show_sample(dataset.get(sample_id), element_plot_kwargs, sample_plot_kwargs)
#
#         return pn.Column(sample_ids_wig, plot_sample_by_widget)
#
#     @staticmethod
#     def _show_class_label(element: Element, element_plot_kwargs: Dict[str, Any] | None):
#         return pn.pane.Markdown(element.to_pd_series().to_frame().T.to_markdown())
#
#     @staticmethod
#     def _show_text(element: Element, element_plot_kwargs: Dict[str, Any] | None):
#         return pn.pane.Markdown(element.data)
