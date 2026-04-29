from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from bridge.display.display_engine import DisplayEngine
from bridge.utils import optional_dependencies

if TYPE_CHECKING:
    from bridge.primitives.dataset import Dataset
    from bridge.primitives.element.element import Element
    from bridge.primitives.sample import Sample


class CaptionedImagesPanelEngine(DisplayEngine):
    """Renders image + caption text samples.

    Expected sample shape:
      - role ``image`` (etype ``image``) — required
      - role ``caption`` (etype ``text``) — required, displayed as Markdown alongside the image
    """

    IMAGE_ROLE = "image"
    CAPTION_ROLE = "caption"

    def show_element(self, element: Element, element_plot_kwargs: Dict[str, Any] | None = None):
        self._validate_dependencies()
        if element.etype == "image":
            return self._plot_single_image(element)
        if element.etype == "text":
            import panel as pn

            return pn.pane.Markdown(element.data)
        raise NotImplementedError(f"CaptionedImagesPanelEngine cannot render etype={element.etype}")

    def show_sample(
        self,
        sample: Sample,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
    ):
        self._validate_dependencies()
        import panel as pn

        for role in (self.IMAGE_ROLE, self.CAPTION_ROLE):
            if role not in sample.elements:
                raise ValueError(
                    f"CaptionedImagesPanelEngine requires '{self.IMAGE_ROLE}' and "
                    f"'{self.CAPTION_ROLE}' roles; missing '{role}'"
                )

        img_pane = pn.pane.HoloViews(self._plot_single_image(sample.one(self.IMAGE_ROLE)))
        caption_pane = pn.pane.Markdown(f"**Caption:** {sample.one(self.CAPTION_ROLE).data}")
        return pn.Row(img_pane, caption_pane)

    def show_dataset(
        self,
        dataset: Dataset,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
        dataset_plot_kwargs: Dict[str, Any] | None = None,
    ):
        self._validate_dependencies()
        import panel as pn

        sample_ids = dataset.sample_ids
        sample_ids_wig = pn.widgets.DiscreteSlider(name="Sample ID", options=sample_ids, value=sample_ids[0])

        @pn.depends(sample_ids_wig.param.value)
        def plot_sample_by_widget(sample_id):
            return self.show_sample(
                dataset.get(sample_id),
                element_plot_kwargs=element_plot_kwargs,
                sample_plot_kwargs=sample_plot_kwargs,
            )

        return pn.Column(sample_ids_wig, plot_sample_by_widget)

    def _plot_single_image(self, element: Element):
        import holoviews as hv

        if element.encoding == "jpeg":
            data: np.ndarray = element.data
        elif element.encoding == "pt":
            data: np.ndarray = element.data.permute(1, 2, 0).numpy()
        else:
            raise NotImplementedError(f"Unsupported encoding for image: {element.encoding}")

        h, w = data.shape[0], data.shape[1]
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis].repeat(3, axis=-1)
        return hv.RGB(data[::-1, :, :], bounds=(0, 0, w, h)).opts(
            aspect="equal", invert_yaxis=True, xaxis=None, yaxis=None
        )

    @staticmethod
    def _validate_dependencies():
        with optional_dependencies("raise"):
            import holoviews  # noqa
            import panel  # noqa

        assert (
            holoviews.Store.current_backend == "bokeh"
        ), f"Holoviews backend: {holoviews.Store.current_backend} is not supported."
