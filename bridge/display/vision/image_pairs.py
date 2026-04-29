from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from bridge.display.display_engine import DisplayEngine
from bridge.utils import optional_dependencies

if TYPE_CHECKING:
    from bridge.primitives.dataset import Dataset
    from bridge.primitives.element.element import Element
    from bridge.primitives.sample import Sample


class ImagePairsPanelEngine(DisplayEngine):
    """Renders source/target image-pair samples side-by-side.

    Expected sample shape:
      - role ``source`` (etype ``image``) — required
      - role ``target`` (etype ``image``) — required
    """

    SOURCE_ROLE = "source"
    TARGET_ROLE = "target"

    def show_element(self, element: Element, element_plot_kwargs: Dict[str, Any] | None = None):
        self._validate_dependencies()
        if element.etype != "image":
            raise NotImplementedError(
                f"ImagePairsPanelEngine only renders image elements, got etype={element.etype}"
            )
        plot = self._plot_single_image(element)
        if element_plot_kwargs:
            plot = plot.opts(**element_plot_kwargs)
        return plot

    def show_sample(
        self,
        sample: Sample,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
    ):
        self._validate_dependencies()
        import holoviews as hv

        for role in (self.SOURCE_ROLE, self.TARGET_ROLE):
            if role not in sample.elements:
                raise ValueError(
                    f"ImagePairsPanelEngine requires both '{self.SOURCE_ROLE}' and "
                    f"'{self.TARGET_ROLE}' roles; missing '{role}'"
                )

        source_plot = self._plot_single_image(sample.one(self.SOURCE_ROLE)).opts(title="source")
        target_plot = self._plot_single_image(sample.one(self.TARGET_ROLE)).opts(title="target")
        return hv.Layout([source_plot, target_plot]).cols(2)

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
