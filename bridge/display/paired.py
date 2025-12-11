from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from bridge.display.display_engine import DisplayEngine
from bridge.utils import optional_dependencies

if TYPE_CHECKING:
    from bridge.primitives.dataset import MultiRoleDataset
    from bridge.primitives.element.element import Element
    from bridge.primitives.sample import MultiRoleSample


class PairedPanel(DisplayEngine["MultiRoleDataset", "MultiRoleSample"]):
    """
    Display engine for MultiRoleDataset/MultiRoleSample showing elements side-by-side.

    Designed for datasets with two roles, displaying each role in a column.

    Supports:
    - Vision-vision pairs (two images)
    - Text-text pairs (two text blocks)
    - Mixed pairs (text + image)
    """

    def __init__(self, bbox_format: str = "xyxy"):
        self._bbox_format = bbox_format

    def _render_element(self, element: Element, role_name: str):
        """Render a single element based on its etype."""
        import panel as pn

        etype = element.etype

        if etype == "image":
            return self._render_image(element)
        elif etype == "text":
            return pn.pane.Markdown(f"{element.data}")
        elif etype == "class_label":
            return pn.pane.Markdown(f"**Label:** {element.data}")
        else:
            return pn.pane.Markdown(f"**{etype}:** {str(element.data)[:500]}")

    def _render_image(self, element: Element):
        """Render an image element using holoviews."""
        import holoviews as hv

        if element.category == "image":
            data: np.ndarray = element.data
        elif element.category == "torch":
            data: np.ndarray = element.data.permute(1, 2, 0).numpy()
        else:
            data: np.ndarray = np.array(element.data)

        h, w = data.shape[0], data.shape[1]
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis].repeat(3, axis=-1)

        img = hv.RGB(data[::-1, :, :], bounds=(0, 0, w, h)).opts(
            aspect="equal", invert_yaxis=True, xaxis=None, yaxis=None
        )
        return img

    def show_element(
        self,
        element: Element,
        element_plot_kwargs: Dict[str, Any] | None = None,
    ):
        self._validate_dependencies()
        return self._render_element(element, "element")

    def show_sample(
        self,
        sample: MultiRoleSample,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
    ):
        """Display sample with side-by-side layout for each role."""
        self._validate_dependencies()
        import panel as pn

        role_names = sample.role_names

        columns = []
        for role in role_names:
            role_elements = sample.get_role(role)
            # Render all elements for this role
            rendered_elements = [self._render_element(e, role) for e in role_elements]

            role_panel = pn.Column(
                pn.pane.Markdown(f"### {role.replace('_', ' ').title()}"),
                *rendered_elements,
            )
            columns.append(role_panel)

        return pn.Row(*columns)

    def show_dataset(
        self,
        dataset: MultiRoleDataset,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
        dataset_plot_kwargs: Dict[str, Any] | None = None,
    ):
        """Display dataset with sample navigation slider."""
        self._validate_dependencies()
        import panel as pn

        sample_ids = dataset.sample_ids
        sample_ids_widget = pn.widgets.DiscreteSlider(
            name="Sample ID",
            options=sample_ids,
            value=sample_ids[0],
        )

        @pn.depends(sample_ids_widget.param.value)
        def plot_sample_by_widget(sample_id):
            return self.show_sample(
                dataset.get(sample_id),
                element_plot_kwargs=element_plot_kwargs,
                sample_plot_kwargs=sample_plot_kwargs,
            )

        return pn.Column(sample_ids_widget, plot_sample_by_widget)

    @staticmethod
    def _validate_dependencies():
        with optional_dependencies("raise"):
            import holoviews  # noqa
            import panel  # noqa

        assert (
            holoviews.Store.current_backend == "bokeh"
        ), f"Holoviews backend: {holoviews.Store.current_backend} is not supported."
