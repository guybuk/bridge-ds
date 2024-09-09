from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import pandas as pd

from bridge.display.display_engine import DisplayEngine
from bridge.primitives.sample import Sample
from bridge.utils import optional_dependencies

if TYPE_CHECKING:
    from bridge.primitives.dataset import Dataset
    from bridge.primitives.element.element import Element
    from bridge.utils.data_objects import BoundingBox


class PanelDetectionClassification(DisplayEngine):
    def __init__(self, bbox_format: str = "xyxy") -> None:
        assert bbox_format in ["xyxy", "xywh", "cxcywh"]
        self._bbox_format = bbox_format

    def show_element(self, element: Element, element_plot_kwargs: Dict[str, Any] | None = None):
        self._validate_dependencies()
        etype = element.etype
        if etype == "image":
            plot = self._plot_single_image(element)
        elif etype == "bbox":
            plot = self._plot_single_bbox(element)
        else:
            raise NotImplementedError(f"Invalid etype: {etype}")
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

        imgs = [self._plot_single_image(element) for element in sample.elements["image"]]
        if "bbox" in sample.elements:
            bboxes = self._plot_list_of_bbox_or_class_labels(sample.elements["bbox"])
        else:
            bboxes = hv.Overlay()
        if "class_label" in sample.elements:
            class_labels = self._plot_list_of_bbox_or_class_labels(sample.elements["class_label"])
        else:
            class_labels = hv.Overlay()
        for i in range(len(imgs)):
            imgs[i] = imgs[i] * bboxes * class_labels
        return hv.Layout(imgs)

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

        data: np.ndarray = element.data
        etype = element.etype
        h, w = data.shape[0], data.shape[1]
        if len(data.shape) == 2:
            # handling special case of grayscale images
            data = data[:, :, np.newaxis].repeat(3, axis=-1)
        img = hv.RGB(data[::-1, :, :], bounds=(0, 0, w, h)).opts(**self._default_kwargs(etype))
        return img

    def _plot_single_bbox(self, element: Element):
        import holoviews as hv

        etype = element.etype
        data: BoundingBox = element.data
        xyxy = self._extract_bbox_coords(data)
        plot = hv.Rectangles([[*xyxy, str(data.class_label)]], label=str(data.class_label)).opts(
            **self._default_kwargs(etype)
        )
        return plot

    def _plot_list_of_bbox_or_class_labels(self, elements: List[Element]):
        import holoviews as hv

        rectangle_list = []
        for element in elements:
            data: BoundingBox = element.data
            if element.etype == "bbox":
                xyxy = self._extract_bbox_coords(data)
                cl = data.class_label
            elif element.etype == "class_label":  # assume cls
                xyxy = [np.nan, np.nan, np.nan, np.nan]
                cl = data
            else:
                raise NotImplementedError()

            rectangle_list.append({"x0": xyxy[0], "y0": xyxy[1], "x1": xyxy[2], "y1": xyxy[3], "class": str(cl)})
        hv_df = pd.DataFrame.from_records(rectangle_list)
        if len(hv_df) == 0:
            return None
        else:
            plots = []
            for i, group in hv_df.groupby("class"):
                p = hv.Rectangles(group, label=i)
                plots.append(p)
            plots = hv.Overlay(plots).opts(hv.opts.Rectangles(**self._default_kwargs("bbox")))
            return plots

    def _extract_bbox_coords(self, data):
        if self._bbox_format == "xyxy":
            xyxy = data.coords
        elif self._bbox_format == "cxcywh":
            x, y, w, h = data.coords
            xyxy = np.clip([x - w / 2, y - h / 2, x + w / 2, y + h / 2], a_min=0, a_max=None)
        elif self._bbox_format == "xywh":
            x_min, y_min, w, h = data.coords
            xyxy = np.clip([x_min, y_min, x_min + w, y_min + h], a_min=0, a_max=None)
        else:
            raise NotImplementedError()
        return xyxy

    @staticmethod
    def _default_kwargs(etype: str) -> Dict[str, Any]:
        import holoviews as hv

        if etype == "image":
            return dict(aspect="equal", invert_yaxis=True, legend_position="left", xaxis=None, yaxis=None)
        elif etype == "bbox":
            return dict(fill_alpha=0.0, line_width=3, line_color=hv.Cycle("Category20"))

    @staticmethod
    def _validate_dependencies():
        with optional_dependencies("raise"):
            import holoviews  # noqa
            import panel  # noqa

        assert (
            holoviews.Store.current_backend == "bokeh"
        ), f"Holoviews backend: {holoviews.Store.current_backend} is not supported."


class PanelVideo(DisplayEngine):
    def __init__(self, fps=10) -> None:
        self._fps = fps

    def show_element(self, element: Element, element_plot_kwargs: Dict[str, Any] | None = None):
        self._validate_dependencies()
        if element.etype != "video":
            raise ValueError(f"Invalid etype: {element.etype}. Expected 'video'.")

        frames = self._extract_frames(element.data)
        return self._create_video_player(frames)

    def show_sample(
        self,
        sample: Sample,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
    ):
        self._validate_dependencies()
        import panel as pn

        if "video" not in sample.elements or "class_label" not in sample.elements:
            raise ValueError("Sample must contain 'video' and 'class_label' elements.")

        video = sample.elements["video"][0]
        class_label = sample.elements["class_label"][0].data
        frames = self._extract_frames(video.data)

        video_player = self._create_video_player(frames)

        return pn.Column(pn.pane.Markdown(f"## Class Label: {class_label}", align="center"), video_player)

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
        if not sample_ids:
            raise ValueError("Dataset is empty.")

        sample_ids_wig = pn.widgets.DiscreteSlider(name="Sample ID", options=sample_ids, value=sample_ids[0])

        @pn.depends(sample_ids_wig.param.value)
        def plot_sample_by_widget(sample_id):
            return self.show_sample(
                dataset.get(sample_id), element_plot_kwargs=element_plot_kwargs, sample_plot_kwargs=sample_plot_kwargs
            )

        return pn.Column(sample_ids_wig, plot_sample_by_widget)

    @staticmethod
    def _validate_dependencies():
        with optional_dependencies("raise"):
            import panel  # noqa
            import cv2  # noqa
            import PIL  # noqa

    @staticmethod
    def _extract_frames(video):
        import cv2
        from PIL import Image

        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        video.release()

        if not frames:
            raise ValueError("No frames were extracted from the video.")

        return frames

    def _create_video_player(self, frames):
        import panel as pn

        player_wig = pn.widgets.Player(
            name="Player", start=0, end=len(frames) - 1, value=0, loop_policy="loop", interval=1000 // self._fps
        )

        def show_frame(step):
            return pn.pane.Image(frames[step])

        fn = pn.param.ParamFunction(pn.bind(show_frame, player_wig.param.value), inplace=True, align="center")

        return pn.Column(fn, player_wig)
