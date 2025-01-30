from __future__ import annotations

import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import albumentations as A
import numpy as np

from bridge.display import DisplayEngine
from bridge.primitives.element.data.cache_mechanism import CacheMechanism
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import Sample
from bridge.primitives.sample.singular_sample import SingularSample
from bridge.primitives.sample.transform.sample_transform import SampleTransform
from bridge.utils import optional_dependencies
from bridge.utils.constants import IS_SAMPLE_COL_NAME
from bridge.utils.data_objects import BoundingBox

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


class AlbumentationsCompose(SampleTransform):
    def __init__(
        self,
        albm_transforms: List[A.BasicTransform],
        bbox_format="pascal_voc",
        kp_format="xy",
    ) -> None:
        self._bbox_format = bbox_format
        self._kp_format = kp_format
        self._transforms = albm_transforms

    def __call__(
        self, sample: Sample, cache_mechanisms: Dict[str, CacheMechanism], display_engine: DisplayEngine | None
    ) -> Sample:
        self._verify_input_sample(sample)

        elements = copy.deepcopy(sample.elements)
        albm_dict = self._elements_to_albm(elements)
        compose = A.Compose(
            self._transforms,
            bbox_params=A.BboxParams(format=self._bbox_format, label_fields=["bbox_labels", "bbox_id"]),
            keypoint_params=A.KeypointParams(format=self._kp_format),
            is_check_shapes=False,
        )
        albm_dict = compose(**albm_dict)
        self._update_elements(elements, albm_dict, cache_mechanisms)
        sample = Sample(elements=elements, display_engine=display_engine)
        return sample

    def _elements_to_albm(self, elements: Dict[str, List[Element]]):
        albm_dict = {"image": elements["image"][0].data, "bboxes": [], "bbox_labels": [], "bbox_id": []}

        if "bbox" in elements:
            for bbox_element in elements["bbox"]:
                data: BoundingBox = bbox_element.data
                albm_dict["bboxes"].append(data.coords.tolist())
                albm_dict["bbox_labels"].append(data.class_label)
                albm_dict["bbox_id"].append(bbox_element.id)
        if "keypoint" in elements:
            raise NotImplementedError("Didn't fully implement keypoints in albumentations yet.")
        return dict(albm_dict)

    def _update_elements(
        self, elements: Dict[str, List[Element]], albm_dict: Dict[str, Any], cache_mechanisms: Dict[str, CacheMechanism]
    ):
        self._update_image(elements, albm_dict, cache_mechanisms)
        if "bbox" in elements:
            self._update_bboxes(elements, albm_dict, cache_mechanisms)

    def _update_bboxes(
        self, elements: Dict[str, List[Element]], albm_dict: Dict[str, Any], cache_mechanisms: Dict[str, CacheMechanism]
    ):
        del_idxs = []
        for i, element in enumerate(elements["bbox"]):
            try:
                idx = albm_dict["bbox_id"].index(element.id)
                bbox_coords = albm_dict["bboxes"][idx]
                class_label = albm_dict["bbox_labels"][idx]
                bbox = BoundingBox(bbox_coords, class_label)
                new_provider = cache_mechanisms["bbox"].store(element, bbox, as_category="obj")
                elements["bbox"][i] = element.copy(load_mechanism=new_provider)
            except ValueError:  # augmentation removed the bbox
                del_idxs.append(i)
        for i in sorted(del_idxs, reverse=True):
            del elements["bbox"][i]

    def _update_image(
        self, elements: Dict[str, List[Element]], albm_dict: Dict[str, Any], cache_mechanisms: Dict[str, CacheMechanism]
    ):
        img_data = albm_dict["image"]
        if isinstance(img_data, np.ndarray):
            new_category = "image"
        else:
            with optional_dependencies(error="raise"):
                import torch
            if isinstance(img_data, torch.Tensor):
                new_category = "torch"
            else:
                raise NotImplementedError(f"invalid data type: {type(img_data)}")
        provider = cache_mechanisms["image"].store(
            elements["image"][0], img_data, as_category=new_category, should_update_elements=False
        )
        elements["image"][0] = elements["image"][0].copy(load_mechanism=provider)

    def _verify_input_sample(self, sample: Sample):
        assert len(sample.elements["image"]) == 1, "AlbumentationCompose supports only a single image per sample"
        assert "keypoint" not in sample.elements, "AlbumentationsCompose does not support keypoints yet"
        assert "mask" not in sample.elements, "AlbumentationsCompose does not support masks yet"


class SliceImage:
    def __init__(
        self,
        slice_height: int,
        slice_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        bbox_format: str = "pascal_voc",
    ) -> None:
        super().__init__()
        self._slice_height = slice_height
        self._slice_width = slice_width
        self._overlap_height_ratio = overlap_height_ratio
        self._overlap_width_ratio = overlap_width_ratio
        self._image_height = image_height
        self._image_width = image_width
        self._bbox_format = bbox_format

    def __call__(
        self, sample: Sample, cache_mechanisms: Dict[str, CacheMechanism] | None, display_engine: DisplayEngine | None
    ) -> List[Sample]:
        image_element = sample.element
        metadata = image_element.metadata

        # Try to get dimensions from metadata first
        image_height = metadata.get("height")
        image_width = metadata.get("width")

        # Fall back to instance variables if not in metadata
        if image_height is None:
            image_height = self._image_height
        if image_width is None:
            image_width = self._image_width

        # Raise exception if dimensions still not found
        if image_height is None or image_width is None:
            raise ValueError("Image dimensions not found in metadata or instance variables")

        slice_bboxes = SliceImage.get_slice_bboxes(
            image_height=image_height,
            image_width=image_width,
            slice_height=self._slice_height,
            slice_width=self._slice_width,
            overlap_height_ratio=self._overlap_height_ratio,
            overlap_width_ratio=self._overlap_width_ratio,
        )

        # Create individual crop transforms for each slice
        crop_transforms = []
        for bbox in slice_bboxes:
            x_min, y_min, x_max, y_max = bbox
            crop = A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
            crop_transforms.append(AlbumentationsCompose([crop], bbox_format=self._bbox_format))

        # Apply each crop transform to create multiple samples
        samples = []
        for i, crop_transform in enumerate(crop_transforms):
            # Transform the sample
            transformed = sample.copy(
                new_sample_id=f"{sample.id}_slice_{i}", new_element_id_suffix=f"_slice_{i}"
            ).transform(crop_transform, cache_mechanisms, display_engine)

            samples.append(transformed)

        return samples

    @staticmethod
    def get_slice_bboxes(
        image_height: int,
        image_width: int,
        slice_height: int,
        slice_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
    ) -> List[List[int]]:
        """Slices `image_pil` in crops.
        Corner values of each slice will be generated using the `slice_height`,
        `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.

        Args:
            image_height (int): Height of the original image.
            image_width (int): Width of the original image.
            slice_height (int, optional): Height of each slice. Default None.
            slice_width (int, optional): Width of each slice. Default None.
            overlap_height_ratio(float): Fractional overlap in height of each
                slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
                overlap of 20 pixels). Default 0.2.
            overlap_width_ratio(float): Fractional overlap in width of each
                slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
                overlap of 20 pixels). Default 0.2.
            auto_slice_resolution (bool): if not set slice parameters such as slice_height and slice_width,
                it enables automatically calculate these params from image resolution and orientation.

        Returns:
            List[List[int]]: List of 4 corner coordinates for each N slices.
                [
                    [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                    ...
                    [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
                ]
        """
        slice_bboxes = []
        y_max = y_min = 0

        if slice_height and slice_width:
            y_overlap = int(overlap_height_ratio * slice_height)
            x_overlap = int(overlap_width_ratio * slice_width)
        else:
            raise ValueError("Compute type is not auto and slice width and height are not provided.")

        while y_max < image_height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < image_width:
                x_max = x_min + slice_width
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    slice_bboxes.append([xmin, ymin, xmax, ymax])
                else:
                    slice_bboxes.append([x_min, y_min, x_max, y_max])
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap
        return slice_bboxes


class VideoToFrames:
    def __call__(
        self,
        sample: SingularSample,
        cache_mechanisms: Dict[str, CacheMechanism] | None,
        display_engine: DisplayEngine | None,
    ) -> List[Sample]:
        video_element = sample.element

        frames = self._extract_frames(video_element.data)

        samples = []

        for i, frame in enumerate(frames):
            elements = defaultdict(list)
            # Transform the sample
            new_sample_id = f"{sample.id}_frame_{i}"
            new_element_id = f"{video_element.id}_frame_{i}"
            frame_element = Element(
                element_id=new_element_id,
                sample_id=new_sample_id,
                etype="image",
                load_mechanism=LoadMechanism(frame, category="image"),
                display_engine=display_engine,
                cache_mechanism=cache_mechanisms.get("image"),
                metadata={**video_element.metadata, IS_SAMPLE_COL_NAME: True},
            )
            new_load_mechanism = cache_mechanisms["image"].store(frame_element, frame_element.data)
            frame_element = frame_element.copy(load_mechanism=new_load_mechanism)

            for etype, elist in sample.elements.items():
                if etype == "video":
                    continue
                elist = [ele.copy(element_id=f"{ele.id}_frame_{i}", sample_id=new_sample_id) for ele in elist]
                elements[etype] = elist

            elements["image"].append(frame_element)

            frame_sample = SingularSample(elements=dict(elements))
            samples.append(frame_sample)
        return samples

        #         import cv2
        # from PIL import Image

        # frames = []
        # while True:
        #     ret, frame = video.read()
        #     if not ret:
        #         break
        #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     frames.append(Image.fromarray(frame_rgb))
        # video.release()

        # if not frames:
        #     raise ValueError("No frames were extracted from the video.")

        # return frames
        pass

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
