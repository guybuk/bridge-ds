from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import albumentations as A
import numpy as np
from PIL.Image import Image

from bridge.display import DisplayEngine
from bridge.primitives.element.data.cache_mechanism import CacheMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import Sample
from bridge.primitives.sample.transform.sample_transform import SampleTransform
from bridge.utils import optional_dependencies
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
        elements = copy.deepcopy(sample.elements)
        albm_dict = self._elements_to_albm(elements)
        targets_dict = {k: k.split("_")[0] for k in albm_dict.keys()}
        compose = A.Compose(
            self._transforms,
            bbox_params=A.BboxParams(format=self._bbox_format),
            keypoint_params=A.KeypointParams(format=self._kp_format),
            additional_targets=targets_dict,
            is_check_shapes=False,
        )
        albm_dict = compose(**albm_dict)
        elements = self._albm_to_elements(elements, albm_dict, cache_mechanisms)
        sample = Sample(elements=elements, display_engine=display_engine)
        return sample

    def _elements_to_albm(self, elements: Dict[str, List[Element]]):
        albm_dict = {}
        assert "image" in elements, "Can't apply albumentations without an image element."
        for i, img_element in enumerate(elements["image"]):
            data = img_element.data
            albm_dict[f"image_{i}"] = data
        albm_dict.update({"image": albm_dict["image_0"], "bboxes": [], "keypoints": []})

        if "bbox" in elements:
            for i, bbox_element in enumerate(elements["bbox"]):
                data: BoundingBox = bbox_element.data
                albm_dict[f"bboxes_{i}"] = [[*(data.coords.tolist()), data.class_label]]
        if "keypoint" in elements:
            raise NotImplementedError("Didn't fully implement keypoints in albumentations yet.")
        return albm_dict

    def _albm_to_elements(
        self, elements: Dict[str, List[Element]], albm_dict: Dict[str, Any], cache_mechanisms: Dict[str, CacheMechanism]
    ):
        albm_to_elements = {"bboxes": "bbox", "keypoints": "keypoint", "image": "image"}
        del albm_dict["image"]
        del albm_dict["bboxes"]
        del albm_dict["keypoints"]
        sorted_desc_albm_keys = sorted(albm_dict.keys(), key=lambda k: int(k.split("_")[1]), reverse=True)
        for albm_key in sorted_desc_albm_keys:
            albm_data = albm_dict[albm_key]
            albm_type, element_idx = albm_key.split("_")
            element_idx = int(element_idx)
            curr_element = elements[albm_to_elements[albm_type]][element_idx]
            if len(albm_data) == 0:  # deleted element
                del elements[albm_to_elements[albm_type]][element_idx]
            else:
                curr_element = self._update_element_with_transformed_data(albm_data, cache_mechanisms, curr_element)
                elements[curr_element.etype][element_idx] = curr_element

        return elements

    @staticmethod
    def _update_element_with_transformed_data(
        albm_data: Union[Image, List, np.ndarray], cache_mechanisms: Dict[str, CacheMechanism], curr_element: Element
    ):
        if curr_element.etype == "bbox":
            albm_data = np.array(albm_data[0])
            new_element_data = BoundingBox(albm_data[:4], class_label=albm_data[4])  # noqa
            new_category = "obj"
        elif curr_element.etype == "image":
            if isinstance(albm_data, np.ndarray):
                new_category = "image"
            else:
                with optional_dependencies(error="raise"):
                    import torch
                if isinstance(albm_data, torch.Tensor):
                    new_category = "torch"
                else:
                    raise NotImplementedError(f"invalid data type: {type(albm_data)}")
            new_element_data = albm_data
        else:
            raise NotImplementedError()

        provider = cache_mechanisms[curr_element.etype].store(
            curr_element, new_element_data, as_category=new_category, should_update_elements=False
        )
        curr_element = Element(
            element_id=curr_element.id,
            etype=curr_element.etype,
            load_mechanism=provider,
            sample_id=curr_element.sample_id,
            metadata=curr_element.metadata,
        )
        return curr_element


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
        self,
        sample: Sample,
        cache_mechanisms: Dict[str, CacheMechanism] | None,
        display_engine: DisplayEngine | None,
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
