from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Union

import albumentations as A
import numpy as np
from PIL.Image import Image

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
        self,
        sample: Sample,
        cache_mechanisms: Dict[str, CacheMechanism],
        display_engine: DisplayEngine | None,
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
            # for i, keypoint in enumerate(elements["keypoint"]):
            #     keypoint: Keypoint
            #     albm_dict[f"keypoints_{i}"] = keypoint.coords
        return albm_dict

    def _albm_to_elements(
        self,
        elements: Dict[str, List[Element]],
        albm_dict: Dict[str, Any],
        cache_mechanisms: Dict[str, CacheMechanism],
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
        albm_data: Union[Image, List, np.ndarray],
        cache_mechanisms: Dict[str, CacheMechanism],
        curr_element: Element,
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
