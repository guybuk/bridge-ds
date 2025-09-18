from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from bridge.primitives.element.element import Element
from bridge.primitives.sample import Sample
from bridge.primitives.sample.transform.sample_transform import SampleTransform
from bridge.utils import optional_dependencies
from bridge.utils.data_objects import BoundingBox

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


class TorchvisionV2Transform(SampleTransform):
    def __init__(
        self,
        transforms: Union[List[v2.Transform], v2.Compose],
        bbox_format="XYWH",
    ):
        if isinstance(transforms, v2.Compose):
            self._transforms = transforms
        else:
            self._transforms = v2.Compose(transforms)
        self._bbox_format = bbox_format

    def __call__(
        self, sample: Sample, cache_mechanisms: Dict[str, CacheMechanism], display_engine: DisplayEngine | None
    ) -> Sample:
        # Extract and validate required elements
        image_element, bbox_elements = self._extract_elements(sample)

        # Convert to torchvision v2 format
        image_tensor, bbox_tensor, labels_tensor = self._convert_to_tv_tensors(image_element, bbox_elements)

        # Apply transforms
        transformed_image, transformed_bboxes, transformed_labels = self._apply_transforms(
            image_tensor, bbox_tensor, labels_tensor
        )

        # Reconstruct elements with transformed data
        new_elements = self._reconstruct_elements(
            sample.elements,
            image_element,
            bbox_elements,
            transformed_image,
            transformed_bboxes,
            transformed_labels,
            cache_mechanisms,
        )

        return Sample(elements=new_elements, display_engine=display_engine)

    def _extract_elements(self, sample: Sample) -> tuple[Element, List[Element]]:
        """Extract and validate image and bbox elements from sample."""
        if "image" not in sample.elements:
            raise ValueError("TorchvisionV2Transform requires 'image' element")

        image_element = sample.elements["image"][0]  # Single image element
        bbox_elements = sample.elements.get("bbox", [])  # List of bbox elements (optional)

        return image_element, bbox_elements

    def _convert_to_tv_tensors(
        self, image_element: Element, bbox_elements: List[Element]
    ) -> tuple[Any, tv_tensors.BoundingBoxes, torch.Tensor]:
        """Convert elements to torchvision v2 tensor format."""
        # Convert image to tv_tensors format
        image_data = image_element.data
        if image_element.category == "image":
            # convert to PIL image
            image_data = Image.fromarray(image_data)
            W, H = image_data.size
        elif image_element.category == "torch":
            image_data = tv_tensors.Image(image_data)
            W, H = image_data.shape[1], image_data.shape[2]

        # Convert bboxes to tv_tensors format with artificial labels
        bbox_coords = []
        bbox_labels = []
        for i, bbox_element in enumerate(bbox_elements):
            bbox_data: BoundingBox = bbox_element.data
            bbox_coords.append(bbox_data.coords)
            bbox_labels.append(i)  # Artificial label as index

        if bbox_coords:
            bbox_tensor = tv_tensors.BoundingBoxes(
                torch.stack([torch.from_numpy(coord) for coord in bbox_coords]),
                format=self._bbox_format,
                canvas_size=(H, W),
            )
            labels_tensor = torch.tensor(bbox_labels)
        else:
            bbox_tensor = tv_tensors.BoundingBoxes(
                torch.empty((0, 4)),
                format=self._bbox_format,
                canvas_size=(H, W),
            )
            labels_tensor = torch.empty(0, dtype=torch.long)

        return image_data, bbox_tensor, labels_tensor

    def _apply_transforms(
        self, image_data: Any, bbox_tensor: tv_tensors.BoundingBoxes, labels_tensor: torch.Tensor
    ) -> tuple[Any, Any, torch.Tensor]:
        """Apply torchvision v2 transforms to tensors."""
        # Apply transforms to image, bboxes, and labels together
        output_dict = self._transforms({"image": image_data, "boxes": bbox_tensor, "labels": labels_tensor})
        transformed_image = output_dict["image"]
        transformed_bboxes = output_dict["boxes"]
        transformed_labels = output_dict["labels"]
        transformed_labels = labels_tensor

        return transformed_image, transformed_bboxes, transformed_labels

    def _reconstruct_elements(
        self,
        original_elements: Dict[str, List[Element]],
        image_element: Element,
        bbox_elements: List[Element],
        transformed_image: Any,
        transformed_bboxes: Any,
        transformed_labels: torch.Tensor,
        cache_mechanisms: Dict[str, CacheMechanism],
    ) -> Dict[str, List[Element]]:
        """Reconstruct elements dict with transformed data."""
        # Start with a copy of all original elements to preserve untransformed ones
        new_elements = copy.deepcopy(original_elements)

        # Update the image element with transformed data
        new_image_element = self._create_transformed_element(image_element, transformed_image, cache_mechanisms)
        new_elements["image"] = [new_image_element]

        # Update bbox elements for surviving bboxes using the transformed labels
        # (SanitizeBoundingBoxes should be included in the user's transform pipeline)
        # The labels contain the original indices of the surviving bboxes
        surviving_bbox_elements = []
        for i, bbox_coords in enumerate(transformed_bboxes):
            original_idx = transformed_labels[i].item()  # Get the original index from the label
            if original_idx < len(bbox_elements):
                original_bbox_element = bbox_elements[original_idx]
                bbox_coords_np = bbox_coords.numpy()
                new_bbox_data = BoundingBox(coords=bbox_coords_np, class_label=original_bbox_element.data.class_label)

                new_bbox_element = self._create_transformed_element(
                    original_bbox_element, new_bbox_data, cache_mechanisms
                )
                surviving_bbox_elements.append(new_bbox_element)

        # Replace the bbox elements with only the surviving ones
        new_elements["bbox"] = surviving_bbox_elements

        return new_elements

    def _create_transformed_element(
        self, original_element: Element, transformed_data: Any, cache_mechanisms: Dict[str, CacheMechanism]
    ) -> Element:
        """Create a new element with transformed data"""
        if original_element.etype == "image":
            if isinstance(transformed_data, np.ndarray) or isinstance(transformed_data, Image.Image):
                new_category = "image"
            else:
                with optional_dependencies(error="raise"):
                    import torch
                if isinstance(transformed_data, torch.Tensor):
                    new_category = "torch"
                else:
                    raise NotImplementedError(f"Invalid data type: {type(transformed_data)}")
        elif original_element.etype == "bbox":
            new_category = "obj"
        else:
            raise NotImplementedError(f"Unsupported element type: {original_element.etype}")

        provider = cache_mechanisms[original_element.etype].store(
            original_element, transformed_data, as_category=new_category, should_update_elements=False
        )

        return Element(
            element_id=original_element.id,
            etype=original_element.etype,
            load_mechanism=provider,
            sample_id=original_element.sample_id,
            metadata=original_element.metadata,
        )
