from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from bridge.display.vision import Panel
from bridge.primitives.dataset import ImageLabelDataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import ImageLabelSample
from bridge.providers.dataset_provider import DatasetProvider
from bridge.utils import optional_dependencies
from bridge.utils.data_objects import ClassLabel

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


class ImageFolder(DatasetProvider[ImageLabelDataset, ImageLabelSample]):
    def __init__(self, root: str | os.PathLike):
        self._root = root

    def build_dataset(
        self,
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> ImageLabelDataset:
        if display_engine is None:
            display_engine = Panel()
        images = []
        labels = []
        for i, class_dir in enumerate(sorted(Path(self._root).iterdir())):
            for img_file in class_dir.iterdir():
                sample_id = len(images)
                img_element = Element(
                    element_id=f"image_{sample_id}",
                    sample_id=sample_id,
                    etype="image",
                    load_mechanism=LoadMechanism.from_url_string(str(img_file), category="image"),
                    metadata={"filename": img_file.name},
                )
                label_element = Element(
                    element_id=f"label_{sample_id}",
                    sample_id=sample_id,
                    etype="class_label",
                    load_mechanism=LoadMechanism(ClassLabel(i, class_dir.name), category="obj"),
                    metadata={"filename": img_file.name},
                )
                images.append(img_element)
                labels.append(label_element)
        return ImageLabelDataset.from_dict(
            {"image": images, "label": labels},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )


class TorchvisionCIFAR10(DatasetProvider[ImageLabelDataset, ImageLabelSample]):
    def __init__(self, root: str | os.PathLike, train: bool = True, download: bool = False):
        with optional_dependencies("raise"):
            from torchvision.datasets import CIFAR10

        self._ds = CIFAR10(root=str(root), train=train, download=download)

    def build_dataset(
        self,
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> ImageLabelDataset:
        if display_engine is None:
            display_engine = Panel()
        images = []
        labels = []
        for i, (img, target) in enumerate(zip(self._ds.data, self._ds.targets)):
            img_element = Element(
                element_id=i,
                etype="image",
                sample_id=i,
                load_mechanism=LoadMechanism(url_or_data=img, category="image"),
            )
            label_element = Element(
                element_id=f"label_{i}",
                etype="class_label",
                sample_id=i,
                load_mechanism=LoadMechanism(
                    url_or_data=ClassLabel(class_idx=target, class_name=self._ds.classes[target]),
                    category="obj",
                ),
            )
            images.append(img_element)
            labels.append(label_element)

        return ImageLabelDataset.from_dict(
            {"image": images, "label": labels},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )
