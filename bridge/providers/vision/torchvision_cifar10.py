from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict

from bridge.display.vision import DetectionPanelEngine
from bridge.primitives.dataset import Dataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import Sample
from bridge.providers.dataset_provider import DatasetProvider
from bridge.utils import optional_dependencies
from bridge.utils.data_objects import ClassLabel

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


class TorchvisionCIFAR10(DatasetProvider[Dataset, Sample]):
    def __init__(self, root: str | os.PathLike, train: bool = True, download: bool = False):
        with optional_dependencies("raise"):
            from torchvision.datasets import CIFAR10

        self._ds = CIFAR10(root=root, train=train, download=download)

    def build_dataset(
        self,
        display_engine: DisplayEngine = DetectionPanelEngine(),
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> Dataset:
        sample_list = []
        annotation_list = []
        for i, (img, target) in enumerate(zip(self._ds.data, self._ds.targets)):
            img_element = Element(
                element_id=i,
                etype="image",
                sample_id=i,
                load_mechanism=LoadMechanism(url_or_data=img, encoding="jpeg"),
            )
            label_element = Element(
                element_id=f"label_{i}",
                etype="class_label",
                sample_id=i,
                load_mechanism=LoadMechanism(
                    url_or_data=ClassLabel(class_idx=target, class_name=self._ds.classes[target]),
                    encoding="pickle",
                ),
            )
            sample_list.append(img_element)
            annotation_list.append(label_element)

        return Dataset.from_role_dict(
            {"image": sample_list, "class_label": annotation_list},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )
