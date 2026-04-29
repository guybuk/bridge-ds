from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from bridge.display.vision import Panel
from bridge.primitives.dataset import SingularDataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample.singular_sample import SingularSample
from bridge.providers.dataset_provider import DatasetProvider
from bridge.utils.data_objects import ClassLabel

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


class ImageFolder(DatasetProvider[SingularDataset, SingularSample]):
    def __init__(self, root: str | os.PathLike):
        self._root = root

    def build_dataset(
        self, display_engine: DisplayEngine = Panel(), cache_mechanisms: Dict[str, CacheMechanism] = None
    ):
        images = []
        classes = []
        for i, class_dir in enumerate(sorted(Path(self._root).iterdir())):
            for img_file in class_dir.iterdir():
                sample_id = len(images)
                img_element = Element(
                    element_id=f"image_{sample_id}",
                    sample_id=sample_id,
                    etype="image",
                    load_mechanism=LoadMechanism.from_url_string(str(img_file), encoding="jpeg"),
                    metadata={"filename": img_file.name},
                )
                class_element = Element(
                    element_id=f"class_{sample_id}",
                    sample_id=sample_id,
                    etype="class_label",
                    load_mechanism=LoadMechanism(ClassLabel(i, class_dir.name), encoding="pickle"),
                    metadata={"filename": img_file.name},
                )
                images.append(img_element)
                classes.append(class_element)
        return SingularDataset.from_lists(
            images, classes, display_engine=display_engine, cache_mechanisms=cache_mechanisms
        )
