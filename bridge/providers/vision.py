from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
from tqdm import tqdm

from bridge.display.basic import SimplePrints
from bridge.primitives.dataset import SingularDataset
from bridge.primitives.element.data.category import DataCategory
from bridge.primitives.element.data.load.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.element.element_type import ElementType
from bridge.primitives.sample.singular_sample import SingularSample
from bridge.providers.dataset_provider import DatasetProvider
from bridge.utils import download_and_extract_archive, optional_dependencies
from bridge.utils.data_objects import BoundingBox, ClassLabel

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache.cache_mechanism import CacheMechanism


class ImageFolder(DatasetProvider[SingularDataset, SingularSample]):
    def __init__(self, root: str | os.PathLike):
        self._root = root

    def build_dataset(
        self, display_engine: DisplayEngine = None, cache_mechanisms: Dict[ElementType, CacheMechanism] = None
    ):
        images = []
        classes = []
        for i, class_dir in enumerate(sorted(Path(self._root).iterdir())):
            for img_file in class_dir.iterdir():
                sample_id = len(images)
                img_element = Element(
                    element_id=f"image_{sample_id}",
                    sample_id=sample_id,
                    etype=ElementType.image,
                    load_mechanism=LoadMechanism.from_url_string(str(img_file), category=DataCategory.image),
                    metadata={"filename": img_file.name},
                )
                class_element = Element(
                    element_id=f"class_{i}",
                    sample_id=sample_id,
                    etype=ElementType.class_label,
                    load_mechanism=LoadMechanism(ClassLabel(i, class_dir.name), category=DataCategory.obj),
                    metadata={"filename": img_file.name},
                )
                images.append(img_element)
                classes.append(class_element)
        return SingularDataset.from_lists(
            images, classes, display_engine=display_engine, cache_mechanisms=cache_mechanisms
        )


class Coco2017Detection(DatasetProvider[SingularDataset, SingularSample]):
    images_download_links = {
        "train": "http://images.cocodataset.org/zips/train2017.zip",
        "val": "http://images.cocodataset.org/zips/val2017.zip",
    }

    annotations_download_links = {
        "train": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "val": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }
    images_source = {"download", "local", "stream"}

    def __init__(self, root: str | os.PathLike, split: str = "train", img_source: str = "stream"):
        from pycocotools.coco import COCO

        assert split in self.images_download_links.keys(), f"Split must be one of: {self.images_download_links.keys()}"

        root = Path(root).expanduser()
        self._images_dir = root / f"{split}2017"
        self._img_source = img_source
        self._ann_file = root / "annotations" / f"instances_{split}2017.json"

        if self._ann_file.exists():
            print(f"Annotations file {self._ann_file} already exists, skipping download.")
        else:
            print("Downloading annotations...")
            download_and_extract_archive(self.annotations_download_links[split], str(root))

        if self._img_source == "download":
            root.mkdir(parents=True, exist_ok=True)
            if self._images_dir.exists():
                print(f"Images directory {self._images_dir} already exists, skipping download.")
            else:
                print("Downloading images...")
                download_and_extract_archive(self.images_download_links[split], str(root))
        self._coco = COCO(self._ann_file)

    def build_dataset(
        self,
        display_engine: DisplayEngine = SimplePrints(),
        cache_mechanisms: Dict[ElementType, CacheMechanism] = None,
    ):
        img_id_list = list(sorted(self._coco.imgs.keys()))
        images = []
        bboxes = []
        for img_id in tqdm(img_id_list):
            coco_img = self._coco.loadImgs(img_id)[0]
            img_file = self._images_dir / coco_img["file_name"]

            if self._img_source == "stream":
                url = coco_img["coco_url"]  # noqa
            else:
                url = str(img_file)
            load_mechanism = LoadMechanism.from_url_string(url, category=DataCategory.image)  # noqa
            img_element = Element(
                element_id=f"{img_id}_img",
                sample_id=img_id,
                etype=ElementType.image,
                load_mechanism=load_mechanism,
                metadata={k: v for k, v in coco_img.items() if k not in Element.keys and k != "id"},
            )
            images.append(img_element)
            coco_annotations = self._coco.loadAnns(self._coco.getAnnIds(img_id))
            for coco_ann_dict in coco_annotations:
                category_id = coco_ann_dict["category_id"]
                bbox_data = BoundingBox(
                    coords=(np.array(coco_ann_dict["bbox"])), class_label=(ClassLabel(category_id))
                )
                load_mechanism = LoadMechanism(bbox_data, category=DataCategory.obj)
                bbox_element = Element(
                    element_id=f"{img_id}_{coco_ann_dict['id']}",
                    sample_id=img_id,
                    etype=ElementType.bbox,
                    load_mechanism=load_mechanism,
                    metadata={
                        "category_id": category_id,
                        "area": coco_ann_dict["area"],
                        "iscrowd": coco_ann_dict["iscrowd"],
                    },
                )
                bboxes.append(bbox_element)
        return SingularDataset.from_lists(
            images, bboxes, display_engine=display_engine, cache_mechanisms=cache_mechanisms
        )


class TorchvisionCIFAR10(DatasetProvider[SingularDataset, SingularSample]):
    def __init__(self, root: str | os.PathLike, train: bool = True, download: bool = False):
        with optional_dependencies("raise"):
            from torchvision.datasets import CIFAR10

        self._ds = CIFAR10(root=root, train=train, download=download)

    def build_dataset(
        self,
        display_engine: DisplayEngine = SimplePrints(),
        cache_mechanisms: Dict[ElementType, CacheMechanism | None] | None = None,
    ):
        sample_list = []
        annotation_list = []
        for i, (img, target) in enumerate(zip(self._ds.data, self._ds.targets)):
            img_element = Element(
                element_id=i,
                etype=ElementType.image,
                sample_id=i,
                load_mechanism=LoadMechanism(url_or_data=img, category=DataCategory.image),
            )
            label_element = Element(
                element_id=f"label_{i}",
                etype=ElementType.class_label,
                sample_id=i,
                load_mechanism=LoadMechanism(
                    url_or_data=ClassLabel(class_idx=target, class_name=self._ds.classes[target]),
                    category=DataCategory.obj,
                ),
            )
            sample_list.append(img_element)
            annotation_list.append(label_element)

        return SingularDataset.from_lists(
            sample_list, annotation_list, display_engine=display_engine, cache_mechanisms=cache_mechanisms
        )
