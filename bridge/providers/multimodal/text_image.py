from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from bridge.display.paired import PairedPanel
from bridge.primitives.dataset import TextImageDataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import TextImageSample
from bridge.providers.dataset_provider import DatasetProvider
from bridge.utils import download_and_extract_archive

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


class CocoCaptions(DatasetProvider[TextImageDataset, TextImageSample]):
    """
    Provider for COCO Captions dataset (text-to-image pairs).

    Returns a TextImageDataset with roles "text" and "image".

    Uses the COCO 2017 images with caption annotations.
    Each image may have multiple captions; this provider creates
    one sample per image-caption pair.

    Example usage:
        provider = CocoCaptions("~/.cache/coco", split="val", captions_per_image=1)
        ds = provider.build_dataset()
        ds.text   # DataFrame of caption elements
        ds.image  # DataFrame of image elements
    """

    images_download_links = {
        "train": "http://images.cocodataset.org/zips/train2017.zip",
        "val": "http://images.cocodataset.org/zips/val2017.zip",
    }

    captions_download_link = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    def __init__(
        self,
        root: str | os.PathLike,
        split: str = "train",
        img_source: str = "stream",
        captions_per_image: int = 1,
    ):
        """
        Args:
            root: Root directory for dataset
            split: "train" or "val"
            img_source: "stream" (URL), "download", or "local"
            captions_per_image: Number of captions to pair with each image (1-5)
        """
        from pycocotools.coco import COCO

        assert (
            split in self.images_download_links.keys()
        ), f"Split must be one of: {list(self.images_download_links.keys())}"
        assert 1 <= captions_per_image <= 5, "captions_per_image must be between 1 and 5"

        root = Path(root).expanduser()
        self._images_dir = root / f"{split}2017"
        self._img_source = img_source
        self._captions_per_image = captions_per_image
        self._ann_file = root / "annotations" / f"captions_{split}2017.json"

        # Download annotations if needed
        if not self._ann_file.exists():
            print("Downloading annotations...")
            download_and_extract_archive(self.captions_download_link, str(root))

        # Download images if requested
        if self._img_source == "download" and not self._images_dir.exists():
            print("Downloading images...")
            download_and_extract_archive(self.images_download_links[split], str(root))

        self._coco = COCO(self._ann_file)

    def build_dataset(
        self,
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> TextImageDataset:
        if display_engine is None:
            display_engine = PairedPanel()

        text_elements = []
        image_elements = []

        sample_idx = 0

        for img_id in sorted(self._coco.imgs.keys()):
            coco_img = self._coco.loadImgs(img_id)[0]
            ann_ids = self._coco.getAnnIds(imgIds=img_id)
            annotations = self._coco.loadAnns(ann_ids)

            # Determine image URL/path
            if self._img_source == "stream":
                img_url = coco_img["coco_url"]  # type: ignore[typeddict-item]
            else:
                img_url = str(self._images_dir / coco_img["file_name"])  # type: ignore[typeddict-item]

            # Create pairs for each caption (up to captions_per_image)
            for caption_data in annotations[: self._captions_per_image]:
                caption_text = caption_data["caption"]  # type: ignore[typeddict-item]

                text_elem = Element(
                    element_id=f"text_{sample_idx}",
                    sample_id=sample_idx,
                    etype="text",
                    load_mechanism=LoadMechanism(caption_text, category="obj"),
                    metadata={"image_id": img_id, "annotation_id": caption_data["id"]},
                )

                image_elem = Element(
                    element_id=f"image_{sample_idx}",
                    sample_id=sample_idx,
                    etype="image",
                    load_mechanism=LoadMechanism.from_url_string(img_url, category="image"),
                    metadata={
                        "image_id": img_id,
                        "file_name": coco_img["file_name"],
                        "width": coco_img["width"],
                        "height": coco_img["height"],
                    },
                )

                text_elements.append(text_elem)
                image_elements.append(image_elem)
                sample_idx += 1

        return TextImageDataset.from_dict(
            {"text": text_elements, "image": image_elements},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )
