from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

from bridge.display.paired import PairedPanel
from bridge.primitives.dataset import PairedDataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import PairedSample
from bridge.providers.dataset_provider import DatasetProvider

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


class Pix2PixFolder(DatasetProvider[PairedDataset, PairedSample]):
    """
    Provider for pix2pix-style image-to-image datasets.

    Expects images where input and output are concatenated horizontally
    (left half = A, right half = B).

    Returns a PairedDataset with roles "source" and "target".

    Common datasets using this format:
    - edges2shoes
    - facades
    - maps

    Example usage:
        provider = Pix2PixFolder("data/facades/train", direction="AtoB")
        ds = provider.build_dataset()
        ds.source  # DataFrame of source images
        ds.target  # DataFrame of target images
    """

    def __init__(
        self,
        root: str | os.PathLike,
        direction: str = "AtoB",
    ):
        """
        Args:
            root: Path to folder containing concatenated images
            direction: "AtoB" (left->right) or "BtoA" (right->left)
        """
        self._root = Path(root)
        self._direction = direction
        assert direction in ["AtoB", "BtoA"], "direction must be 'AtoB' or 'BtoA'"

    def build_dataset(
        self,
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> PairedDataset:
        import numpy as np
        from PIL import Image

        if display_engine is None:
            display_engine = PairedPanel()

        source_elements = []
        target_elements = []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        for i, img_path in enumerate(sorted(self._root.iterdir())):
            if img_path.suffix.lower() not in image_extensions:
                continue

            # Load and split the concatenated image
            img = np.array(Image.open(img_path))
            h, w = img.shape[:2]
            mid = w // 2

            if self._direction == "AtoB":
                source_img, target_img = img[:, :mid], img[:, mid:]
            else:  # BtoA
                target_img, source_img = img[:, :mid], img[:, mid:]

            source_elem = Element(
                element_id=f"source_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism(source_img, category="obj"),
                metadata={"filename": img_path.name},
            )
            target_elem = Element(
                element_id=f"target_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism(target_img, category="obj"),
                metadata={"filename": img_path.name},
            )

            source_elements.append(source_elem)
            target_elements.append(target_elem)

        return PairedDataset.from_dict(
            {"source": source_elements, "target": target_elements},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )


class AlignedImageFolders(DatasetProvider[PairedDataset, PairedSample]):
    """
    Provider for paired images stored in separate aligned folders.

    Returns a PairedDataset with roles "source" and "target".

    Directory structure:
        root/
            A/
                image1.jpg
                image2.jpg
            B/
                image1.jpg
                image2.jpg

    Images are matched by filename.

    Example usage:
        provider = AlignedImageFolders(
            "data/day2night",
            folder_names=("day", "night"),
        )
        ds = provider.build_dataset()
        ds.source  # DataFrame of source images (from first folder)
        ds.target  # DataFrame of target images (from second folder)
    """

    def __init__(
        self,
        root: str | os.PathLike,
        folder_names: Tuple[str, str] = ("A", "B"),
    ):
        """
        Args:
            root: Root directory containing the two folders
            folder_names: Names of the two subdirectories (first=source, second=target)
        """
        self._root = Path(root)
        self._folder_names = folder_names

    def build_dataset(
        self,
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> PairedDataset:
        if display_engine is None:
            display_engine = PairedPanel()

        source_folder = self._root / self._folder_names[0]
        target_folder = self._root / self._folder_names[1]

        source_elements = []
        target_elements = []

        # Get matching filenames
        source_files = {f.name: f for f in source_folder.iterdir() if f.is_file()}
        target_files = {f.name: f for f in target_folder.iterdir() if f.is_file()}
        common_names = sorted(set(source_files.keys()) & set(target_files.keys()))

        for i, name in enumerate(common_names):
            source_elem = Element(
                element_id=f"source_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism.from_url_string(str(source_files[name]), category="image"),
                metadata={"filename": name},
            )
            target_elem = Element(
                element_id=f"target_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism.from_url_string(str(target_files[name]), category="image"),
                metadata={"filename": name},
            )

            source_elements.append(source_elem)
            target_elements.append(target_elem)

        return PairedDataset.from_dict(
            {"source": source_elements, "target": target_elements},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )
