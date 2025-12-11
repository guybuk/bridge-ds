from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

from bridge.display.paired import PairedPanel
from bridge.primitives.dataset import MultiRoleDataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import MultiRoleSample
from bridge.providers.dataset_provider import DatasetProvider

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


class Pix2PixFolder(DatasetProvider[MultiRoleDataset, MultiRoleSample]):
    """
    Provider for pix2pix-style image-to-image datasets.

    Expects images where input and output are concatenated horizontally
    (left half = A, right half = B).

    Common datasets using this format:
    - edges2shoes
    - facades
    - maps

    Example usage:
        provider = Pix2PixFolder(
            "data/facades/train",
            role_names=("input", "output"),
            direction="AtoB"
        )
    """

    def __init__(
        self,
        root: str | os.PathLike,
        role_names: Tuple[str, str] = ("input", "output"),
        direction: str = "AtoB",
    ):
        """
        Args:
            root: Path to folder containing concatenated images
            role_names: Names for input/output roles
            direction: "AtoB" (left->right) or "BtoA" (right->left)
        """
        self._root = Path(root)
        self._role_names = role_names
        self._direction = direction
        assert direction in ["AtoB", "BtoA"], "direction must be 'AtoB' or 'BtoA'"

    def build_dataset(
        self,
        display_engine: DisplayEngine = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> MultiRoleDataset:
        import numpy as np
        from PIL import Image

        if display_engine is None:
            display_engine = PairedPanel()

        first_elements = []
        second_elements = []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        for i, img_path in enumerate(sorted(self._root.iterdir())):
            if img_path.suffix.lower() not in image_extensions:
                continue

            # Load and split the concatenated image
            img = np.array(Image.open(img_path))
            h, w = img.shape[:2]
            mid = w // 2

            if self._direction == "AtoB":
                left_img, right_img = img[:, :mid], img[:, mid:]
            else:  # BtoA
                right_img, left_img = img[:, :mid], img[:, mid:]

            first_elem = Element(
                element_id=f"first_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism(left_img, category="obj"),
                metadata={"filename": img_path.name},
            )
            second_elem = Element(
                element_id=f"second_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism(right_img, category="obj"),
                metadata={"filename": img_path.name},
            )

            first_elements.append(first_elem)
            second_elements.append(second_elem)

        return MultiRoleDataset.from_dict(
            {self._role_names[0]: first_elements, self._role_names[1]: second_elements},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )


class AlignedImageFolders(DatasetProvider[MultiRoleDataset, MultiRoleSample]):
    """
    Provider for paired images stored in separate aligned folders.

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
            role_names=("day", "night")
        )
    """

    def __init__(
        self,
        root: str | os.PathLike,
        folder_names: Tuple[str, str] = ("A", "B"),
        role_names: Tuple[str, str] = ("input", "output"),
    ):
        """
        Args:
            root: Root directory containing the two folders
            folder_names: Names of the two subdirectories
            role_names: Role names for the dataset
        """
        self._root = Path(root)
        self._folder_names = folder_names
        self._role_names = role_names

    def build_dataset(
        self,
        display_engine: DisplayEngine = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> MultiRoleDataset:
        if display_engine is None:
            display_engine = PairedPanel()

        first_folder = self._root / self._folder_names[0]
        second_folder = self._root / self._folder_names[1]

        first_elements = []
        second_elements = []

        # Get matching filenames
        first_files = {f.name: f for f in first_folder.iterdir() if f.is_file()}
        second_files = {f.name: f for f in second_folder.iterdir() if f.is_file()}
        common_names = sorted(set(first_files.keys()) & set(second_files.keys()))

        for i, name in enumerate(common_names):
            first_elem = Element(
                element_id=f"first_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism.from_url_string(str(first_files[name]), category="image"),
                metadata={"filename": name},
            )
            second_elem = Element(
                element_id=f"second_{i}",
                sample_id=i,
                etype="image",
                load_mechanism=LoadMechanism.from_url_string(str(second_files[name]), category="image"),
                metadata={"filename": name},
            )

            first_elements.append(first_elem)
            second_elements.append(second_elem)

        return MultiRoleDataset.from_dict(
            {self._role_names[0]: first_elements, self._role_names[1]: second_elements},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )
