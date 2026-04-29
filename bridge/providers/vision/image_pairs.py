from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from bridge.display.vision import ImagePairsPanelEngine
from bridge.primitives.dataset import Dataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import Sample
from bridge.providers.dataset_provider import DatasetProvider

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


def _collect_files_by_stem(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if p.stem in files:
            raise ValueError(
                f"Duplicate stem {p.stem!r} in {directory}: "
                f"{files[p.stem].name} and {p.name}"
            )
        files[p.stem] = p
    return files


class ImagePairs(DatasetProvider[Dataset, Sample]):
    """Paired-image dataset with `source` and `target` roles (both images).

    Layout: ``root/source/<name>.<ext>`` and ``root/target/<name>.<ext>`` with
    matching filename stems. Sample id is the filename stem. Useful for
    image-to-image tasks: super-resolution, style transfer, denoising.
    Pass ``encoding="png"`` (or another registered image encoding) for
    non-JPEG inputs.
    """

    SOURCE_ROLE = "source"
    TARGET_ROLE = "target"

    def __init__(self, root: str | os.PathLike, encoding: str = "jpeg"):
        self._root = Path(root)
        self._encoding = encoding
        self._source_dir = self._root / self.SOURCE_ROLE
        self._target_dir = self._root / self.TARGET_ROLE

    def build_dataset(
        self,
        display_engine: DisplayEngine | None = ImagePairsPanelEngine(),
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> Dataset:
        source_files = _collect_files_by_stem(self._source_dir)
        target_files = _collect_files_by_stem(self._target_dir)

        only_source = source_files.keys() - target_files.keys()
        only_target = target_files.keys() - source_files.keys()
        if only_source or only_target:
            raise ValueError(
                f"Unpaired stems. Only in source/: {sorted(only_source)}. "
                f"Only in target/: {sorted(only_target)}."
            )

        source_elems = []
        target_elems = []
        for stem in sorted(source_files):
            source_elems.append(self._build_element(stem, source_files[stem], self.SOURCE_ROLE))
            target_elems.append(self._build_element(stem, target_files[stem], self.TARGET_ROLE))

        return Dataset.from_role_dict(
            {self.SOURCE_ROLE: source_elems, self.TARGET_ROLE: target_elems},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )

    def _build_element(self, sample_id: str, path: Path, role: str) -> Element:
        return Element(
            element_id=f"{role}_{sample_id}",
            sample_id=sample_id,
            etype="image",
            role=role,
            load_mechanism=LoadMechanism.from_url_string(str(path), encoding=self._encoding),
            metadata={"filename": path.name},
        )
