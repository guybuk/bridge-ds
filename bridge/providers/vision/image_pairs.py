from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from bridge.display.vision import Panel
from bridge.primitives.dataset import Dataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import Sample
from bridge.providers.dataset_provider import DatasetProvider

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


class ImagePairs(DatasetProvider[Dataset, Sample]):
    """Paired-image dataset with `source` and `target` roles (both images).

    Layout: ``root/source/<name>.<ext>`` and ``root/target/<name>.<ext>`` with
    matching filename stems. Sample id is the filename stem. Useful for
    image-to-image tasks: super-resolution, style transfer, denoising.
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
        display_engine: DisplayEngine | None = Panel(),
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> Dataset:
        source_files = {p.stem: p for p in sorted(self._source_dir.iterdir())}
        target_files = {p.stem: p for p in sorted(self._target_dir.iterdir())}

        unpaired = source_files.keys() ^ target_files.keys()
        if unpaired:
            raise ValueError(f"Unpaired stems between source/ and target/: {sorted(unpaired)}")

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
