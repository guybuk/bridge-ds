from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from bridge.primitives.dataset import Dataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import Sample
from bridge.providers.dataset_provider import DatasetProvider

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


class CaptionedImages(DatasetProvider[Dataset, Sample]):
    """Image + text-caption dataset; canonical multimodal example.

    Layout:
      - ``root/images/<stem>.<ext>`` — one image per sample
      - ``root/captions.json`` — ``{stem: caption_string}``

    Sample id = image filename stem. Roles: ``image`` (etype ``image``) and
    ``caption`` (etype ``text``, encoding ``utf8``).
    """

    IMAGE_ROLE = "image"
    CAPTION_ROLE = "caption"

    def __init__(
        self,
        root: str | os.PathLike,
        captions_filename: str = "captions.json",
        image_encoding: str = "jpeg",
    ):
        self._root = Path(root)
        self._image_dir = self._root / "images"
        self._captions_path = self._root / captions_filename
        self._image_encoding = image_encoding

    def build_dataset(
        self,
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> Dataset:
        with self._captions_path.open() as f:
            captions = json.load(f)

        image_files = {p.stem: p for p in sorted(self._image_dir.iterdir())}
        missing = image_files.keys() - captions.keys()
        if missing:
            raise ValueError(f"Images without a caption entry: {sorted(missing)}")

        image_elems = []
        caption_elems = []
        for stem, img_path in image_files.items():
            image_elems.append(
                Element(
                    element_id=f"image_{stem}",
                    sample_id=stem,
                    etype="image",
                    role=self.IMAGE_ROLE,
                    load_mechanism=LoadMechanism.from_url_string(str(img_path), encoding=self._image_encoding),
                    metadata={"filename": img_path.name},
                )
            )
            caption_elems.append(
                Element(
                    element_id=f"caption_{stem}",
                    sample_id=stem,
                    etype="text",
                    role=self.CAPTION_ROLE,
                    load_mechanism=LoadMechanism(captions[stem], encoding="utf8"),
                )
            )

        return Dataset.from_role_dict(
            {self.IMAGE_ROLE: image_elems, self.CAPTION_ROLE: caption_elems},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )
