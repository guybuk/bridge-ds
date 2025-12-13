from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict

from bridge.display.text import Panel
from bridge.primitives.dataset import TextLabelDataset
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.element import Element
from bridge.primitives.sample import TextLabelSample
from bridge.providers.dataset_provider import DatasetProvider
from bridge.utils import download_and_extract_archive
from bridge.utils.data_objects import ClassLabel

if TYPE_CHECKING:
    from bridge.display import DisplayEngine
    from bridge.primitives.element.data.cache_mechanism import CacheMechanism


class LargeMovieReviewDataset(DatasetProvider[TextLabelDataset, TextLabelSample]):
    """
    Provider for IMDB Large Movie Review Dataset (sentiment classification).

    Returns a TextLabelDataset with roles "text" and "label".

    Example usage:
        provider = LargeMovieReviewDataset("~/.cache/imdb", split="train", download=True)
        ds = provider.build_dataset()
        ds.text   # DataFrame of text elements
        ds.label  # DataFrame of label elements (pos/neg)
    """

    dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    def __init__(self, root: str | os.PathLike, split: str = "train", download: bool = False):
        root = Path(root).expanduser()

        if download:
            if (root / "aclImdb_v1.tar.gz").exists():
                print("Archive file aclImdb_v1.tar.gz already exists, skipping download.")
            else:
                download_and_extract_archive(self.dataset_url, str(root))
        self._split_root = root / "aclImdb" / split

    def build_dataset(
        self,
        display_engine: DisplayEngine | None = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> TextLabelDataset:
        if display_engine is None:
            display_engine = Panel()
        text_elements = []
        label_elements = []

        class_dir_list = [d for d in list(self._split_root.iterdir()) if d.is_dir()]
        for class_idx, class_dir in enumerate(sorted(class_dir_list)):
            for textfile in class_dir.iterdir():
                load_mechanism = LoadMechanism.from_url_string(str(textfile), "text")
                text_element = Element(
                    element_id=f"text_{textfile.stem}",
                    sample_id=textfile.stem,
                    etype="text",
                    load_mechanism=load_mechanism,
                )
                load_mechanism = LoadMechanism(ClassLabel(class_idx, class_dir.name), category="obj")
                label_element = Element(
                    element_id=f"label_{textfile.stem}",
                    sample_id=textfile.stem,
                    etype="class_label",
                    load_mechanism=load_mechanism,
                )
                text_elements.append(text_element)
                label_elements.append(label_element)

        return TextLabelDataset.from_dict(
            {"text": text_elements, "label": label_elements},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )
