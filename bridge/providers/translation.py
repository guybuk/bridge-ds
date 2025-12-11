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


class ParallelCorpus(DatasetProvider[MultiRoleDataset, MultiRoleSample]):
    """
    Provider for parallel text corpora (translation datasets).

    Supports two formats:
    - Tab-separated file: source\\ttarget per line
    - Separate aligned files: source.txt and target.txt with aligned lines

    Example usage:
        # Tab-separated format
        provider = ParallelCorpus("data/en-fr.tsv", role_names=("english", "french"))

        # Separate files format
        provider = ParallelCorpus(
            "data/train.en",
            target_file="data/train.fr",
            role_names=("english", "french")
        )
    """

    def __init__(
        self,
        source_file: str | os.PathLike,
        target_file: str | os.PathLike | None = None,
        role_names: Tuple[str, str] = ("source", "target"),
        separator: str = "\t",
    ):
        """
        Args:
            source_file: Path to source language file (or combined file if target_file is None)
            target_file: Path to target language file (None if using tab-separated format)
            role_names: Names for the two languages/roles
            separator: Separator for combined file format (default: tab)
        """
        self._source_file = Path(source_file)
        self._target_file = Path(target_file) if target_file else None
        self._role_names = role_names
        self._separator = separator

    def build_dataset(
        self,
        display_engine: DisplayEngine = None,
        cache_mechanisms: Dict[str, CacheMechanism | None] | None = None,
    ) -> MultiRoleDataset:
        if display_engine is None:
            display_engine = PairedPanel()

        source_elements = []
        target_elements = []

        if self._target_file is None:
            # Tab-separated format
            with open(self._source_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    parts = line.strip().split(self._separator)
                    if len(parts) >= 2:
                        source_text, target_text = parts[0], parts[1]

                        source_elem = Element(
                            element_id=f"source_{i}",
                            sample_id=i,
                            etype="text",
                            load_mechanism=LoadMechanism(source_text, category="obj"),
                        )
                        target_elem = Element(
                            element_id=f"target_{i}",
                            sample_id=i,
                            etype="text",
                            load_mechanism=LoadMechanism(target_text, category="obj"),
                        )

                        source_elements.append(source_elem)
                        target_elements.append(target_elem)
        else:
            # Separate files format
            with (
                open(self._source_file, "r", encoding="utf-8") as sf,
                open(self._target_file, "r", encoding="utf-8") as tf,
            ):
                for i, (source_line, target_line) in enumerate(zip(sf, tf)):
                    source_elem = Element(
                        element_id=f"source_{i}",
                        sample_id=i,
                        etype="text",
                        load_mechanism=LoadMechanism(source_line.strip(), category="obj"),
                    )
                    target_elem = Element(
                        element_id=f"target_{i}",
                        sample_id=i,
                        etype="text",
                        load_mechanism=LoadMechanism(target_line.strip(), category="obj"),
                    )

                    source_elements.append(source_elem)
                    target_elements.append(target_elem)

        return MultiRoleDataset.from_dict(
            {self._role_names[0]: source_elements, self._role_names[1]: target_elements},
            display_engine=display_engine,
            cache_mechanisms=cache_mechanisms,
        )
