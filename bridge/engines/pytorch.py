from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset as PytorchDataset

if TYPE_CHECKING:
    from bridge.primitives.dataset import Dataset


class PytorchEngineDataset(PytorchDataset):
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index) -> Any:
        elements = self._dataset.iget(index).elements
        outs = defaultdict(list)
        for etype, e_list in elements.items():
            for e in e_list:
                outs[str(etype)].append(e.data)
        return dict(outs)
