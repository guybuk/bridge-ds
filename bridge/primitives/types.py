from __future__ import annotations

from typing import TypeVar

from bridge.primitives.dataset import Dataset
from bridge.primitives.sample import Sample

D = TypeVar("D", bound=Dataset)
S = TypeVar("S", bound=Sample)
