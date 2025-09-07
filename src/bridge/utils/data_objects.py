from dataclasses import dataclass

import numpy as np


@dataclass
class ClassLabel:
    class_idx: int
    class_name: str | None = None

    def __post_init__(self):
        if self.class_name is None:
            self.class_name = str(self.class_idx)

    def __str__(self):
        return self.class_name

    def __hash__(self):
        return hash(f"{self.class_idx},{self.class_name}")

    def __eq__(self, other):
        if isinstance(other, str):
            return self.class_name == other
        else:
            return hash(self) == hash(other)


@dataclass
class BoundingBox:
    coords: np.ndarray
    class_label: ClassLabel | None = None

    def __post_init__(self):
        self.coords = self.coords.squeeze()
        assert self.coords.shape == (4,), f"Input coords to BoundingBox is wrong, {self.coords.shape}. Expected: 4"

    def __str__(self):
        if self.class_label is not None:
            return f"BoundingBox(class_name={self.class_label},coords={self.coords}"
        else:
            return f"BoundingBox(coords={self.coords})"


@dataclass
class Keypoint:
    coords: np.ndarray
    class_label: int | None = None
    class_name: str | None = None

    def __post_init__(self):
        self.coords = self.coords.squeeze()
        assert self.coords.shape == (2,), f"Input coords to Keypoint is wrong, {self.coords.shape}. Expected: 2"

    def __str__(self):
        if self.class_name is not None:
            return f"Keypoint(class_name={self.class_name},coords={self.coords}"
        else:
            return f"Keypoint(coords={self.coords})"
