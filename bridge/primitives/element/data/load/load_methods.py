from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

import numpy as np

from bridge.primitives.element.data.category import DataCategory
from bridge.primitives.element.data.uri_components import URIComponents

if TYPE_CHECKING:
    from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE


class LoadCategory(ABC):
    @staticmethod
    @abstractmethod
    def load(url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        pass


class LoadImage(LoadCategory):
    @staticmethod
    def load(url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return np.array(url_or_data)  # assumes object is a PIL image or np.ndarray

        if url_or_data.scheme not in ["http", "https", "file", ""]:
            raise NotImplementedError()
        from skimage.io import imread

        return imread(str(url_or_data))


class LoadText(LoadCategory):
    @staticmethod
    def load(url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return url_or_data  # assumes object is a string

        return open(str(url_or_data), "r").read()


class LoadTorch(LoadCategory):
    @staticmethod
    def load(url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return url_or_data
        import torch

        return torch.load(str(url_or_data))


class LoadNumpy(LoadCategory):
    @staticmethod
    def load(url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return url_or_data  # assume that is already np array
        return np.load(str(url_or_data))


class LoadObj(LoadCategory):
    @staticmethod
    def load(url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return url_or_data  # assume that is already np array
        raise NotImplementedError()


class LoadingMethodExecutor:
    LOAD_DATA_METHODS: Dict[DataCategory, LoadCategory] = {
        DataCategory.image: LoadImage,
        DataCategory.text: LoadText,
        DataCategory.torch: LoadTorch,
        DataCategory.numpy: LoadNumpy,
        DataCategory.obj: LoadObj,
    }

    def load(self, url_or_data: URIComponents | ELEMENT_DATA_TYPE, category: DataCategory) -> ELEMENT_DATA_TYPE:
        return self.LOAD_DATA_METHODS[category].load(url_or_data)
