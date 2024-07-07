from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import PIL.Image

from bridge.primitives.element.data.category import DataCategory
from bridge.primitives.element.data.load.load_mechanism import LoadMechanism

if TYPE_CHECKING:
    from bridge.primitives.element.data.uri_components import URIComponents
    from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE


class CacheCategory(ABC):
    @staticmethod
    @abstractmethod
    def store(
        data: ELEMENT_DATA_TYPE,
        url: URIComponents | None,
        category: DataCategory,
    ) -> LoadMechanism:
        ...


class CacheImage(CacheCategory):
    @staticmethod
    def store(data: ELEMENT_DATA_TYPE, url: URIComponents | None, category: DataCategory) -> LoadMechanism:
        if url is None:
            return LoadMechanism(PIL.Image.fromarray(data), category)

        if url.scheme not in ["", "file"]:
            raise NotImplementedError()
        from skimage.io import imsave

        path = Path(str(url)).expanduser()

        Path.mkdir(path.parent, parents=True, exist_ok=True)
        imsave(path, data)
        return LoadMechanism.from_url_string(str(url), category)


class CacheTorch(CacheCategory):
    @staticmethod
    def store(data: ELEMENT_DATA_TYPE, url: URIComponents | None, category: DataCategory) -> LoadMechanism:
        if url is None:
            return LoadMechanism(data, category)

        if url.scheme not in ["", "file"]:
            raise NotImplementedError()
        import torch

        path = Path(str(url))

        Path.mkdir(path.parent, parents=True, exist_ok=True)
        torch.save(data, path)
        return LoadMechanism.from_url_string(str(url), category)


class CacheObj(CacheCategory):
    @staticmethod
    def store(data: ELEMENT_DATA_TYPE, url: URIComponents | None, category: DataCategory) -> LoadMechanism:
        if url is None:
            return LoadMechanism(data, category)

        if url.scheme not in ["", "file"]:
            raise NotImplementedError()
        import pickle

        path = Path(str(url))

        Path.mkdir(path.parent, parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return LoadMechanism.from_url_string(str(url), category)


class CachingMethodExecutor:
    CACHE_DATA_METHODS: Dict[DataCategory, CacheCategory] = {
        DataCategory.image: CacheImage,
        DataCategory.torch: CacheTorch,
        DataCategory.obj: CacheObj,
    }

    def store(self, data: ELEMENT_DATA_TYPE, url: URIComponents | None, category: DataCategory) -> LoadMechanism:
        return self.CACHE_DATA_METHODS[category].store(data, url, category)
