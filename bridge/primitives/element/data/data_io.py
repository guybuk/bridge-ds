import abc
from pathlib import Path
from typing import Any

import numpy as np

from bridge.primitives.element.data.encoding_registry import register
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.data.uri_components import URIComponents
from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE


class DataIO(abc.ABC):
    @property
    @abc.abstractmethod
    def encoding(self):
        pass

    @property
    @abc.abstractmethod
    def extension(self):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        pass

    @classmethod
    @abc.abstractmethod
    def store(cls, data: Any, url: URIComponents | None) -> LoadMechanism:
        pass


@register
class JPEGDataIO(DataIO):
    encoding = "image"
    extension = ".jpg"

    @classmethod
    def load(cls, url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return np.array(url_or_data)  # assumes object is a PIL image or np.ndarray

        if url_or_data.scheme not in ["http", "https", "file", ""]:
            raise NotImplementedError("Only loading from local or http(s) URLs is supported for now.")
        from skimage.io import imread

        return imread(str(url_or_data))

    @classmethod
    def store(cls, data: Any, url: URIComponents | None) -> LoadMechanism:
        data = np.array(data)
        if url is None:
            return LoadMechanism(data, cls.encoding)

        if url.scheme not in ["", "file"]:
            raise NotImplementedError("Only saving locally is supported for now.")
        from skimage.io import imsave

        path = Path(str(url)).expanduser()
        Path.mkdir(path.parent, parents=True, exist_ok=True)
        imsave(path, data)
        return LoadMechanism.from_url_string(str(path), cls.encoding)


@register
class TorchDataIO(DataIO):
    encoding = "torch"
    extension = ".pt"

    @classmethod
    def load(cls, url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return url_or_data  # assume that is already torch tensor
        import torch

        return torch.load(str(url_or_data))

    @classmethod
    def store(cls, data: Any, url: URIComponents | None) -> LoadMechanism:
        if url is None:
            return LoadMechanism(data, cls.encoding)

        if url.scheme not in ["", "file"]:
            raise NotImplementedError("Only saving locally is supported for now.")
        import torch

        path = Path(str(url))

        Path.mkdir(path.parent, parents=True, exist_ok=True)
        torch.save(data, path)
        return LoadMechanism.from_url_string(str(url), cls.encoding)


@register
class TextDataIO(DataIO):
    encoding = "text"
    extension = ".txt"

    @classmethod
    def load(cls, url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return url_or_data
        return Path(str(url_or_data)).read_text()

    @classmethod
    def store(cls, data: Any, url: URIComponents | None) -> LoadMechanism:
        if url is None:
            return LoadMechanism(data, cls.encoding)

        if url.scheme not in ["", "file"]:
            raise NotImplementedError("Only saving locally is supported for now.")

        path = Path(str(url))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        return LoadMechanism.from_url_string(str(path), cls.encoding)


@register
class ObjDataIO(DataIO):
    encoding = "obj"
    extension = ".pkl"

    @classmethod
    def load(cls, url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return url_or_data

        if url_or_data.scheme not in ["", "file"]:
            raise NotImplementedError("Only loading locally is supported for now.")
        import pickle

        with open(str(url_or_data), "rb") as f:
            return pickle.load(f)

    @classmethod
    def store(cls, data: Any, url: URIComponents | None) -> LoadMechanism:
        if url is None:
            return LoadMechanism(data, cls.encoding)

        if url.scheme not in ["", "file"]:
            raise NotImplementedError("Only saving locally is supported for now.")
        import pickle

        path = Path(str(url))

        Path.mkdir(path.parent, parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return LoadMechanism.from_url_string(str(url), cls.encoding)
