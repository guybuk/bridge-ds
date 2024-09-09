import abc
from pathlib import Path
from typing import Any

import numpy as np

from bridge.primitives.element.data.category_registry import register
from bridge.primitives.element.data.load_mechanism import LoadMechanism
from bridge.primitives.element.data.uri_components import URIComponents
from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE


class DataIO(abc.ABC):
    @property
    @abc.abstractmethod
    def category(self):
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
    category = "image"
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
        import PIL.Image
        from skimage.io import imsave

        if url is None:
            return LoadMechanism(PIL.Image.fromarray(data), cls.category)

        if url.scheme not in ["", "file"]:
            raise NotImplementedError("Only saving locally is supported for now.")

        path = Path(str(url)).expanduser()

        Path.mkdir(path.parent, parents=True, exist_ok=True)
        imsave(path, data)
        return LoadMechanism.from_url_string(str(url), cls.category)


@register
class TorchDataIO(DataIO):
    category = "torch"
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
            return LoadMechanism(data, cls.category)

        if url.scheme not in ["", "file"]:
            raise NotImplementedError("Only saving locally is supported for now.")
        import torch

        path = Path(str(url))

        Path.mkdir(path.parent, parents=True, exist_ok=True)
        torch.save(data, path)
        return LoadMechanism.from_url_string(str(url), cls.category)


@register
class NumpyDataIO(DataIO):
    category = "numpy"
    extension = ".npy"

    @classmethod
    def load(cls, url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return url_or_data
        return np.load(str(url_or_data))

    @classmethod
    def store(cls, data: Any, url: URIComponents | None) -> LoadMechanism:
        raise NotImplementedError()


@register
class TextDataIO(DataIO):
    category = "text"
    extension = ".txt"

    @classmethod
    def load(cls, url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return url_or_data
        return open(str(url_or_data), "r").read()

    @classmethod
    def store(cls, data: Any, url: URIComponents | None) -> LoadMechanism:
        raise NotImplementedError()


@register
class ObjDataIO(DataIO):
    category = "obj"
    extension = ".pkl"

    @classmethod
    def load(cls, url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return url_or_data
        raise NotImplementedError()

    @classmethod
    def store(cls, data: Any, url: URIComponents | None) -> LoadMechanism:
        if url is None:
            return LoadMechanism(data, cls.category)

        if url.scheme not in ["", "file"]:
            raise NotImplementedError("Only saving locally is supported for now.")
        import pickle

        path = Path(str(url))

        Path.mkdir(path.parent, parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return LoadMechanism.from_url_string(str(url), cls.category)


@register
class AviDataIO(DataIO):
    category = "avi"
    extension = ".avi"

    @classmethod
    def load(cls, url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
        if not isinstance(url_or_data, URIComponents):
            return url_or_data
        import cv2
        from PIL import Image

        video = cv2.VideoCapture(str(url_or_data))
        return video
        frames = []

        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        video.release()
        return frames

    @classmethod
    def store(cls, data: Any, url: URIComponents | None) -> LoadMechanism:
        raise NotImplementedError()
