from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np

from bridge.primitives.element.data.category import DataCategory
from bridge.primitives.element.data.uri_components import URIComponents

if TYPE_CHECKING:
    from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE

REGISTRY = {}


def register(category: DataCategory):
    def decorator(method: Callable[[URIComponents | ELEMENT_DATA_TYPE], ELEMENT_DATA_TYPE]):
        if category in REGISTRY:
            warnings.warn(f"Overriding method for category {category}...")
        REGISTRY[category] = method
        return method

    return decorator


@register(DataCategory.image)
def _load_image(url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
    if not isinstance(url_or_data, URIComponents):
        return np.array(url_or_data)  # assumes object is a PIL image or np.ndarray

    if url_or_data.scheme not in ["http", "https", "file", ""]:
        raise NotImplementedError("Only loading from local or http(s) URLs is supported for now.")
    from skimage.io import imread

    return imread(str(url_or_data))


@register(DataCategory.text)
def _load_text(url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
    if not isinstance(url_or_data, URIComponents):
        return url_or_data  # assumes object is a string

    return open(str(url_or_data), "r").read()


@register(DataCategory.torch)
def _load_torch(url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
    if not isinstance(url_or_data, URIComponents):
        return url_or_data  # assume that is already torch tensor
    import torch

    return torch.load(str(url_or_data))


@register(DataCategory.numpy)
def _load_numpy(url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
    if not isinstance(url_or_data, URIComponents):
        return url_or_data  # assume that is already np array
    return np.load(str(url_or_data))


@register(DataCategory.obj)
def _load_obj(url_or_data: URIComponents | ELEMENT_DATA_TYPE) -> ELEMENT_DATA_TYPE:
    if not isinstance(url_or_data, URIComponents):
        return url_or_data
    raise NotImplementedError()


def load(url_or_data: URIComponents | ELEMENT_DATA_TYPE, category: DataCategory) -> ELEMENT_DATA_TYPE:
    return REGISTRY[category](url_or_data)


__all__ = ["load", "register"]
