from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from bridge.primitives.element.data.category import DataCategory
from bridge.primitives.element.data.load.load_mechanism import LoadMechanism

if TYPE_CHECKING:
    from bridge.primitives.element.data.uri_components import URIComponents

REGISTRY = {}


def register(category: DataCategory):
    def decorator(method: Callable[[Any, URIComponents, DataCategory], LoadMechanism]):
        if category in REGISTRY:
            warnings.warn(f"Overriding method for category {category}...")
        REGISTRY[category] = method
        return method

    return decorator


@register(DataCategory.image)
def _store_image(data: Any, url: URIComponents | None, category: DataCategory) -> LoadMechanism:
    import PIL.Image
    from skimage.io import imsave

    if url is None:
        return LoadMechanism(PIL.Image.fromarray(data), category)

    if url.scheme not in ["", "file"]:
        raise NotImplementedError("Only saving locally is supported for now.")

    path = Path(str(url)).expanduser()

    Path.mkdir(path.parent, parents=True, exist_ok=True)
    imsave(path, data)
    return LoadMechanism.from_url_string(str(url), category)


@register(DataCategory.torch)
def _store_torch(data: Any, url: URIComponents | None, category: DataCategory) -> LoadMechanism:
    if url is None:
        return LoadMechanism(data, category)

    if url.scheme not in ["", "file"]:
        raise NotImplementedError("Only saving locally is supported for now.")
    import torch

    path = Path(str(url))

    Path.mkdir(path.parent, parents=True, exist_ok=True)
    torch.save(data, path)
    return LoadMechanism.from_url_string(str(url), category)


@register(DataCategory.obj)
def _store_obj(data: Any, url: URIComponents | None, category: DataCategory) -> LoadMechanism:
    if url is None:
        return LoadMechanism(data, category)

    if url.scheme not in ["", "file"]:
        raise NotImplementedError("Only saving locally is supported for now.")
    import pickle

    path = Path(str(url))

    Path.mkdir(path.parent, parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return LoadMechanism.from_url_string(str(url), category)


def store(data: Any, url: URIComponents | None, category: DataCategory) -> LoadMechanism:
    return REGISTRY[category](data, url, category)


__all__ = ["register", "store"]
