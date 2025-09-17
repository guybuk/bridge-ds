from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bridge.primitives.element.data.uri_components import URIComponents
from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE

if TYPE_CHECKING:
    from bridge.primitives.element.data.load_mechanism import LoadMechanism

REGISTRY = {}


def is_registered(category: str) -> bool:
    return category in REGISTRY


def list_registered_categories() -> list[str]:
    return list(REGISTRY.keys())


def register(cls):
    if cls.category in REGISTRY:
        raise ValueError(f"Category {cls.category} is already registered.")
    REGISTRY[cls.category] = cls
    return cls


def store(data: Any, url: URIComponents | None, category: str) -> LoadMechanism:
    return REGISTRY[category].store(data, url)


def load(url_or_data: URIComponents | ELEMENT_DATA_TYPE, category: str) -> ELEMENT_DATA_TYPE:
    return REGISTRY[category].load(url_or_data)


def extension(category: str) -> str:
    return REGISTRY[category].extension


# register default data io classes
import bridge.primitives.element.data.data_io  # noqa
