from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bridge.primitives.element.data.uri_components import URIComponents
from bridge.primitives.element.element_data_type import ELEMENT_DATA_TYPE

if TYPE_CHECKING:
    from bridge.primitives.element.data.load_mechanism import LoadMechanism

REGISTRY = {}


def is_registered(encoding: str) -> bool:
    return encoding in REGISTRY


def list_registered_encodings() -> list[str]:
    return list(REGISTRY.keys())


def register(cls):
    if cls.encoding in REGISTRY:
        raise ValueError(f"Encoding {cls.encoding} is already registered.")
    REGISTRY[cls.encoding] = cls
    return cls


def store(data: Any, url: URIComponents | None, encoding: str) -> LoadMechanism:
    return REGISTRY[encoding].store(data, url)


def load(url_or_data: URIComponents | ELEMENT_DATA_TYPE, encoding: str) -> ELEMENT_DATA_TYPE:
    return REGISTRY[encoding].load(url_or_data)


def extension(encoding: str) -> str:
    return REGISTRY[encoding].extension


# register default data io classes
import bridge.primitives.element.data.data_io  # noqa
