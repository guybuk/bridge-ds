from bridge.utils import StrEnum


class DataCategory(StrEnum):
    image = "image"
    torch = "torch"
    numpy = "numpy"
    text = "text"
    obj = "obj"
