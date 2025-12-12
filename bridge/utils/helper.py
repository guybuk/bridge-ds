import abc
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Dict, List

from typing_extensions import Self


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class Dictable(ABC):
    keys: ClassVar[List[str]]

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, dic: Dict[str, Any], **kwargs) -> Self:
        pass


class Displayable(abc.ABC):
    @abc.abstractmethod
    def show(self, **kwargs):
        pass
