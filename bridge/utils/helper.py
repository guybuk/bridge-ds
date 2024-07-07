import abc
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class Dictable(ABC):
    @property
    @abstractmethod
    def keys(self) -> List[str]:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, dic: Dict[str, Any], **kwargs):
        pass


class Displayable(abc.ABC):
    @abc.abstractmethod
    def show(self, **kwargs):
        pass
