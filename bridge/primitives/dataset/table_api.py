import abc
from typing import Callable, Dict, Sequence

import pandas as pd
from typing_extensions import Self


class TableAPI(abc.ABC):
    @property
    @abc.abstractmethod
    def elements(self) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def select(self, selector: Callable) -> Self:
        pass

    @abc.abstractmethod
    def assign(self, **kwargs: Dict[str, Callable[[pd.DataFrame], Sequence]]) -> Self:
        pass

    @abc.abstractmethod
    def sort(self, by: str, ascending: bool = True) -> Self:
        pass
