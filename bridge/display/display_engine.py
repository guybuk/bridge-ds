from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generic

from bridge.primitives.types import D, S

if TYPE_CHECKING:
    from bridge.primitives.element.element import Element


class DisplayEngine(ABC, Generic[D, S]):
    @abstractmethod
    def show_element(
        self,
        element: Element,
        element_plot_kwargs: Dict[str, Any] | None = None,
    ):
        pass

    @abstractmethod
    def show_sample(
        self,
        sample: S,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
    ):
        pass

    @abstractmethod
    def show_dataset(
        self,
        dataset: D,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
        dataset_plot_kwargs: Dict[str, Any] | None = None,
    ):
        pass
