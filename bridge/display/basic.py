from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict

from bridge.display.display_engine import DisplayEngine

if TYPE_CHECKING:
    from bridge.primitives.dataset import Dataset
    from bridge.primitives.element.element import Element
    from bridge.primitives.sample import Sample


class SimplePrints(DisplayEngine):
    def show_element(
        self,
        element: Element,
        element_plot_kwargs: Dict[str, Any] | None = None,
    ):
        print_dict = element.to_dict()
        print(json.dumps(print_dict, indent=4, default=str))

    def show_sample(
        self,
        sample: Sample,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
    ):
        element_dicts = {}
        for e_type, e_list in sample.elements.items():
            element_dicts[f"etype={e_type}"] = [e.to_dict() for e in e_list]

        print_dict = {f"Sample ID {sample.id}": {"Elements": element_dicts}}
        print(json.dumps(print_dict, indent=4, default=str))

    def show_dataset(
        self,
        dataset: Dataset,
        element_plot_kwargs: Dict[str, Any] | None = None,
        sample_plot_kwargs: Dict[str, Any] | None = None,
        dataset_plot_kwargs: Dict[str, Any] | None = None,
    ):
        for sample in dataset:
            self.show_sample(sample)
            print()
