import os

import albumentations as A
import holoviews as hv
import pandas as pd
import panel as pn
import pytest

from bridge.display.vision import Holoviews
from bridge.primitives.element.data.cache.cache_mechanism import CacheMechanism
from bridge.primitives.element.data.uri_components import URIComponents
from bridge.primitives.element.element_type import ElementType
from bridge.primitives.sample.transform.vision import AlbumentationsCompose
from bridge.providers.vision import TorchvisionCIFAR10
from bridge.utils.constants import ELEMENT_COLS

hv.extension("bokeh")
pn.extension()


def test_cifar10(tmp_path):
    subset_len = 10

    ds = TorchvisionCIFAR10(root=tmp_path, download=True).build_dataset()
    ds_small = ds.select(lambda e: e.index.get_level_values(ELEMENT_COLS.SAMPLE_ID).drop_duplicates()[:subset_len])
    ds_small_copy = ds_small.elements
    transform = AlbumentationsCompose(albm_transforms=[A.RandomCrop(height=16, width=16)], bbox_format="coco")
    caches = {
        ElementType.image: CacheMechanism(URIComponents.from_str(str(tmp_path / "cache"))),
        ElementType.class_label: CacheMechanism(),
    }
    ds_small_t = ds_small.transform_samples(
        transform,
        display_engine=Holoviews(bbox_format="xywh"),
        cache_mechanisms=caches,
    )
    for sample in ds_small_t:
        for etype, elist in sample.elements.items():
            for e in elist:
                _ = e.data

    pd.testing.assert_frame_equal(ds_small_copy, ds_small.elements)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(ds_small_t.elements, ds_small.elements)
    assert len(os.listdir(tmp_path / "cache")) == subset_len
