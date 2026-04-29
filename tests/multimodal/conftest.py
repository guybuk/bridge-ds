"""Shared fixtures for multimodal tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _holoviews_bokeh_backend():
    """Panel display engines assert the holoviews backend is bokeh; notebooks
    set this via hv.extension('bokeh'). Match that here so tests don't need
    to repeat the dance."""
    import holoviews as hv

    hv.extension("bokeh")
