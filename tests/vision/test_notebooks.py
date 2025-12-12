from pathlib import Path

import pytest

NOTEBOOK_DIR = Path.cwd() / "docs" / "source" / "user_guide" / "notebooks"
NOTEBOOKS_LIST = [
    p for p in NOTEBOOK_DIR.rglob(r"*[!\.ipynb_checkpoints]*.ipynb") if ".ipynb_checkpoints" not in str(p)
]


@pytest.fixture(
    params=NOTEBOOKS_LIST,
    ids=[p.name for p in NOTEBOOKS_LIST],
)
def tb(request):
    from testbook import testbook

    with testbook(request.param, timeout=-1) as tb:
        yield tb


# @pytest.mark.skip(reason="This test needs to be moved to the correct location")
def test_notebook(tb):
    tb.execute()
