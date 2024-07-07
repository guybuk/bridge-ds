from pathlib import Path

import pytest
from testbook import testbook

NOTEBOOK_DIR = Path(__file__).parent.parent / "notebooks"

NOTEBOOKS_LIST = [p for p in NOTEBOOK_DIR.rglob("*[!\.ipynb_checkpoints]*.ipynb") if ".ipynb_checkpoints" not in str(p)]


@pytest.fixture(
    params=NOTEBOOKS_LIST,
    ids=[p.name for p in NOTEBOOKS_LIST],
)
def tb(request):
    with testbook(request.param, timeout=-1) as tb:
        yield tb


def test_notebook(tb):
    with tb.patch("tempfile.mkdtemp", return_value="./data"):
        tb.execute()
