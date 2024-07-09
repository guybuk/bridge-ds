from pathlib import Path

import pytest

NOTEBOOK_DIR = Path.cwd() / "docs" / "source" / "notebooks"
DATA_DIR = Path.cwd() / "tests" / "data"

NOTEBOOKS_LIST = [p for p in NOTEBOOK_DIR.rglob("*[!\.ipynb_checkpoints]*.ipynb") if ".ipynb_checkpoints" not in str(p)]


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
    with tb.patch("tempfile.mkdtemp", return_value=str(DATA_DIR)):
        for cell in tb.cells:
            if "!pip install bridge-ds" in cell["source"]:
                cell["source"] = cell["source"].replace("!pip install bridge-ds", "")
        tb.execute()
