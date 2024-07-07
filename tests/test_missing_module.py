import pytest

from bridge.utils import optional_dependencies
from bridge.utils.optional_import import _generate_msg  # noqa


@pytest.fixture(autouse=True)
def assert_missing_module():
    with pytest.raises(ImportError):
        import missing_module  # noqa


def test_missing_module_raise():
    with pytest.raises(ImportError) as e_info:
        with optional_dependencies("raise"):
            import missing_module  # noqa
    assert e_info.value.msg == _generate_msg("missing_module", None)


def test_missing_module_raise_extra():
    with pytest.raises(ImportError) as e_info:
        with optional_dependencies("raise", ["dev"]):
            import missing_module  # noqa
    assert e_info.value.msg == _generate_msg("missing_module", ["dev"])


def test_missing_module_warn():
    with pytest.warns(UserWarning, match=_generate_msg("missing_module", None)):
        with optional_dependencies("warn"):
            import missing_module  # noqa


def test_missing_module_ignore(assert_missing_module):
    with optional_dependencies("ignore"):
        import missing_module  # noqa
