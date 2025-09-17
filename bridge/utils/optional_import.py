import warnings
from contextlib import contextmanager
from typing import List


@contextmanager
def optional_dependencies(error: str = "ignore", extras: List[str] | None = None):
    assert error in {"raise", "warn", "ignore"}
    try:
        yield None
    except ImportError as e:
        dep_name = e.name
        msg = _generate_msg(dep_name, extras)
        if error == "raise":
            raise ImportError(msg) from e
        if error == "warn":
            warnings.warn(msg, UserWarning)


def _generate_msg(dep_name: str, extras: List[str] | None):
    if extras is not None:
        extras_str = ", or alternatively install: " + ", ".join([f"bridge_ds[{extra}]" for extra in extras])
    else:
        extras_str = "."
    msg = f'Missing optional dependency "{dep_name}". Use pip or conda to install directly' + extras_str
    return msg
