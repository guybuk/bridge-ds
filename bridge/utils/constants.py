# Col Names
import dataclasses
from dataclasses import dataclass
from typing import List


def recursively_list_columns(col_names) -> List[str]:
    cols = []
    for col in dataclasses.fields(col_names):
        field = getattr(col_names, col.name)
        if col.type == str:
            cols.append(field)
        else:
            cols.extend(recursively_list_columns(field))
    return list(set(cols))


class FlatAttributeListMixin:
    def list(self) -> List[str]:
        return recursively_list_columns(self)


@dataclass
class LoadMechanismCols(FlatAttributeListMixin):
    CATEGORY: str
    URL_OR_DATA: str


@dataclass
class ElementCols(FlatAttributeListMixin):
    ID: str
    SAMPLE_ID: str
    ETYPE: str
    LOAD_MECHANISM: LoadMechanismCols


ELEMENT_ID_COL_NAME = "element_id"
SAMPLE_ID_COL_NAME = "sample_id"
DATA_CATEGORY_COL_NAME = "category"
ELEMENT_TYPE_COL_NAME = "element_type"

ELEMENT_DATA_COL_NAME = "data"


LOAD_MECHANISM_COL_NAMES = LoadMechanismCols(CATEGORY=DATA_CATEGORY_COL_NAME, URL_OR_DATA=ELEMENT_DATA_COL_NAME)

ELEMENT_COLS = ElementCols(
    ID=ELEMENT_ID_COL_NAME,
    SAMPLE_ID=SAMPLE_ID_COL_NAME,
    ETYPE=ELEMENT_TYPE_COL_NAME,
    LOAD_MECHANISM=LOAD_MECHANISM_COL_NAMES,
)


INDICES = [ELEMENT_COLS.SAMPLE_ID, ELEMENT_COLS.ID]

IS_SAMPLE_COL_NAME = "is_example"

__all__ = ["ELEMENT_COLS", "INDICES", "IS_SAMPLE_COL_NAME"]
