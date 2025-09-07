from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from typing_extensions import Self


@dataclass
class URIComponents:
    scheme: str = ""
    netloc: str = ""
    path: str = ""
    params: str = ""
    query: str = ""
    fragment: str = ""

    def __str__(self):
        return urlunparse((self.scheme, self.netloc, self.path, self.params, self.query, self.fragment))

    def __next__(self):
        raise NotImplementedError("This is here for pandas...")

    @classmethod
    def from_str(cls, s: str) -> Self:
        return cls(*urlparse(s))
