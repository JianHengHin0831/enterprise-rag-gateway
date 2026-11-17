"""Post-generation citation check: every [n] must refer to a context passed this request."""

import re
from dataclasses import dataclass

_CITATION = re.compile(r"\[(\d{1,3})\]")

DISCLAIMER = (
    "\n\n(Note: some citation numbers do not match any retrieved source. "
    "Please verify before relying on this answer.)"
)


@dataclass(frozen=True)
class CitationCheck:
    cited: tuple[int, ...]
    invalid: tuple[int, ...]

    @property
    def has_citation(self) -> bool:
        return bool(self.cited)

    @property
    def mismatch(self) -> bool:
        return bool(self.invalid)


def check_citations(answer: str, num_contexts: int) -> CitationCheck:
    cited = sorted({int(n) for n in _CITATION.findall(answer)})
    invalid = [n for n in cited if not 1 <= n <= num_contexts]
    return CitationCheck(cited=tuple(cited), invalid=tuple(invalid))
