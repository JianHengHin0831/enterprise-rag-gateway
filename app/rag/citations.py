"""Post-generation citation check: every [n] must refer to a context passed this request."""

import re
from dataclasses import dataclass

_CITATION = re.compile(r"\[(\d{1,3})\]")

DISCLAIMER = "\n\n（提示：部分引用编号无法对应到检索资料，请核实后再使用。）"


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
