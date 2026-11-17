"""Repository policy: no Chinese (CJK) characters in any tracked text file."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", ".ruff_cache", ".pytest_cache", "__pycache__", "logs"}
# CJK symbols/punctuation, CJK unified ideographs (+ext A), fullwidth forms.
_RANGES = [(0x3000, 0x303F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xFF00, 0xFFEF)]
CJK = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _RANGES) + "]")


def test_no_cjk_characters_in_repo():
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if CJK.search(line):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{lineno}")
    assert not offenders, "CJK characters found:\n" + "\n".join(offenders)
