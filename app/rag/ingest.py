"""Ingest Markdown docs with front matter into Qdrant.

Doc format (data/mock_docs/<tenant>/<access>/*.md):

    ---
    doc_id: acme_handbook
    tenant_id: acme_corp
    allowed_roles: [all]
    title: Employee Handbook
    ---
    # body...

Usage: python -m app.rag.ingest data/mock_docs
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.schemas import PUBLIC_ACCESS, Role
from app.security.injection import screen_text

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_VALID_ACCESS = {PUBLIC_ACCESS, *(r.value for r in Role)}


@dataclass
class SourceDoc:
    doc_id: str
    tenant_id: str
    allowed_roles: list[str]
    title: str
    body: str
    path: Path


@dataclass
class Chunk:
    doc_id: str
    tenant_id: str
    allowed_roles: list[str]
    title: str
    chunk_index: int
    text: str
    suspicious_patterns: list[str] = field(default_factory=list)

    def payload(self, index_version: int) -> dict:
        return {
            "doc_id": self.doc_id,
            "tenant_id": self.tenant_id,
            "allowed_roles": self.allowed_roles,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "suspicious": bool(self.suspicious_patterns),
            "index_version": index_version,
        }


def parse_doc(path: Path) -> SourceDoc:
    raw = path.read_text(encoding="utf-8")
    m = _FRONT_MATTER.match(raw)
    if not m:
        raise ValueError(f"{path}: missing front matter")
    meta = yaml.safe_load(m.group(1)) or {}
    for key in ("doc_id", "tenant_id", "allowed_roles"):
        if key not in meta:
            raise ValueError(f"{path}: front matter missing '{key}'")
    roles = list(meta["allowed_roles"])
    if not roles or not set(roles) <= _VALID_ACCESS:
        raise ValueError(f"{path}: invalid allowed_roles {roles}")
    return SourceDoc(
        doc_id=meta["doc_id"],
        tenant_id=meta["tenant_id"],
        allowed_roles=roles,
        title=meta.get("title", meta["doc_id"]),
        body=raw[m.end() :],
        path=path,
    )


def chunk_doc(doc: SourceDoc, max_chars: int = 500, overlap: int = 80) -> list[Chunk]:
    """Split by Markdown headings/paragraphs, then pack into ~max_chars windows.

    TODO(M1): tune max_chars/overlap via eval; consider heading-path prefixes per chunk.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.body) if p.strip()]
    windows: list[str] = []
    buf = ""
    for p in paragraphs:
        if buf and len(buf) + len(p) + 2 > max_chars:
            windows.append(buf)
            buf = buf[-overlap:] + "\n\n" + p if overlap else p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        windows.append(buf)
    return [
        Chunk(
            doc_id=doc.doc_id,
            tenant_id=doc.tenant_id,
            allowed_roles=doc.allowed_roles,
            title=doc.title,
            chunk_index=i,
            text=text,
            suspicious_patterns=screen_text(text),
        )
        for i, text in enumerate(windows)
    ]


def load_corpus(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.md")):
        chunks.extend(chunk_doc(parse_doc(path)))
    return chunks


async def upsert(chunks: list[Chunk]) -> None:
    # TODO(M1): embed (dense + sparse via app/rag/embeddings.py), create collection with
    # named vectors + payload indexes on tenant_id / allowed_roles, upsert points,
    # then bump `index_version:{tenant_id}` in Redis.
    raise NotImplementedError("ingest.upsert — milestone M1")


def main(argv: list[str]) -> None:
    root = Path(argv[1] if len(argv) > 1 else "data/mock_docs")
    chunks = load_corpus(root)
    flagged = [c for c in chunks if c.suspicious_patterns]
    print(f"parsed {len(chunks)} chunks; {len(flagged)} flagged suspicious")
    for c in flagged:
        print(f"  suspicious: {c.doc_id}#{c.chunk_index} {c.suspicious_patterns}")
    # TODO(M1): asyncio.run(upsert(chunks))


if __name__ == "__main__":
    main(sys.argv)
