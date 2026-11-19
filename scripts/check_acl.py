"""Live ACL leak check against the running Qdrant (uses the configured embedder).

For every tenant x role, runs a set of probe queries through HybridRetriever and verifies
each hit against the doc's front matter in data/mock_docs. Exit code 1 on any leak.

Usage: python -m scripts.check_acl [--top-k 50]
"""

import argparse
import asyncio
import sys
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from app.config import get_settings
from app.rag.embeddings import build_embedder
from app.rag.ingest import load_corpus
from app.rag.retriever import HybridRetriever
from app.schemas import PUBLIC_ACCESS, Identity, Role

PROBES = [
    "What is the P6 salary band and how is N+1 severance calculated?",
    "corporate bank account number and SWIFT code",
    "maximum enterprise discount and per-seat price",
    "order-svc error code and which database stores user profiles",
    "on-call compensation and SEV1 escalation",
    "parental leave weeks and offer approval chain",
    "headcount plan and support operations restructuring",
    "FY2026 department budget and variance rules",
    "employee handbook annual leave",
]


async def run(top_k: int) -> int:
    settings = get_settings()
    meta = {
        c.doc_id: (c.tenant_id, set(c.allowed_roles))
        for c in load_corpus(Path("data/mock_docs"), settings.known_tenants)
    }
    client = AsyncQdrantClient(url=settings.qdrant_url)
    retriever = HybridRetriever(client, build_embedder(settings), settings.qdrant_collection)
    leaks = 0
    try:
        for tenant in sorted(settings.known_tenants):
            for role in Role:
                identity = Identity(tenant_id=tenant, role=role, user_id="acl_check")
                visible = {role.value, PUBLIC_ACCESS}
                seen: set[str] = set()
                for q in PROBES:
                    for hit in await retriever.search(q, identity, top_k):
                        seen.add(hit.doc_id)
                        doc_tenant, roles = meta.get(hit.doc_id, ("?", set()))
                        if doc_tenant != tenant or not roles & visible:
                            leaks += 1
                            print(f"LEAK {tenant}/{role.value}: {hit.ref} ({doc_tenant} {roles})")
                print(f"{tenant:10} {role.value:9} sees {sorted(seen)}")
    finally:
        await client.close()
    print(f"\nleaks: {leaks}")
    return 1 if leaks else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m scripts.check_acl")
    parser.add_argument("--top-k", type=int, default=50)
    sys.exit(asyncio.run(run(parser.parse_args().top_k)))


if __name__ == "__main__":
    main()
