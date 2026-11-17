"""Query-level ACL filter for Qdrant.

Invariant: the filter is derived ONLY from a verified Identity, and must be passed to the
vector DB query itself (and to every `prefetch` of a hybrid query). Never retrieve first and
filter in Python — that leaks unauthorized chunks into memory and pushes authorized results
out of the Top-K.
"""

from qdrant_client import models

from app.schemas import PUBLIC_ACCESS, Identity


def visible_access_groups(identity: Identity) -> list[str]:
    return sorted({identity.role.value, PUBLIC_ACCESS})


def build_acl_filter(identity: Identity) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id", match=models.MatchValue(value=identity.tenant_id)
            ),
            models.FieldCondition(
                key="allowed_roles",
                match=models.MatchAny(any=visible_access_groups(identity)),
            ),
        ]
    )
