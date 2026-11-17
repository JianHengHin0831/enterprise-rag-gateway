"""JWT verification -> Identity.

This is the ONLY place an Identity may be created from request data. In production the
token would be issued by an upstream SSO / API gateway; here `scripts/mint_token.py` issues
dev tokens with the shared secret.
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings
from app.schemas import Identity, Role

_REQUIRED_CLAIMS = ["exp", "sub", "tenant_id", "role"]


class AuthError(Exception):
    pass


def decode_token(token: str, settings: Settings) -> Identity:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            options={"require": _REQUIRED_CLAIMS},
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {type(exc).__name__}") from exc

    tenant_id = payload["tenant_id"]
    if tenant_id not in settings.known_tenants:
        raise AuthError("unknown tenant")
    try:
        role = Role(payload["role"])
    except ValueError as exc:
        raise AuthError("unknown role") from exc
    return Identity(tenant_id=tenant_id, role=role, user_id=str(payload["sub"]))


def issue_token(
    *, tenant_id: str, role: Role, user_id: str, settings: Settings, ttl: timedelta
) -> str:
    """Dev/test helper. Not exposed over HTTP."""
    now = datetime.now(UTC)
    claims = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role.value,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), settings.jwt_algorithm)


async def get_identity(
    authorization: Annotated[str | None, Header()] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> Identity:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    try:
        return decode_token(authorization[7:].strip(), settings)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
