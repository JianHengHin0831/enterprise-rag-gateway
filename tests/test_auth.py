from datetime import timedelta

import jwt
import pytest

from app.schemas import Role
from app.security.auth import AuthError, decode_token


def test_valid_token_yields_identity(settings, make_token):
    ident = decode_token(make_token(role=Role.HR, user="u9"), settings)
    assert (ident.tenant_id, ident.role, ident.user_id) == ("acme_corp", Role.HR, "u9")


def test_expired_token_rejected(settings, make_token):
    with pytest.raises(AuthError):
        decode_token(make_token(ttl=timedelta(seconds=-1)), settings)


def test_wrong_secret_rejected(settings):
    forged = jwt.encode(
        {"sub": "x", "tenant_id": "acme_corp", "role": "hr", "exp": 9999999999},
        "attacker-secret-attacker-secret-32b",
        "HS256",
    )
    with pytest.raises(AuthError):
        decode_token(forged, settings)


def test_alg_none_rejected(settings):
    forged = jwt.encode(
        {"sub": "x", "tenant_id": "acme_corp", "role": "hr", "exp": 9999999999}, None, "none"
    )
    with pytest.raises(AuthError):
        decode_token(forged, settings)


def test_unknown_tenant_and_role_rejected(settings, make_token):
    with pytest.raises(AuthError):
        decode_token(make_token(tenant="evil_corp"), settings)
    bad_role = jwt.encode(
        {"sub": "x", "tenant_id": "acme_corp", "role": "admin", "exp": 9999999999},
        settings.jwt_secret.get_secret_value(),
        "HS256",
    )
    with pytest.raises(AuthError):
        decode_token(bad_role, settings)
