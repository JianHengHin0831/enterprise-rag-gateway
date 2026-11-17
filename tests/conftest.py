from datetime import timedelta

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.schemas import Identity, Role
from app.security.auth import issue_token


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, jwt_secret=SecretStr("test-secret-at-least-32-bytes-long!!"))


@pytest.fixture
def make_token(settings: Settings):
    def _make(
        tenant: str = "acme_corp",
        role: Role = Role.INTERN,
        user: str = "u1",
        ttl: timedelta = timedelta(minutes=5),
    ) -> str:
        return issue_token(tenant_id=tenant, role=role, user_id=user, settings=settings, ttl=ttl)

    return _make


@pytest.fixture
def intern() -> Identity:
    return Identity(tenant_id="acme_corp", role=Role.INTERN, user_id="u_intern")


@pytest.fixture
def hr() -> Identity:
    return Identity(tenant_id="acme_corp", role=Role.HR, user_id="u_hr")
