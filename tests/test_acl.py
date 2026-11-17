from app.schemas import Identity, Role
from app.security.acl import build_acl_filter, visible_access_groups


def _conditions(identity: Identity) -> dict:
    f = build_acl_filter(identity)
    return {c.key: c.match for c in f.must}


def test_filter_pins_tenant(intern):
    assert _conditions(intern)["tenant_id"].value == "acme_corp"


def test_intern_sees_only_public(intern):
    assert visible_access_groups(intern) == ["all", "intern"]
    assert "hr" not in _conditions(intern)["allowed_roles"].any


def test_hr_sees_hr_and_public(hr):
    assert set(_conditions(hr)["allowed_roles"].any) == {"all", "hr"}


def test_other_tenant_filter_differs():
    beta = Identity(tenant_id="beta_inc", role=Role.HR, user_id="b1")
    assert _conditions(beta)["tenant_id"].value == "beta_inc"


def test_filter_is_conjunction(hr):
    f = build_acl_filter(hr)
    assert f.should is None and f.must_not is None and len(f.must) == 2
