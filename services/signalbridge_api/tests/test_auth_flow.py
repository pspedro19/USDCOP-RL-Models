"""
Integration tests for the SignalBridge authentication flow.

These hit a RUNNING SignalBridge API (default http://localhost:8085) plus its
Postgres (sb_users) and Redis (lockout + blacklist). They are skipped
automatically if the API is not reachable, so they are safe to collect in any
environment.

As of the admin-approval flow (migration 053), registration returns 202 PENDING
(no tokens) and login is blocked until an admin approves. Tests that need a
usable, logged-in account provision one through the admin+MailHog path via the
shared `_approval_helpers` (and skip if that path is unavailable).

Run:
    pytest services/signalbridge_api/tests/test_auth_flow.py -v
    AUTH_BASE_URL=http://localhost:8085 pytest .../test_auth_flow.py -v
"""

from __future__ import annotations

import time

import _approval_helpers as H
import httpx
import pytest

BASE_URL = H.BASE_URL
AUTH = H.AUTH
STRONG_PW = H.STRONG_PW

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not H.api_up(), reason=f"SignalBridge API not reachable at {BASE_URL}"),
]


@pytest.fixture()
def client():
    with httpx.Client(timeout=15.0) as c:
        yield c


@pytest.fixture()
def approved_user(client):
    """An active, logged-in account provisioned through the full approval flow.

    Requires the admin+MailHog stack (docker-compose.mailhog.yml). Skips otherwise.
    """
    if not H.mailhog_up():
        pytest.skip(f"MailHog not reachable at {H.MAILHOG_URL}")
    token = H.admin_token(client)
    if not token:
        pytest.skip("bootstrap admin not available")
    return H.provision_approved_user(client, token)


# --- Registration (now admin-gated) -----------------------------------------

def test_register_returns_pending(client):
    r = H.register(client, H.unique_email("reg"))
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "pending_review"
    assert "access_token" not in body


def test_register_duplicate_email_conflicts(client):
    email = H.unique_email("dup")
    assert H.register(client, email).status_code == 202
    r2 = H.register(client, email)
    assert r2.status_code == 409, r2.text


@pytest.mark.parametrize("bad_pw", ["short1A", "alllowercase1", "ALLUPPERCASE1", "NoDigitsHere"])
def test_register_rejects_weak_passwords(client, bad_pw):
    r = H.register(client, H.unique_email("weak"), password=bad_pw)
    assert r.status_code == 422, f"expected validation error for {bad_pw!r}, got {r.status_code}"


# --- Login ------------------------------------------------------------------

def test_login_pending_before_approval_403(client):
    email = H.unique_email("pending")
    H.register(client, email)
    r = H.login(client, email, STRONG_PW)
    assert r.status_code == 403, r.text
    assert H.error_code(r) == "AUTH_1005"  # ACCOUNT_PENDING_APPROVAL


def test_login_wrong_password_401(client):
    # Wrong password fails at credential check BEFORE the approval gate → 401.
    email = H.unique_email("wrongpw")
    H.register(client, email)
    r = H.login(client, email, "Wrong123!")
    assert r.status_code == 401, r.text


def test_login_unknown_user_401(client):
    r = H.login(client, H.unique_email("ghost"), STRONG_PW)
    assert r.status_code == 401, r.text


def test_login_success_after_approval(approved_user, client):
    r = H.login(client, approved_user["email"], approved_user["password"])
    assert r.status_code == 200, r.text
    assert r.json()["access_token"]
    assert r.json()["must_reset_password"] is False


# --- Brute-force lockout (requires Redis) -----------------------------------

def test_repeated_failures_trigger_lockout(client):
    """Wrong-password attempts (credential failures, pre-approval-gate) must 429.

    The throttle locks per-email AND per-IP. All tests reach the container through
    one Docker-gateway IP, so this test targets a UNIQUE synthetic client IP via
    X-Forwarded-For (which `_client_ip` honors) to avoid poisoning the shared IP
    that sibling tests (incl. admin logins) rely on.
    """
    import uuid as _uuid

    email = H.unique_email("lock")
    H.register(client, email)
    fake_ip = f"203.0.113.{_uuid.uuid4().int % 254 + 1}"  # TEST-NET-3, isolated
    xff = {"X-Forwarded-For": fake_ip}

    statuses = []
    for i in range(10):
        # Wrong password MUST be >= 8 chars, else LoginRequest validation returns
        # 422 before authenticate() runs and no failure is ever counted.
        r = H.login(client, email, f"wrongpass-{i}", headers=xff)
        statuses.append(r.status_code)
        if r.status_code == 429:
            assert "Retry-After" in r.headers
            break

    assert 429 in statuses, (
        f"expected a 429 lockout within 10 attempts, got {statuses}. "
        "Is Redis reachable from the API?"
    )


# --- Refresh + logout blacklist ---------------------------------------------

def test_refresh_returns_new_tokens(approved_user, client):
    login = H.login(client, approved_user["email"], approved_user["password"]).json()
    r = client.post(f"{AUTH}/refresh", json={"refresh_token": login["refresh_token"]})
    assert r.status_code == 200, r.text
    assert r.json()["access_token"]


def test_logout_revokes_access_token(approved_user, client):
    """A logged-out access token must be rejected by a protected endpoint."""
    access = approved_user["access_token"]
    headers = {"Authorization": f"Bearer {access}"}

    protected = H.USERS_ME
    before = client.get(protected, headers=headers)
    if before.status_code == 404:
        pytest.skip("protected probe path /api/users/me not present in this build")
    assert before.status_code == 200, before.text

    out = client.post(f"{AUTH}/logout", headers=headers)
    assert out.status_code == 200, out.text

    # Give Redis a beat, then confirm the token is now revoked.
    time.sleep(0.2)
    after = client.get(protected, headers=headers)
    assert after.status_code == 401, f"revoked token should be rejected, got {after.status_code}"
