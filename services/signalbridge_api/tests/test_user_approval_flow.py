"""
E2E: admin-gated registration + email flow.

Exercises the full lifecycle against a RUNNING SignalBridge (SIGNALBRIDGE_DEV_MODE
=false) with a MailHog sink and a bootstrap admin — i.e. the stack brought up with:

    docker compose -f docker-compose.yml -f docker-compose.mailhog.yml up -d

Run:
    pytest services/signalbridge_api/tests/test_user_approval_flow.py -v

Skips cleanly if the API, MailHog, or the bootstrap admin is not available.
"""

from __future__ import annotations

import _approval_helpers as H
import httpx
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.skipif(not H.api_up(), reason=f"SignalBridge API not reachable at {H.BASE_URL}"),
    pytest.mark.skipif(not H.mailhog_up(), reason=f"MailHog not reachable at {H.MAILHOG_URL}"),
]


@pytest.fixture()
def client():
    with httpx.Client(timeout=15.0) as c:
        yield c


@pytest.fixture()
def admin(client):
    token = H.admin_token(client)
    if not token:
        pytest.skip("bootstrap admin not available (set ADMIN_BOOTSTRAP_* + restart signalbridge)")
    return token


def test_full_register_approve_reset_login_flow(client, admin):
    email = H.unique_email("flow")

    # 1. Register → 202 pending, no tokens, "under review" email sent.
    r = H.register(client, email)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "pending_review"
    assert "access_token" not in body
    submitted = H.wait_for_email(email, contains="reviewing")
    assert "reviewing" in submitted

    # 2. Login before approval → 403 PENDING_APPROVAL.
    pre = H.login(client, email, H.STRONG_PW)
    assert pre.status_code == 403, pre.text
    assert H.error_code(pre) == "AUTH_1005"

    # 3. Admin sees the user in the pending queue and approves.
    user_id = H.find_user_id(client, admin, email)
    assert user_id, "registered user not visible in the admin pending queue"
    appr = H.approve(client, admin, user_id)
    assert appr.status_code == 200, appr.text
    assert appr.json()["status"] == "approved"
    assert appr.json()["email_sent"] is True

    # ... temp-password email arrives; parse it and confirm the reset link.
    approved_mail = H.wait_for_email(email, contains="Temporary password:")
    temp_pw = H.extract_temp_password(approved_mail)
    assert "/reset-password" in approved_mail  # reset link present

    # 4. Login with the temporary password → success, must_reset_password flag set.
    first = H.login(client, email, temp_pw)
    assert first.status_code == 200, first.text
    assert first.json()["must_reset_password"] is True
    temp_access = first.json()["access_token"]

    # 5. Reset password → then login with the new password → full access.
    rp = H.reset_password(client, temp_access, temp_pw, H.STRONG_PW)
    assert rp.status_code == 200, rp.text
    assert rp.json()["must_reset_password"] is False

    final = H.login(client, email, H.STRONG_PW)
    assert final.status_code == 200
    assert final.json()["must_reset_password"] is False
    me = client.get(
        H.USERS_ME,
        headers={"Authorization": f"Bearer {final.json()['access_token']}"},
    )
    assert me.status_code == 200, me.text
    assert me.json()["email"].lower() == email.lower()
    assert me.json()["status"] == "approved"


def test_reject_path_blocks_login_and_emails(client, admin):
    email = H.unique_email("reject")
    assert H.register(client, email).status_code == 202

    user_id = H.find_user_id(client, admin, email)
    assert user_id
    rej = H.reject(client, admin, user_id, reason="incomplete KYC")
    assert rej.status_code == 200, rej.text
    assert rej.json()["status"] == "rejected"

    # Rejected account cannot log in (correct password) → 403 ACCOUNT_REJECTED.
    r = H.login(client, email, H.STRONG_PW)
    assert r.status_code == 403, r.text
    assert H.error_code(r) == "AUTH_1006"

    # Rejection email delivered with the reason.
    mail = H.wait_for_email(email, contains="not")
    assert "incomplete KYC" in mail


def test_non_admin_cannot_reach_admin_routes(client, admin):
    """A freshly-provisioned regular user is forbidden from the admin surface."""
    user = H.provision_approved_user(client, admin)
    r = client.get(
        f"{H.ADMIN_API}/users",
        params={"status": "pending"},
        headers={"Authorization": f"Bearer {user['access_token']}"},
    )
    assert r.status_code == 403, r.text
    assert H.error_code(r) == "AUTH_1004"


def test_approve_is_idempotent_guard(client, admin):
    """Approving an already-approved user is a 409 conflict, not a double-issue."""
    email = H.unique_email("dbl")
    assert H.register(client, email).status_code == 202
    uid = H.find_user_id(client, admin, email)
    assert H.approve(client, admin, uid).status_code == 200
    again = H.approve(client, admin, uid)
    assert again.status_code == 409, again.text
