"""
Shared helpers for the admin-approval + email E2E (DRY across auth test suites).

They talk to a RUNNING SignalBridge API and a MailHog sink. Environment:
    AUTH_BASE_URL            default http://localhost:8085
    MAILHOG_URL              default http://localhost:8025
    ADMIN_BOOTSTRAP_EMAIL    default admin@signalbridge.io      (matches docker-compose.mailhog.yml)
    ADMIN_BOOTSTRAP_PASSWORD default Admin!Bootstrap1

Everything degrades to a skip if the API / MailHog / bootstrap admin is absent, so
the tests are safe to collect anywhere.
"""

from __future__ import annotations

import os
import re
import time
import uuid

import httpx

BASE_URL = os.getenv("AUTH_BASE_URL", "http://localhost:8085")
AUTH = f"{BASE_URL}/api/auth"
ADMIN_API = f"{BASE_URL}/api/admin"
USERS_ME = f"{BASE_URL}/api/users/me"
MAILHOG_URL = os.getenv("MAILHOG_URL", "http://localhost:8025").rstrip("/")
ADMIN_EMAIL = os.getenv("ADMIN_BOOTSTRAP_EMAIL", "admin@signalbridge.io")
ADMIN_PASSWORD = os.getenv("ADMIN_BOOTSTRAP_PASSWORD", "Admin!Bootstrap1")

STRONG_PW = "Str0ngPass!"
_TEMP_PW_RE = re.compile(r"[Tt]emporary password:\s*([^\s<]+)")


# --- reachability ----------------------------------------------------------- #
def api_up() -> bool:
    try:
        httpx.get(f"{BASE_URL}/health", timeout=2.0)
        return True
    except Exception:
        return False


def mailhog_up() -> bool:
    try:
        return httpx.get(f"{MAILHOG_URL}/api/v2/messages?limit=1", timeout=2.0).status_code == 200
    except Exception:
        return False


def unique_email(tag: str) -> str:
    return f"pytest_{uuid.uuid4().hex[:8]}_{tag}@example.com"


def error_code(response: httpx.Response) -> str | None:
    """Extract the machine-readable error code regardless of envelope shape.

    SignalBridgeException errors serialize the code at the top level
    (``{"code": ...}``); raw FastAPI HTTPExceptions nest it under ``detail``.
    """
    try:
        body = response.json()
    except Exception:
        return None
    if isinstance(body.get("code"), str):
        return body["code"]
    detail = body.get("detail")
    if isinstance(detail, dict):
        return detail.get("code")
    return None


# --- API calls -------------------------------------------------------------- #
def register(client: httpx.Client, email: str, password: str = STRONG_PW, name: str = "Pytest User"):
    return client.post(f"{AUTH}/register", json={"email": email, "password": password, "name": name})


def login(client: httpx.Client, email: str, password: str, headers: dict | None = None):
    return client.post(
        f"{AUTH}/login", json={"email": email, "password": password}, headers=headers
    )


def admin_token(client: httpx.Client) -> str | None:
    """Log in the bootstrap admin; None if unavailable (→ caller skips)."""
    r = login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    if r.status_code != 200:
        return None
    return r.json().get("access_token")


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def list_users(client: httpx.Client, token: str, status: str = "pending"):
    return client.get(f"{ADMIN_API}/users", params={"status": status}, headers=_auth_headers(token))


def find_user_id(client: httpx.Client, token: str, email: str, status: str = "pending") -> str | None:
    r = list_users(client, token, status)
    if r.status_code != 200:
        return None
    for u in r.json():
        if u["email"].lower() == email.lower():
            return u["id"]
    return None


def approve(client: httpx.Client, token: str, user_id: str):
    return client.post(f"{ADMIN_API}/users/{user_id}/approve", headers=_auth_headers(token))


def reject(client: httpx.Client, token: str, user_id: str, reason: str | None = None):
    return client.post(
        f"{ADMIN_API}/users/{user_id}/reject",
        json={"reason": reason} if reason is not None else {},
        headers=_auth_headers(token),
    )


def reset_password(client: httpx.Client, token: str, current_password: str, new_password: str):
    return client.post(
        f"{AUTH}/reset-password",
        json={"current_password": current_password, "new_password": new_password},
        headers=_auth_headers(token),
    )


# --- MailHog ---------------------------------------------------------------- #
def _all_messages() -> list[dict]:
    r = httpx.get(f"{MAILHOG_URL}/api/v2/messages?limit=200", timeout=5.0)
    r.raise_for_status()
    return r.json().get("items", [])


def _message_text(item: dict) -> str:
    """Concatenate the raw + content body so a substring/regex search hits the
    plain part regardless of MailHog's storage shape."""
    parts = [
        item.get("Raw", {}).get("Data", "") or "",
        item.get("Content", {}).get("Body", "") or "",
    ]
    for p in item.get("MIME", {}).get("Parts", []) or []:
        parts.append(p.get("Body", "") or "")
    return "\n".join(parts)


def wait_for_email(to_email: str, contains: str | None = None, timeout: float = 15.0) -> str:
    """Return the newest message body addressed to `to_email` (optionally also
    containing `contains`). Raises AssertionError on timeout."""
    deadline = time.time() + timeout
    last_err = "no message found"
    while time.time() < deadline:
        try:
            for item in _all_messages():  # MailHog returns newest-first
                body = _message_text(item)
                if to_email.lower() in body.lower() and (contains is None or contains in body):
                    return body
        except Exception as exc:  # transient MailHog hiccup
            last_err = str(exc)
        time.sleep(0.5)
    raise AssertionError(f"email to {to_email} (contains={contains!r}) not received: {last_err}")


def extract_temp_password(email_body: str) -> str:
    m = _TEMP_PW_RE.search(email_body)
    assert m, f"temporary password not found in email body:\n{email_body[:800]}"
    return m.group(1)


# --- composite: provision a fully-usable, approved user --------------------- #
def provision_approved_user(
    client: httpx.Client, token: str, new_password: str = STRONG_PW
) -> dict:
    """register → admin approve → read temp pw from email → reset → return a dict
    with {email, password, access_token} for an active, ready-to-use account."""
    email = unique_email("provision")
    assert register(client, email).status_code == 202
    user_id = find_user_id(client, token, email)
    assert user_id, f"pending user {email} not visible to admin"
    assert approve(client, token, user_id).status_code == 200

    body = wait_for_email(email, contains="Temporary password:")
    temp_pw = extract_temp_password(body)

    first = login(client, email, temp_pw)
    assert first.status_code == 200, first.text
    assert first.json().get("must_reset_password") is True
    temp_access = first.json()["access_token"]

    assert reset_password(client, temp_access, temp_pw, new_password).status_code == 200
    final = login(client, email, new_password)
    assert final.status_code == 200, final.text
    return {"email": email, "password": new_password, "access_token": final.json()["access_token"]}
