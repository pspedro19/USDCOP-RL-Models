"""
Email service - account-approval notifications.

Design (ports & adapters):
    - ``EmailMessage``      value object (to / subject / text / html).
    - ``EmailSender``       port (Protocol). Transport only - knows nothing about users.
    - ``SMTPEmailSender``   adapter over smtplib (prod + MailHog in tests).
    - ``ConsoleEmailSender``adapter that logs (dev / no SMTP configured).
    - ``AccountEmailTemplates`` SSOT for the email copy (what to say).
    - ``AccountNotifier``   facade the routes depend on (what to send, when).

Adding a provider (SendGrid, SES, …) = a new EmailSender adapter; no caller
changes (Open/Closed). smtplib is blocking, so sends run in a threadpool to keep
the async event loop free. Delivery is best-effort: a send failure is logged and
returns False - it NEVER rolls back the state transition that triggered it.
"""

from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Protocol

import structlog
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings, settings

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Value object + port
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmailMessage:
    to: str
    subject: str
    text: str
    html: str | None = None


class EmailSender(Protocol):
    """Transport port. Implementations must be safe to call from a threadpool."""

    def send(self, message: EmailMessage) -> bool:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
class SMTPEmailSender:
    """Send via SMTP. Works with a real relay or a MailHog sink (no auth/TLS)."""

    def __init__(
        self,
        host: str,
        port: int,
        from_email: str,
        user: str | None = None,
        password: str | None = None,
        use_tls: bool = False,
        timeout: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._from = from_email
        self._user = user
        self._password = password
        self._use_tls = use_tls
        self._timeout = timeout

    def send(self, message: EmailMessage) -> bool:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = message.subject
        msg["From"] = self._from
        msg["To"] = message.to
        # Use us-ascii for ASCII bodies (7bit → the raw MIME stays human/regex
        # readable, e.g. for MailHog assertions); fall back to utf-8 otherwise.
        text_cs = "us-ascii" if message.text.isascii() else "utf-8"
        msg.attach(MIMEText(message.text, "plain", text_cs))
        if message.html:
            html_cs = "us-ascii" if message.html.isascii() else "utf-8"
            msg.attach(MIMEText(message.html, "html", html_cs))

        with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as server:
            if self._use_tls:
                server.starttls()
            if self._user and self._password:
                server.login(self._user, self._password)
            server.sendmail(self._from, [message.to], msg.as_string())
        logger.info("email.sent", to=message.to, subject=message.subject)
        return True


class ConsoleEmailSender:
    """Fallback that logs the message instead of sending. Used when no SMTP host
    is configured so the flow degrades gracefully in local/dev runs."""

    def send(self, message: EmailMessage) -> bool:
        logger.info(
            "email.console",
            to=message.to,
            subject=message.subject,
            body=message.text,
        )
        return True


def build_email_sender(cfg: Settings) -> EmailSender:
    """Factory: SMTP when a host is configured, else the console fallback."""
    if cfg.smtp_host:
        return SMTPEmailSender(
            host=cfg.smtp_host,
            port=cfg.smtp_port,
            from_email=cfg.email_from,
            user=cfg.smtp_user,
            password=cfg.smtp_password,
            use_tls=cfg.smtp_use_tls,
        )
    return ConsoleEmailSender()


# --------------------------------------------------------------------------- #
# Templates (SSOT for copy)
# --------------------------------------------------------------------------- #
class AccountEmailTemplates:
    """Single source of truth for account-lifecycle email copy."""

    @staticmethod
    def submitted(name: str) -> tuple[str, str, str]:
        subject = "SignalBridge - registration received"
        text = (
            f"Hello {name},\n\n"
            "Thank you for registering with SignalBridge. Your request has been "
            "received and an administrator is now reviewing it.\n\n"
            "You will receive another email once your account is approved, which "
            "will include your username and a temporary password.\n\n"
            "- The SignalBridge team"
        )
        html = _wrap_html(
            f"<p>Hello {name},</p>"
            "<p>Thank you for registering with <b>SignalBridge</b>. Your request "
            "has been received and an administrator is now reviewing it.</p>"
            "<p>You will receive another email once your account is approved, "
            "which will include your username and a temporary password.</p>"
            "<p>- The SignalBridge team</p>"
        )
        return subject, text, html

    @staticmethod
    def approved(name: str, email: str, temp_password: str, reset_url: str) -> tuple[str, str, str]:
        subject = "SignalBridge - your account is approved"
        text = (
            f"Hello {name},\n\n"
            "Your SignalBridge account has been approved. You can now sign in "
            "with the temporary credentials below and you will be prompted to set "
            "a new password.\n\n"
            f"  Username: {email}\n"
            f"  Temporary password: {temp_password}\n\n"
            f"Set your new password here: {reset_url}\n\n"
            "For your security, this temporary password must be changed on first "
            "login.\n\n"
            "- The SignalBridge team"
        )
        html = _wrap_html(
            f"<p>Hello {name},</p>"
            "<p>Your <b>SignalBridge</b> account has been approved. Sign in with "
            "the temporary credentials below; you will be prompted to set a new "
            "password.</p>"
            f"<ul><li><b>Username:</b> {email}</li>"
            f"<li><b>Temporary password:</b> <code>{temp_password}</code></li></ul>"
            f'<p><a href="{reset_url}">Set your new password</a></p>'
            "<p>For your security, this temporary password must be changed on "
            "first login.</p>"
            "<p>- The SignalBridge team</p>"
        )
        return subject, text, html

    @staticmethod
    def rejected(name: str, reason: str | None) -> tuple[str, str, str]:
        subject = "SignalBridge - registration update"
        reason_line = f"\n\nReason: {reason}" if reason else ""
        text = (
            f"Hello {name},\n\n"
            "After review, your SignalBridge registration was not approved at this "
            f"time.{reason_line}\n\n"
            "If you believe this is a mistake, please contact the administrator.\n\n"
            "- The SignalBridge team"
        )
        reason_html = f"<p><b>Reason:</b> {reason}</p>" if reason else ""
        html = _wrap_html(
            f"<p>Hello {name},</p>"
            "<p>After review, your <b>SignalBridge</b> registration was not "
            "approved at this time.</p>"
            f"{reason_html}"
            "<p>If you believe this is a mistake, please contact the "
            "administrator.</p>"
            "<p>- The SignalBridge team</p>"
        )
        return subject, text, html


def _wrap_html(body: str) -> str:
    return (
        '<html><body style="font-family:Arial,Helvetica,sans-serif;'
        'color:#1a1a1a;line-height:1.5">' + body + "</body></html>"
    )


# --------------------------------------------------------------------------- #
# Facade the routes depend on
# --------------------------------------------------------------------------- #
class AccountNotifier:
    """High-level account notifications. Composes a transport (EmailSender) with
    the templates and builds links from the public URL. Best-effort: every method
    returns whether the email was dispatched and swallows/logs transport errors."""

    def __init__(self, sender: EmailSender, public_url: str) -> None:
        self._sender = sender
        self._public_url = public_url.rstrip("/")

    @property
    def reset_url(self) -> str:
        return f"{self._public_url}/reset-password"

    async def _send(self, message: EmailMessage) -> bool:
        try:
            return await run_in_threadpool(self._sender.send, message)
        except Exception as exc:
            logger.warning("email.send_failed", to=message.to, error=str(exc))
            return False

    async def registration_submitted(self, to: str, name: str) -> bool:
        subject, text, html = AccountEmailTemplates.submitted(name)
        return await self._send(EmailMessage(to=to, subject=subject, text=text, html=html))

    async def registration_approved(self, to: str, name: str, temp_password: str) -> bool:
        subject, text, html = AccountEmailTemplates.approved(
            name=name, email=to, temp_password=temp_password, reset_url=self.reset_url
        )
        return await self._send(EmailMessage(to=to, subject=subject, text=text, html=html))

    async def registration_rejected(self, to: str, name: str, reason: str | None) -> bool:
        subject, text, html = AccountEmailTemplates.rejected(name, reason)
        return await self._send(EmailMessage(to=to, subject=subject, text=text, html=html))


def get_notifier() -> AccountNotifier:
    """FastAPI dependency provider for the account notifier."""
    return AccountNotifier(build_email_sender(settings), settings.app_public_url)
