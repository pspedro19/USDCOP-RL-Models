"""
Admin contracts — request/response shapes for the user-approval routes.

Kept separate from user.py so the admin surface (a small, privileged API) has an
explicit, self-documenting contract boundary.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .common import BaseContract
from .user import UserRole, UserStatus


class PendingUserItem(BaseContract):
    """A single row in the admin's review queue."""

    id: UUID
    email: str
    name: str | None = None
    status: UserStatus
    role: UserRole
    created_at: datetime


class RejectRequest(BaseModel):
    """Optional reason attached to a rejection (surfaced in the applicant email)."""

    reason: str | None = Field(default=None, max_length=500)


class AdminActionResponse(BaseContract):
    """Result of an approve/reject action."""

    id: UUID
    email: str
    status: UserStatus
    email_sent: bool = Field(
        description="Whether the notification email was dispatched. The state "
        "transition is authoritative regardless of email delivery."
    )
    message: str
