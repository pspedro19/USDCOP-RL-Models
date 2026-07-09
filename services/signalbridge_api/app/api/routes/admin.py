"""
Admin routes — user-approval queue.

All routes are gated by ``require_admin`` (role == admin). The role is read from
the database-backed current user, so a demotion takes effect immediately without
waiting for a token to expire.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.admin import (
    AdminActionResponse,
    PendingUserItem,
    RejectRequest,
)
from app.contracts.user import UserRole, UserStatus
from app.core.database import get_db
from app.core.exceptions import AuthorizationError
from app.middleware.auth import get_current_active_user
from app.models import User
from app.services.email import AccountNotifier, get_notifier
from app.services.user import UserService

router = APIRouter(prefix="/admin", tags=["Admin"])


async def require_admin(
    user: User = Depends(get_current_active_user),
) -> User:
    """Authorize an administrator. 403 for any non-admin principal."""
    if getattr(user, "role", UserRole.USER.value) != UserRole.ADMIN.value:
        raise AuthorizationError(message="Administrator role required")
    return user


@router.get("/users", response_model=list[PendingUserItem])
async def list_users(
    status: UserStatus = Query(
        default=UserStatus.PENDING, description="Filter by approval status"
    ),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List users by approval status (defaults to the pending review queue)."""
    users = await UserService(db).list_by_status(status)
    return [
        PendingUserItem(
            id=u.id,
            email=u.email,
            name=u.name,
            status=UserStatus(u.status),
            role=UserRole(u.role),
            created_at=u.created_at,
        )
        for u in users
    ]


@router.post("/users/{user_id}/approve", response_model=AdminActionResponse)
async def approve_user(
    user_id: UUID,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    notifier: AccountNotifier = Depends(get_notifier),
):
    """Approve a pending user and email them a temporary password + reset link."""
    service = UserService(db)
    user, temp_password = await service.approve(user_id=user_id, admin_id=admin.id)

    email_sent = await notifier.registration_approved(
        to=user.email, name=user.name or user.email, temp_password=temp_password
    )

    return AdminActionResponse(
        id=user.id,
        email=user.email,
        status=UserStatus(user.status),
        email_sent=email_sent,
        message="User approved; temporary-password email dispatched.",
    )


@router.post("/users/{user_id}/reject", response_model=AdminActionResponse)
async def reject_user(
    user_id: UUID,
    body: RejectRequest | None = None,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    notifier: AccountNotifier = Depends(get_notifier),
):
    """Reject a pending user and email them the outcome."""
    reason = body.reason if body else None
    service = UserService(db)
    user = await service.reject(user_id=user_id, admin_id=admin.id, reason=reason)

    email_sent = await notifier.registration_rejected(
        to=user.email, name=user.name or user.email, reason=reason
    )

    return AdminActionResponse(
        id=user.id,
        email=user.email,
        status=UserStatus(user.status),
        email_sent=email_sent,
        message="User rejected; notification email dispatched.",
    )
