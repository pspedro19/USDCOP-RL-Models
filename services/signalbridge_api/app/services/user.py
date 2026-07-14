"""
User service for user management operations.
"""

import secrets
import string
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.auth import RegisterRequest
from app.contracts.user import UserProfile, UserProfileUpdate, UserRole, UserStatus
from app.core.exceptions import (
    AccountNotApprovedError,
    AuthenticationError,
    ConflictError,
    ErrorCode,
    NotFoundError,
)
from app.core.security import get_password_hash, verify_password
from app.models import TradingConfig, User


class UserService:
    """Service for user-related operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: UUID) -> User | None:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        result = await self.db.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()

    async def create(self, data: RegisterRequest) -> User:
        """
        Create a new user.

        Args:
            data: Registration data

        Returns:
            Created user

        Raises:
            ConflictError: If email already exists
        """
        # Check if email exists
        existing = await self.get_by_email(data.email)
        if existing:
            raise ConflictError(
                message="Email already registered",
                details={"email": data.email},
            )

        # Create user in the PENDING state — no access until an admin approves.
        # is_active stays False so even a status bug cannot yield a usable login.
        user = User(
            email=data.email.lower(),
            hashed_password=get_password_hash(data.password.get_secret_value()),
            name=data.name,
            is_active=False,
            is_verified=False,
            status=UserStatus.PENDING.value,
            role=UserRole.USER.value,
            must_reset_password=False,
        )

        self.db.add(user)
        await self.db.flush()

        # Create default trading config
        trading_config = TradingConfig(
            user_id=user.id,
            trading_enabled=False,
        )
        self.db.add(trading_config)

        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def authenticate(self, email: str, password: str) -> User:
        """
        Authenticate user with email and password.

        Args:
            email: User email
            password: User password

        Returns:
            Authenticated user

        Raises:
            AuthenticationError: If credentials are invalid
        """
        user = await self.get_by_email(email)

        if not user:
            raise AuthenticationError(
                message="Invalid email or password",
                error_code=ErrorCode.INVALID_CREDENTIALS,
            )

        if not verify_password(password, user.hashed_password):
            raise AuthenticationError(
                message="Invalid email or password",
                error_code=ErrorCode.INVALID_CREDENTIALS,
            )

        # Credentials are correct past this point — approval/enablement failures
        # are NOT brute-force failures, so they raise a distinct (403) exception
        # that the login route does not count toward the lockout.
        if user.status == UserStatus.PENDING.value:
            raise AccountNotApprovedError(
                message="Your account is pending administrator approval",
                error_code=ErrorCode.ACCOUNT_PENDING_APPROVAL,
            )
        if user.status == UserStatus.REJECTED.value:
            raise AccountNotApprovedError(
                message="Your registration was not approved",
                error_code=ErrorCode.ACCOUNT_REJECTED,
            )

        if not user.is_active:
            raise AuthenticationError(
                message="Account is disabled",
                error_code=ErrorCode.INSUFFICIENT_PERMISSIONS,
            )

        # Update last login
        user.last_login = datetime.utcnow()
        await self.db.commit()

        return user

    # ------------------------------------------------------------------ #
    # Admin-approval lifecycle
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_temp_password(length: int = 16) -> str:
        """Cryptographically-random temporary password that satisfies the
        password policy (upper + lower + digit)."""
        alphabet = string.ascii_letters + string.digits
        while True:
            pw = "".join(secrets.choice(alphabet) for _ in range(length))
            if (
                any(c.islower() for c in pw)
                and any(c.isupper() for c in pw)
                and any(c.isdigit() for c in pw)
            ):
                return pw

    async def list_by_status(self, status: UserStatus) -> list[User]:
        """List users in a given approval state (newest first)."""
        result = await self.db.execute(
            select(User)
            .where(User.status == status.value)
            .order_by(User.created_at.desc())
        )
        return list(result.scalars().all())

    async def _require_pending(self, user_id: UUID) -> User:
        user = await self.get_by_id(user_id)
        if not user:
            raise NotFoundError(
                message="User not found", resource_type="User", resource_id=str(user_id)
            )
        if user.status != UserStatus.PENDING.value:
            raise ConflictError(
                message=f"User is already {user.status}",
                details={"status": user.status},
            )
        return user

    async def approve(self, user_id: UUID, admin_id: UUID) -> tuple[User, str]:
        """Approve a pending user: activate, issue a temporary password, and force
        a reset on first login. Returns (user, temp_password) — the caller emails
        the temp password; it is never persisted in clear text."""
        user = await self._require_pending(user_id)
        temp_password = self.generate_temp_password()

        user.status = UserStatus.APPROVED.value
        user.is_active = True
        user.hashed_password = get_password_hash(temp_password)
        user.must_reset_password = True
        user.approved_by = admin_id
        user.approved_at = datetime.utcnow()
        user.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(user)
        return user, temp_password

    async def reject(
        self, user_id: UUID, admin_id: UUID, reason: str | None = None
    ) -> User:
        """Reject a pending user."""
        user = await self._require_pending(user_id)
        user.status = UserStatus.REJECTED.value
        user.is_active = False
        user.rejected_at = datetime.utcnow()
        user.rejection_reason = reason
        user.approved_by = admin_id
        user.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def reset_password(
        self, user_id: UUID, current_password: str, new_password: str
    ) -> User:
        """Consume a temporary (or current) password and set a new one, clearing
        the must_reset_password flag."""
        user = await self.get_by_id(user_id)
        if not user:
            raise NotFoundError(
                message="User not found", resource_type="User", resource_id=str(user_id)
            )
        if not verify_password(current_password, user.hashed_password):
            raise AuthenticationError(
                message="Current password is incorrect",
                error_code=ErrorCode.INVALID_CREDENTIALS,
            )
        user.hashed_password = get_password_hash(new_password)
        user.must_reset_password = False
        user.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def bootstrap_admin(
        self, email: str, password: str, name: str = "Administrator"
    ) -> User:
        """Idempotently ensure an approved admin exists. If the email is present
        it is promoted to an active, approved admin (password left untouched);
        otherwise it is created. Safe to call on every startup."""
        existing = await self.get_by_email(email)
        if existing:
            changed = False
            if existing.role != UserRole.ADMIN.value:
                existing.role = UserRole.ADMIN.value
                changed = True
            if existing.status != UserStatus.APPROVED.value:
                existing.status = UserStatus.APPROVED.value
                changed = True
            if not existing.is_active:
                existing.is_active = True
                changed = True
            if changed:
                existing.updated_at = datetime.utcnow()
                await self.db.commit()
                await self.db.refresh(existing)
            return existing

        admin = User(
            email=email.lower(),
            hashed_password=get_password_hash(password),
            name=name,
            is_active=True,
            is_verified=True,
            status=UserStatus.APPROVED.value,
            role=UserRole.ADMIN.value,
            must_reset_password=False,
        )
        self.db.add(admin)
        await self.db.flush()
        self.db.add(TradingConfig(user_id=admin.id, trading_enabled=False))
        await self.db.commit()
        await self.db.refresh(admin)
        return admin

    async def bootstrap_guest(
        self, email: str, password: str, name: str = "Invitado"
    ) -> User:
        """Idempotently ensure the shared demo/guest account exists: role 'free',
        approved, active, is_test=True. Unlike bootstrap_admin, the password IS
        reset to the env value on every startup — the dashboard's guest endpoint
        logs in with these exact credentials server-side, so env must stay the
        truth. Safe to call on every startup."""
        from sqlalchemy import text

        existing = await self.get_by_email(email)
        if existing:
            existing.hashed_password = get_password_hash(password)
            existing.is_active = True
            existing.status = UserStatus.APPROVED.value
            existing.must_reset_password = False
            existing.updated_at = datetime.utcnow()
            guest = existing
        else:
            guest = User(
                email=email.lower(),
                hashed_password=get_password_hash(password),
                name=name,
                is_active=True,
                is_verified=True,
                status=UserStatus.APPROVED.value,
                role=UserRole.USER.value,  # overwritten to 'free' below (raw SQL)
                must_reset_password=False,
            )
            self.db.add(guest)
            await self.db.flush()
            self.db.add(TradingConfig(user_id=guest.id, trading_enabled=False))

        # role 'free' is outside the UserRole enum and is_test is not mapped on the
        # model — set both via SQL (the 056 trigger also flags *.local, belt+braces).
        await self.db.execute(
            text(
                "UPDATE sb_users SET role='free', is_test=TRUE WHERE email=:email"
            ),
            {"email": email.lower()},
        )
        await self.db.commit()
        await self.db.refresh(guest)
        return guest

    async def update_profile(
        self,
        user_id: UUID,
        data: UserProfileUpdate,
    ) -> User:
        """
        Update user profile.

        Args:
            user_id: User ID
            data: Update data

        Returns:
            Updated user

        Raises:
            NotFoundError: If user not found
        """
        user = await self.get_by_id(user_id)

        if not user:
            raise NotFoundError(
                message="User not found",
                resource_type="User",
                resource_id=str(user_id),
            )

        # Check email uniqueness if changing
        if data.email and data.email.lower() != user.email:
            existing = await self.get_by_email(data.email)
            if existing:
                raise ConflictError(
                    message="Email already in use",
                    details={"email": data.email},
                )
            user.email = data.email.lower()

        if data.name:
            user.name = data.name

        user.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def change_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str,
    ) -> bool:
        """
        Change user password.

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password

        Returns:
            True if successful

        Raises:
            NotFoundError: If user not found
            AuthenticationError: If current password is wrong
        """
        user = await self.get_by_id(user_id)

        if not user:
            raise NotFoundError(
                message="User not found",
                resource_type="User",
                resource_id=str(user_id),
            )

        if not verify_password(current_password, user.hashed_password):
            raise AuthenticationError(
                message="Current password is incorrect",
                error_code=ErrorCode.INVALID_CREDENTIALS,
            )

        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        await self.db.commit()

        return True

    def to_profile(self, user: User) -> UserProfile:
        """Convert User model to UserProfile response."""
        return UserProfile(
            id=user.id,
            email=user.email,
            name=user.name,
            created_at=user.created_at,
            last_login=user.last_login,
            is_active=user.is_active,
            is_verified=user.is_verified,
            status=getattr(user, "status", UserStatus.APPROVED.value),
            role=getattr(user, "role", UserRole.USER.value),
            must_reset_password=getattr(user, "must_reset_password", False),
        )
