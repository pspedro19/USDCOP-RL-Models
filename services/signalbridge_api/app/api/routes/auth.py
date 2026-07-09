"""
Authentication routes.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.auth import (
    AuthToken,
    LoginRequest,
    PasswordResetRequest,
    RegisterRequest,
    RegisterResponse,
    TokenRefreshRequest,
)
from app.contracts.user import UserProfile
from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import AuthenticationError, ErrorCode
from app.core.login_security import LoginThrottle, TokenBlacklist
from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
)
from app.middleware.auth import get_current_active_user
from app.models import User
from app.services.email import AccountNotifier, get_notifier
from app.services.user import UserService

router = APIRouter(prefix="/auth", tags=["Authentication"])

_bearer = HTTPBearer(auto_error=False)


def _is_trusted_proxy(peer: str | None) -> bool:
    """Only private/docker-network peers may assert forwarded headers (audit A8-13)."""
    if not peer:
        return False
    import ipaddress

    try:
        return ipaddress.ip_address(peer).is_private or peer == "127.0.0.1"
    except ValueError:
        return False


def _client_ip(request: Request) -> str:
    """Client IP for lockout accounting. Forwarded headers are honored ONLY when the
    direct peer is a trusted (private-network) proxy — an internet client spoofing
    X-Forwarded-For must not be able to shift its lockout bucket (audit A8-13)."""
    peer = request.client.host if request.client else None
    if _is_trusted_proxy(peer):
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
    return peer or "unknown"


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def register(
    data: RegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    notifier: AccountNotifier = Depends(get_notifier),
):
    """
    Register a new user account (admin-approval flow).

    The account is created in the PENDING state and NO tokens are returned. The
    applicant is emailed that an administrator is reviewing the request; access is
    granted only after an admin approves (see POST /api/admin/users/{id}/approve).

    Per-IP throttle (audit A8-03): reuses the Redis lockout — >5 registrations from
    one IP within 15 min locks that IP out of /register (degrades open if Redis down,
    same documented posture as login; spam only ever fills the admin PENDING queue).
    """
    ip = _client_ip(request)
    locked_for = await LoginThrottle.check_locked("register", ip)
    if locked_for > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": True, "message": "Demasiados registros desde esta IP; "
                    f"intenta de nuevo en {locked_for}s"},
        )
    await LoginThrottle.record_failure("register", ip)

    user_service = UserService(db)

    # Create pending user (raises ConflictError on duplicate email).
    user = await user_service.create(data)

    # Best-effort notification — never blocks/rolls back the registration.
    await notifier.registration_submitted(to=user.email, name=user.name or user.email)

    return RegisterResponse()


@router.post("/login", response_model=AuthToken)
async def login(
    data: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate user and return tokens.

    Protected by brute-force lockout: after repeated failures an identity
    (email + client IP) is locked for a cooldown window and receives HTTP 429.
    """
    email = data.email
    ip = _client_ip(request)

    # 1. Reject early if this identity/IP is currently locked out.
    locked_for = await LoginThrottle.check_locked(email, ip)
    if locked_for > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": True,
                "code": ErrorCode.RATE_LIMITED.value,
                "message": "Too many failed login attempts. Try again later.",
                "details": {"retry_after": locked_for},
            },
            headers={"Retry-After": str(locked_for)},
        )

    user_service = UserService(db)

    # 2. Authenticate; record failures for lockout accounting.
    try:
        user = await user_service.authenticate(
            email=email,
            password=data.password.get_secret_value(),
        )
    except AuthenticationError:
        await LoginThrottle.record_failure(email, ip)
        raise

    # 3. Success clears the counters.
    await LoginThrottle.clear(email, ip)

    token_data = {"sub": str(user.id), "email": user.email, "role": user.role}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return AuthToken(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        must_reset_password=bool(getattr(user, "must_reset_password", False)),
    )


@router.post("/reset-password", response_model=UserProfile)
async def reset_password(
    data: PasswordResetRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Set a new password, consuming the current (possibly admin-issued temporary)
    one. Clears the ``must_reset_password`` flag so the account becomes fully
    usable. Requires a valid access token (issued at login with the temp password).
    """
    user_service = UserService(db)
    user = await user_service.reset_password(
        user_id=current_user.id,
        current_password=data.current_password.get_secret_value(),
        new_password=data.new_password.get_secret_value(),
    )
    return user_service.to_profile(user)


@router.post("/refresh", response_model=AuthToken)
async def refresh_token(
    data: TokenRefreshRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Refresh access token using refresh token.

    Returns new access and refresh tokens.
    """
    # Verify refresh token
    payload = verify_token(data.refresh_token, token_type="refresh")

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": ErrorCode.TOKEN_INVALID.value,
                "message": "Invalid or expired refresh token",
            },
        )

    # Reject rotated-out (blacklisted) refresh tokens — one-time-use rotation (audit A8-06).
    if payload.get("jti") and await TokenBlacklist.is_blacklisted(payload["jti"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": ErrorCode.TOKEN_INVALID.value,
                "message": "Refresh token already used (rotated)",
            },
        )

    user_id = payload.get("sub")

    # Verify user still exists and is active
    user_service = UserService(db)
    user = await user_service.get_by_id(user_id)

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": ErrorCode.TOKEN_INVALID.value,
                "message": "User not found or inactive",
            },
        )

    # Rotate: blacklist the PRESENTED refresh token before issuing new ones, so a
    # stolen refresh token cannot be replayed for its full 7-day lifetime (audit A8-06).
    if payload.get("jti"):
        import time as _time

        _ttl = int(payload.get("exp", 0)) - int(_time.time())
        await TokenBlacklist.add(payload["jti"], max(_ttl, 0))

    # Generate new tokens
    token_data = {"sub": str(user.id), "email": user.email}

    access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token(token_data)

    return AuthToken(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
):
    """
    Logout user by revoking the presented access token.

    The token's `jti` is added to a Redis blacklist with a TTL equal to its
    remaining lifetime, so `get_current_user` rejects it on subsequent requests.
    Idempotent: missing/invalid tokens are treated as already logged out.
    """
    if credentials:
        payload = verify_token(credentials.credentials, token_type="access")
        if payload and payload.get("jti"):
            import time

            ttl = int(payload.get("exp", 0)) - int(time.time())
            await TokenBlacklist.add(payload["jti"], max(ttl, 0))

    return {"message": "Logged out successfully"}
