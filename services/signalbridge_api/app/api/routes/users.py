"""
User routes.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.auth import get_current_active_user
from app.models import User
from app.contracts.user import UserProfile, UserProfileUpdate
from app.contracts.auth import PasswordChangeRequest
from app.contracts.common import SuccessResponse
from app.services.user import UserService

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get current user's profile.
    """
    user_service = UserService(db)
    return user_service.to_profile(current_user)


@router.patch("/me", response_model=UserProfile)
async def update_current_user_profile(
    data: UserProfileUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update current user's profile.
    """
    user_service = UserService(db)
    updated_user = await user_service.update_profile(current_user.id, data)
    return user_service.to_profile(updated_user)


@router.post("/me/password", response_model=SuccessResponse)
async def change_password(
    data: PasswordChangeRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Change current user's password.
    """
    user_service = UserService(db)
    await user_service.change_password(
        user_id=current_user.id,
        current_password=data.current_password.get_secret_value(),
        new_password=data.new_password.get_secret_value(),
    )

    return SuccessResponse(message="Password changed successfully")
