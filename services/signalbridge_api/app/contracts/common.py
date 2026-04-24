"""
Common contracts used across the application.
"""

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseContract(BaseModel):
    """Base contract with common configuration."""

    class Config:
        from_attributes = True
        populate_by_name = True
        use_enum_values = True


class TimestampMixin(BaseModel):
    """Mixin for timestamps."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""

    items: list[T]
    total: int = Field(ge=0)
    page: int = Field(ge=1, default=1)
    limit: int = Field(ge=1, le=100, default=20)
    has_more: bool = False

    @property
    def total_pages(self) -> int:
        return (self.total + self.limit - 1) // self.limit if self.limit > 0 else 0


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool = True
    message: str = "Operation completed successfully"
    data: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Generic error response."""

    error: bool = True
    code: str
    message: str
    details: dict[str, Any] | None = None


class PaginationParams(BaseModel):
    """Query parameters for pagination."""

    page: int = Field(ge=1, default=1)
    limit: int = Field(ge=1, le=100, default=20)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.limit
