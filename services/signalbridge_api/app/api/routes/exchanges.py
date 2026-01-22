"""
Exchange credential routes.
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.auth import get_current_active_user
from app.models import User
from app.contracts.exchange import (
    ExchangeCredentials,
    ExchangeCredentialsCreate,
    ExchangeCredentialsUpdate,
    ExchangeBalance,
    ExchangeInfo,
    SupportedExchange,
    ExchangeValidationResult,
)
from app.contracts.common import SuccessResponse
from app.services.exchange import ExchangeService
from app.adapters import ExchangeAdapterFactory

router = APIRouter(prefix="/exchanges", tags=["Exchanges"])


@router.get("/supported", response_model=List[ExchangeInfo])
async def get_supported_exchanges():
    """
    Get list of supported exchanges.
    """
    supported = ExchangeAdapterFactory.get_supported_exchanges()

    return [
        ExchangeInfo(
            exchange=exchange,
            name=exchange.value.upper(),
            is_connected=False,
            supports_testnet=True,
            base_url=f"https://api.{exchange.value}.com",
            rate_limit=60,
            features=["spot", "market_orders", "limit_orders"],
        )
        for exchange in supported
    ]


@router.get("/credentials", response_model=List[ExchangeCredentials])
async def list_credentials(
    exchange: Optional[SupportedExchange] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List user's exchange credentials.
    """
    exchange_service = ExchangeService(db)
    credentials = await exchange_service.get_credentials(
        user_id=current_user.id,
        exchange=exchange,
    )

    return [exchange_service.to_response(cred) for cred in credentials]


@router.post("/credentials", response_model=ExchangeCredentials, status_code=201)
async def create_credential(
    data: ExchangeCredentialsCreate,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create new exchange credentials.
    """
    # Get client IP for audit
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)

    exchange_service = ExchangeService(db)
    credential = await exchange_service.create_credential(
        user_id=current_user.id,
        data=data,
        ip_address=client_ip,
    )

    return exchange_service.to_response(credential)


@router.get("/credentials/{credential_id}", response_model=ExchangeCredentials)
async def get_credential(
    credential_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific exchange credential.
    """
    exchange_service = ExchangeService(db)
    credentials = await exchange_service.get_credentials(
        user_id=current_user.id,
        credential_id=credential_id,
    )

    if not credentials:
        from app.core.exceptions import NotFoundError
        raise NotFoundError(
            message="Credential not found",
            resource_type="ExchangeCredential",
            resource_id=str(credential_id),
        ).to_http_exception()

    return exchange_service.to_response(credentials[0])


@router.patch("/credentials/{credential_id}", response_model=ExchangeCredentials)
async def update_credential(
    credential_id: UUID,
    data: ExchangeCredentialsUpdate,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update exchange credentials.
    """
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)

    exchange_service = ExchangeService(db)
    credential = await exchange_service.update_credential(
        user_id=current_user.id,
        credential_id=credential_id,
        data=data,
        ip_address=client_ip,
    )

    return exchange_service.to_response(credential)


@router.delete("/credentials/{credential_id}", response_model=SuccessResponse)
async def delete_credential(
    credential_id: UUID,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete exchange credentials.
    """
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else None)

    exchange_service = ExchangeService(db)
    await exchange_service.delete_credential(
        user_id=current_user.id,
        credential_id=credential_id,
        ip_address=client_ip,
    )

    return SuccessResponse(message="Credential deleted successfully")


@router.post("/credentials/{credential_id}/validate", response_model=ExchangeValidationResult)
async def validate_credential(
    credential_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Validate exchange credentials by testing connection.
    """
    exchange_service = ExchangeService(db)
    credentials = await exchange_service.get_credentials(
        user_id=current_user.id,
        credential_id=credential_id,
    )

    if not credentials:
        from app.core.exceptions import NotFoundError
        raise NotFoundError(
            message="Credential not found",
            resource_type="ExchangeCredential",
            resource_id=str(credential_id),
        ).to_http_exception()

    credential = credentials[0]
    is_valid = await exchange_service.validate_credential(
        user_id=current_user.id,
        credential_id=credential_id,
    )

    return ExchangeValidationResult(
        is_valid=is_valid,
        exchange=SupportedExchange(credential.exchange),
        permissions=["spot_trading"] if is_valid else [],
        error_message=None if is_valid else "Credential validation failed",
    )


@router.get("/credentials/{credential_id}/balances", response_model=List[ExchangeBalance])
async def get_balances(
    credential_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get account balances for an exchange credential.
    """
    exchange_service = ExchangeService(db)
    balances = await exchange_service.get_balances(
        user_id=current_user.id,
        credential_id=credential_id,
    )

    return balances
