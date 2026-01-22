"""
Exchange service for managing exchange credentials and connections.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ExchangeCredential, CredentialAuditLog
from app.contracts.exchange import (
    ExchangeCredentials,
    ExchangeCredentialsCreate,
    ExchangeCredentialsUpdate,
    ExchangeBalance,
    SupportedExchange,
)
from app.adapters import get_exchange_adapter
from app.services.vault import vault_service
from app.core.exceptions import NotFoundError, ExchangeError, ErrorCode


class ExchangeService:
    """Service for exchange credential management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_credentials(
        self,
        user_id: UUID,
        credential_id: Optional[UUID] = None,
        exchange: Optional[SupportedExchange] = None,
    ) -> List[ExchangeCredential]:
        """
        Get exchange credentials for a user.

        Args:
            user_id: User ID
            credential_id: Optional specific credential ID
            exchange: Optional filter by exchange

        Returns:
            List of credentials
        """
        query = select(ExchangeCredential).where(
            ExchangeCredential.user_id == user_id
        )

        if credential_id:
            query = query.where(ExchangeCredential.id == credential_id)

        if exchange:
            query = query.where(ExchangeCredential.exchange == exchange.value)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_credential_by_id(
        self,
        credential_id: UUID,
        user_id: UUID,
    ) -> Optional[ExchangeCredential]:
        """Get a specific credential."""
        result = await self.db.execute(
            select(ExchangeCredential).where(
                and_(
                    ExchangeCredential.id == credential_id,
                    ExchangeCredential.user_id == user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def create_credential(
        self,
        user_id: UUID,
        data: ExchangeCredentialsCreate,
        ip_address: Optional[str] = None,
    ) -> ExchangeCredential:
        """
        Create new exchange credentials.

        Args:
            user_id: User ID
            data: Credential data
            ip_address: Client IP for audit

        Returns:
            Created credential
        """
        # Encrypt sensitive data
        encrypted_key, key_version = vault_service.encrypt(
            data.api_key.get_secret_value()
        )
        encrypted_secret, _ = vault_service.encrypt(
            data.api_secret.get_secret_value()
        )

        encrypted_passphrase = None
        if data.passphrase:
            encrypted_passphrase, _ = vault_service.encrypt(
                data.passphrase.get_secret_value()
            )

        # Create credential
        credential = ExchangeCredential(
            user_id=user_id,
            exchange=data.exchange.value,
            label=data.label,
            encrypted_api_key=encrypted_key,
            encrypted_api_secret=encrypted_secret,
            encrypted_passphrase=encrypted_passphrase,
            key_version=key_version,
            is_testnet=data.is_testnet,
            is_active=True,
            is_valid=True,
        )

        self.db.add(credential)
        await self.db.flush()

        # Create audit log
        audit = CredentialAuditLog(
            credential_id=credential.id,
            action="created",
            actor_id=user_id,
            ip_address=ip_address,
            details=f"Created credentials for {data.exchange.value}",
        )
        self.db.add(audit)

        await self.db.commit()
        await self.db.refresh(credential)

        return credential

    async def update_credential(
        self,
        user_id: UUID,
        credential_id: UUID,
        data: ExchangeCredentialsUpdate,
        ip_address: Optional[str] = None,
    ) -> ExchangeCredential:
        """
        Update exchange credentials.

        Args:
            user_id: User ID
            credential_id: Credential ID
            data: Update data
            ip_address: Client IP for audit

        Returns:
            Updated credential
        """
        credential = await self.get_credential_by_id(credential_id, user_id)

        if not credential:
            raise NotFoundError(
                message="Credential not found",
                resource_type="ExchangeCredential",
                resource_id=str(credential_id),
            )

        changes = []

        if data.label is not None:
            credential.label = data.label
            changes.append("label")

        if data.api_key is not None:
            encrypted_key, key_version = vault_service.encrypt(
                data.api_key.get_secret_value()
            )
            credential.encrypted_api_key = encrypted_key
            credential.key_version = key_version
            changes.append("api_key")

        if data.api_secret is not None:
            encrypted_secret, _ = vault_service.encrypt(
                data.api_secret.get_secret_value()
            )
            credential.encrypted_api_secret = encrypted_secret
            changes.append("api_secret")

        if data.passphrase is not None:
            encrypted_passphrase, _ = vault_service.encrypt(
                data.passphrase.get_secret_value()
            )
            credential.encrypted_passphrase = encrypted_passphrase
            changes.append("passphrase")

        if data.is_active is not None:
            credential.is_active = data.is_active
            changes.append("is_active")

        credential.updated_at = datetime.utcnow()

        # Create audit log
        audit = CredentialAuditLog(
            credential_id=credential.id,
            action="modified",
            actor_id=user_id,
            ip_address=ip_address,
            details=f"Updated: {', '.join(changes)}",
        )
        self.db.add(audit)

        await self.db.commit()
        await self.db.refresh(credential)

        return credential

    async def delete_credential(
        self,
        user_id: UUID,
        credential_id: UUID,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Delete exchange credentials.

        Args:
            user_id: User ID
            credential_id: Credential ID
            ip_address: Client IP for audit

        Returns:
            True if deleted
        """
        credential = await self.get_credential_by_id(credential_id, user_id)

        if not credential:
            raise NotFoundError(
                message="Credential not found",
                resource_type="ExchangeCredential",
                resource_id=str(credential_id),
            )

        # Create audit log before deletion
        audit = CredentialAuditLog(
            credential_id=credential.id,
            action="deleted",
            actor_id=user_id,
            ip_address=ip_address,
            details=f"Deleted credentials for {credential.exchange}",
        )
        self.db.add(audit)

        await self.db.delete(credential)
        await self.db.commit()

        return True

    async def validate_credential(
        self,
        user_id: UUID,
        credential_id: UUID,
    ) -> bool:
        """
        Validate exchange credentials by testing connection.

        Args:
            user_id: User ID
            credential_id: Credential ID

        Returns:
            True if valid
        """
        credential = await self.get_credential_by_id(credential_id, user_id)

        if not credential:
            raise NotFoundError(
                message="Credential not found",
                resource_type="ExchangeCredential",
                resource_id=str(credential_id),
            )

        # Decrypt credentials
        api_key = vault_service.decrypt(
            credential.encrypted_api_key,
            credential.key_version,
        )
        api_secret = vault_service.decrypt(
            credential.encrypted_api_secret,
            credential.key_version,
        )

        passphrase = None
        if credential.encrypted_passphrase:
            passphrase = vault_service.decrypt(
                credential.encrypted_passphrase,
                credential.key_version,
            )

        # Test connection
        adapter = get_exchange_adapter(
            exchange=SupportedExchange(credential.exchange),
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            testnet=credential.is_testnet,
        )

        try:
            is_valid = await adapter.validate_credentials()
            credential.is_valid = is_valid
            credential.last_validated = datetime.utcnow()
            await self.db.commit()
            return is_valid
        finally:
            await adapter.close()

    async def get_balances(
        self,
        user_id: UUID,
        credential_id: UUID,
    ) -> List[ExchangeBalance]:
        """
        Get balances for an exchange credential.

        Args:
            user_id: User ID
            credential_id: Credential ID

        Returns:
            List of balances
        """
        credential = await self.get_credential_by_id(credential_id, user_id)

        if not credential:
            raise NotFoundError(
                message="Credential not found",
                resource_type="ExchangeCredential",
                resource_id=str(credential_id),
            )

        # Decrypt credentials
        api_key = vault_service.decrypt(
            credential.encrypted_api_key,
            credential.key_version,
        )
        api_secret = vault_service.decrypt(
            credential.encrypted_api_secret,
            credential.key_version,
        )

        passphrase = None
        if credential.encrypted_passphrase:
            passphrase = vault_service.decrypt(
                credential.encrypted_passphrase,
                credential.key_version,
            )

        # Get balances
        adapter = get_exchange_adapter(
            exchange=SupportedExchange(credential.exchange),
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            testnet=credential.is_testnet,
        )

        try:
            balances = await adapter.get_balance()
            credential.last_used = datetime.utcnow()
            await self.db.commit()

            return [
                ExchangeBalance(
                    asset=b.asset,
                    free=b.free,
                    locked=b.locked,
                    total=b.total,
                )
                for b in balances
            ]
        finally:
            await adapter.close()

    def to_response(self, credential: ExchangeCredential) -> ExchangeCredentials:
        """Convert credential model to response."""
        # Decrypt and mask API key for display
        api_key = vault_service.decrypt(
            credential.encrypted_api_key,
            credential.key_version,
        )
        masked_key = vault_service.mask_key(api_key)

        return ExchangeCredentials(
            id=credential.id,
            user_id=credential.user_id,
            exchange=SupportedExchange(credential.exchange),
            label=credential.label,
            is_testnet=credential.is_testnet,
            is_active=credential.is_active,
            is_valid=credential.is_valid,
            created_at=credential.created_at,
            updated_at=credential.updated_at,
            last_used=credential.last_used,
            api_key_masked=masked_key,
        )
