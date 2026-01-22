"""
Vault service for secure API key encryption/decryption.
Implements AES-256-GCM as specified in the spec.
"""

import base64
import os
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from app.core.config import settings
from app.core.exceptions import VaultError, ErrorCode


class VaultService:
    """
    Secure vault for encrypting/decrypting sensitive data.
    Uses AES-256-GCM encryption as required by spec.
    """

    # Nonce size for GCM (96 bits = 12 bytes recommended by NIST)
    NONCE_SIZE = 12
    # Key size for AES-256 (256 bits = 32 bytes)
    KEY_SIZE = 32
    # Salt size for key derivation
    SALT_SIZE = 16
    # Current key version
    KEY_VERSION = "v1"

    def __init__(self, master_key: str = None):
        """
        Initialize vault with master encryption key.

        Args:
            master_key: Base64-encoded master key or passphrase
        """
        self._master_key = master_key or settings.vault_encryption_key
        self._derived_keys: dict[str, bytes] = {}

    def _derive_key(self, salt: bytes) -> bytes:
        """
        Derive encryption key from master key using PBKDF2.

        Args:
            salt: Random salt for key derivation

        Returns:
            32-byte derived key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.KEY_SIZE,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(self._master_key.encode())

    def encrypt(self, plaintext: str) -> Tuple[str, str]:
        """
        Encrypt plaintext using AES-256-GCM.

        Args:
            plaintext: The secret text to encrypt

        Returns:
            Tuple of (encrypted_data_base64, key_version)

        Raises:
            VaultError: If encryption fails
        """
        try:
            # Generate random salt and nonce
            salt = os.urandom(self.SALT_SIZE)
            nonce = os.urandom(self.NONCE_SIZE)

            # Derive key from master key
            key = self._derive_key(salt)

            # Encrypt with AES-256-GCM
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)

            # Combine salt + nonce + ciphertext and encode as base64
            encrypted_data = salt + nonce + ciphertext
            encrypted_base64 = base64.b64encode(encrypted_data).decode()

            return encrypted_base64, self.KEY_VERSION

        except Exception as e:
            raise VaultError(
                message=f"Encryption failed: {str(e)}",
                error_code=ErrorCode.VAULT_ENCRYPTION_FAILED,
            )

    def decrypt(self, encrypted_base64: str, key_version: str = None) -> str:
        """
        Decrypt ciphertext using AES-256-GCM.

        Args:
            encrypted_base64: Base64-encoded encrypted data
            key_version: Version of the key used for encryption

        Returns:
            Decrypted plaintext

        Raises:
            VaultError: If decryption fails
        """
        try:
            # Decode base64
            encrypted_data = base64.b64decode(encrypted_base64)

            # Extract salt, nonce, and ciphertext
            salt = encrypted_data[: self.SALT_SIZE]
            nonce = encrypted_data[self.SALT_SIZE : self.SALT_SIZE + self.NONCE_SIZE]
            ciphertext = encrypted_data[self.SALT_SIZE + self.NONCE_SIZE :]

            # Derive key from master key using same salt
            key = self._derive_key(salt)

            # Decrypt with AES-256-GCM
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)

            return plaintext.decode()

        except Exception as e:
            raise VaultError(
                message=f"Decryption failed: {str(e)}",
                error_code=ErrorCode.VAULT_DECRYPTION_FAILED,
            )

    def mask_key(self, key: str, visible_chars: int = 4) -> str:
        """
        Mask an API key for display.

        Args:
            key: The full API key
            visible_chars: Number of characters to show at start and end

        Returns:
            Masked key (e.g., "abc...xyz")
        """
        if len(key) <= visible_chars * 2:
            return "*" * len(key)

        return f"{key[:visible_chars]}...{key[-visible_chars:]}"

    def rotate_encryption(
        self,
        encrypted_base64: str,
        old_key_version: str,
    ) -> Tuple[str, str]:
        """
        Re-encrypt data with current key version.
        Used for key rotation.

        Args:
            encrypted_base64: Data encrypted with old key
            old_key_version: Version of the old key

        Returns:
            Tuple of (new_encrypted_data_base64, new_key_version)
        """
        # Decrypt with old key (assuming same master key for now)
        plaintext = self.decrypt(encrypted_base64, old_key_version)

        # Re-encrypt with current key
        return self.encrypt(plaintext)

    async def get_exchange_credentials(
        self,
        credential_id,
        db_session=None,
    ) -> dict:
        """
        Get exchange credentials from database and decrypt.

        Args:
            credential_id: UUID of the credential record
            db_session: Database session (optional, will create if not provided)

        Returns:
            Dict with decrypted credentials:
            {
                "exchange": SupportedExchange,
                "api_key": str,
                "api_secret": str,
                "passphrase": str | None,
                "testnet": bool
            }
        """
        from uuid import UUID
        from sqlalchemy import text
        from app.contracts.exchange import SupportedExchange

        if db_session is None:
            from app.core.database import get_db_session
            async for session in get_db_session():
                db_session = session
                break

        # Query credentials from database
        result = await db_session.execute(
            text("""
                SELECT
                    exchange,
                    encrypted_api_key,
                    encrypted_api_secret,
                    encrypted_passphrase,
                    key_version,
                    is_testnet
                FROM exchange_credentials
                WHERE id = :credential_id AND is_active = true
            """),
            {"credential_id": str(credential_id)}
        )
        row = result.first()

        if not row:
            raise VaultError(
                message=f"Credentials not found: {credential_id}",
                error_code=ErrorCode.VAULT_DECRYPTION_FAILED,
            )

        # Decrypt sensitive fields
        api_key = self.decrypt(row.encrypted_api_key, row.key_version)
        api_secret = self.decrypt(row.encrypted_api_secret, row.key_version)
        passphrase = None
        if row.encrypted_passphrase:
            passphrase = self.decrypt(row.encrypted_passphrase, row.key_version)

        return {
            "exchange": SupportedExchange(row.exchange),
            "api_key": api_key,
            "api_secret": api_secret,
            "passphrase": passphrase,
            "testnet": row.is_testnet,
        }

    async def store_exchange_credentials(
        self,
        user_id,
        exchange: str,
        api_key: str,
        api_secret: str,
        passphrase: str = None,
        label: str = "default",
        is_testnet: bool = False,
        db_session=None,
    ) -> str:
        """
        Encrypt and store exchange credentials.

        Args:
            user_id: User ID
            exchange: Exchange name
            api_key: Plain API key
            api_secret: Plain API secret
            passphrase: Optional passphrase
            label: Credential label
            is_testnet: Whether testnet credentials
            db_session: Database session

        Returns:
            credential_id: UUID of created credential
        """
        from uuid import uuid4
        from sqlalchemy import text

        if db_session is None:
            from app.core.database import get_db_session
            async for session in get_db_session():
                db_session = session
                break

        # Encrypt sensitive data
        encrypted_key, key_version = self.encrypt(api_key)
        encrypted_secret, _ = self.encrypt(api_secret)
        encrypted_passphrase = None
        if passphrase:
            encrypted_passphrase, _ = self.encrypt(passphrase)

        credential_id = uuid4()

        await db_session.execute(
            text("""
                INSERT INTO exchange_credentials (
                    id, user_id, exchange, label,
                    encrypted_api_key, encrypted_api_secret, encrypted_passphrase,
                    key_version, is_testnet, is_active, created_at
                ) VALUES (
                    :id, :user_id, :exchange, :label,
                    :encrypted_api_key, :encrypted_api_secret, :encrypted_passphrase,
                    :key_version, :is_testnet, true, NOW()
                )
            """),
            {
                "id": str(credential_id),
                "user_id": str(user_id),
                "exchange": exchange,
                "label": label,
                "encrypted_api_key": encrypted_key,
                "encrypted_api_secret": encrypted_secret,
                "encrypted_passphrase": encrypted_passphrase,
                "key_version": key_version,
                "is_testnet": is_testnet,
            }
        )
        await db_session.commit()

        return str(credential_id)

    async def health_check(self) -> bool:
        """Check if vault is operational."""
        try:
            test_data = "health_check_test"
            encrypted, version = self.encrypt(test_data)
            decrypted = self.decrypt(encrypted, version)
            return decrypted == test_data
        except Exception:
            return False


# Singleton instance
vault_service = VaultService()
