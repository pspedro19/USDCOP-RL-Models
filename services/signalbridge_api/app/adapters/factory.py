"""
Exchange Adapter Factory.
Implements Factory pattern for creating exchange adapters.
"""


from app.contracts.exchange import SupportedExchange
from app.core.exceptions import ErrorCode, ValidationError

from .base import ExchangeAdapter
from .binance import BinanceAdapter
from .mexc import MEXCAdapter


class ExchangeAdapterFactory:
    """
    Factory for creating exchange adapters.
    Implements the Factory pattern for easy adapter instantiation.
    """

    # Registry of exchange adapters
    _adapters: dict[SupportedExchange, type[ExchangeAdapter]] = {
        SupportedExchange.MEXC: MEXCAdapter,
        SupportedExchange.BINANCE: BinanceAdapter,
    }

    @classmethod
    def create(
        cls,
        exchange: SupportedExchange,
        api_key: str,
        api_secret: str,
        passphrase: str | None = None,
        testnet: bool = False,
    ) -> ExchangeAdapter:
        """
        Create an exchange adapter instance.

        Args:
            exchange: The exchange to create adapter for
            api_key: Exchange API key
            api_secret: Exchange API secret
            passphrase: Optional passphrase
            testnet: Whether to use testnet

        Returns:
            Configured exchange adapter

        Raises:
            ValidationError: If exchange is not supported
        """
        adapter_class = cls._adapters.get(exchange)

        if not adapter_class:
            raise ValidationError(
                message=f"Exchange '{exchange}' is not supported",
                error_code=ErrorCode.INVALID_EXCHANGE,
                details={"supported_exchanges": list(cls.get_supported_exchanges())},
            )

        return adapter_class(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            testnet=testnet,
        )

    @classmethod
    def register(
        cls,
        exchange: SupportedExchange,
        adapter_class: type[ExchangeAdapter],
    ) -> None:
        """
        Register a new exchange adapter.

        Args:
            exchange: Exchange identifier
            adapter_class: Adapter class to register
        """
        cls._adapters[exchange] = adapter_class

    @classmethod
    def get_supported_exchanges(cls) -> list[SupportedExchange]:
        """Get list of supported exchanges."""
        return list(cls._adapters.keys())

    @classmethod
    def is_supported(cls, exchange: SupportedExchange) -> bool:
        """Check if an exchange is supported."""
        return exchange in cls._adapters


def get_exchange_adapter(
    exchange: SupportedExchange,
    api_key: str,
    api_secret: str,
    passphrase: str | None = None,
    testnet: bool = False,
) -> ExchangeAdapter:
    """
    Convenience function for creating exchange adapters.

    Args:
        exchange: The exchange to create adapter for
        api_key: Exchange API key
        api_secret: Exchange API secret
        passphrase: Optional passphrase
        testnet: Whether to use testnet

    Returns:
        Configured exchange adapter
    """
    return ExchangeAdapterFactory.create(
        exchange=exchange,
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        testnet=testnet,
    )
