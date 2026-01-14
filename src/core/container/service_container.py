"""
Service Container
=================

Thread-safe dependency injection container.

Supports:
- Singleton registration
- Factory registration
- Lazy initialization
- Scoped lifetimes

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Type
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Lifetime(Enum):
    """Service lifetime options."""
    SINGLETON = "singleton"  # One instance for entire application
    TRANSIENT = "transient"  # New instance each resolution
    SCOPED = "scoped"        # One instance per scope


class ServiceContainer:
    """
    Thread-safe dependency injection container.

    Singleton Pattern: Single global instance.
    Registry Pattern: Dynamic service registration.

    Usage:
        # Register services
        container = ServiceContainer.get_instance()
        container.register_singleton("redis", redis_client)
        container.register_factory("inference_engine", lambda c: InferenceEngine(...))

        # Resolve services
        redis = container.resolve("redis")
        engine = container.resolve("inference_engine")

        # Type-safe resolution
        engine = container.resolve_typed("inference_engine", InferenceEngine)
    """

    _instance: Optional['ServiceContainer'] = None
    _lock = Lock()

    def __init__(self):
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[['ServiceContainer'], Any]] = {}
        self._lifetimes: Dict[str, Lifetime] = {}
        self._scopes: Dict[str, Dict[str, Any]] = {}
        self._current_scope: Optional[str] = None
        self._container_lock = Lock()

    @classmethod
    def get_instance(cls) -> 'ServiceContainer':
        """
        Get singleton container instance.

        Thread-safe lazy initialization.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("ServiceContainer initialized")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
            logger.info("ServiceContainer reset")

    def register_singleton(self, name: str, instance: Any) -> 'ServiceContainer':
        """
        Register a singleton instance.

        Args:
            name: Service name
            instance: Pre-created instance

        Returns:
            Self for chaining
        """
        with self._container_lock:
            self._singletons[name] = instance
            self._lifetimes[name] = Lifetime.SINGLETON
            logger.debug(f"Registered singleton: {name}")
        return self

    def register_factory(
        self,
        name: str,
        factory: Callable[['ServiceContainer'], Any],
        lifetime: Lifetime = Lifetime.SINGLETON
    ) -> 'ServiceContainer':
        """
        Register a factory function.

        Args:
            name: Service name
            factory: Function that creates the service (receives container)
            lifetime: Service lifetime

        Returns:
            Self for chaining
        """
        with self._container_lock:
            self._factories[name] = factory
            self._lifetimes[name] = lifetime
            logger.debug(f"Registered factory: {name} (lifetime={lifetime.value})")
        return self

    def register_type(
        self,
        name: str,
        service_type: Type[T],
        lifetime: Lifetime = Lifetime.SINGLETON,
        **kwargs
    ) -> 'ServiceContainer':
        """
        Register a type for automatic instantiation.

        Args:
            name: Service name
            service_type: Type to instantiate
            lifetime: Service lifetime
            **kwargs: Constructor arguments

        Returns:
            Self for chaining
        """
        def factory(container: ServiceContainer) -> T:
            return service_type(**kwargs)

        return self.register_factory(name, factory, lifetime)

    def resolve(self, name: str) -> Any:
        """
        Resolve a service by name.

        Args:
            name: Service name

        Returns:
            Service instance

        Raises:
            KeyError: If service not registered
        """
        with self._container_lock:
            # Check for existing singleton
            if name in self._singletons:
                return self._singletons[name]

            # Check for scoped instance
            if self._current_scope and name in self._scopes.get(self._current_scope, {}):
                return self._scopes[self._current_scope][name]

            # Check for factory
            if name not in self._factories:
                raise KeyError(f"Service not registered: '{name}'")

            # Create instance
            factory = self._factories[name]
            instance = factory(self)
            lifetime = self._lifetimes.get(name, Lifetime.SINGLETON)

            # Store based on lifetime
            if lifetime == Lifetime.SINGLETON:
                self._singletons[name] = instance
            elif lifetime == Lifetime.SCOPED and self._current_scope:
                if self._current_scope not in self._scopes:
                    self._scopes[self._current_scope] = {}
                self._scopes[self._current_scope][name] = instance

            logger.debug(f"Resolved service: {name}")
            return instance

    def resolve_typed(self, name: str, expected_type: Type[T]) -> T:
        """
        Resolve with type checking.

        Args:
            name: Service name
            expected_type: Expected type

        Returns:
            Service instance of expected type

        Raises:
            TypeError: If instance is not of expected type
        """
        instance = self.resolve(name)

        if not isinstance(instance, expected_type):
            raise TypeError(
                f"Service '{name}' is {type(instance).__name__}, "
                f"expected {expected_type.__name__}"
            )

        return instance

    def try_resolve(self, name: str) -> Optional[Any]:
        """
        Try to resolve, return None if not found.

        Args:
            name: Service name

        Returns:
            Service instance or None
        """
        try:
            return self.resolve(name)
        except KeyError:
            return None

    def is_registered(self, name: str) -> bool:
        """Check if service is registered."""
        return name in self._singletons or name in self._factories

    def get_registered_services(self) -> list:
        """Get list of registered service names."""
        names = set(self._singletons.keys()) | set(self._factories.keys())
        return sorted(names)

    # =========================================================================
    # Scope management
    # =========================================================================

    def begin_scope(self, scope_name: str) -> 'ServiceContainer':
        """
        Begin a new scope.

        Args:
            scope_name: Unique scope identifier

        Returns:
            Self for chaining
        """
        with self._container_lock:
            self._current_scope = scope_name
            self._scopes[scope_name] = {}
            logger.debug(f"Began scope: {scope_name}")
        return self

    def end_scope(self) -> 'ServiceContainer':
        """
        End current scope and dispose scoped instances.

        Returns:
            Self for chaining
        """
        with self._container_lock:
            if self._current_scope:
                # Dispose scoped instances
                scope_instances = self._scopes.pop(self._current_scope, {})
                for name, instance in scope_instances.items():
                    if hasattr(instance, 'dispose'):
                        try:
                            instance.dispose()
                        except Exception as e:
                            logger.error(f"Error disposing {name}: {e}")

                logger.debug(f"Ended scope: {self._current_scope}")
                self._current_scope = None
        return self

    # =========================================================================
    # Bulk operations
    # =========================================================================

    def clear(self) -> None:
        """Clear all registrations (for testing)."""
        with self._container_lock:
            self._singletons.clear()
            self._factories.clear()
            self._lifetimes.clear()
            self._scopes.clear()
            self._current_scope = None
            logger.info("ServiceContainer cleared")

    def health_check(self) -> Dict[str, Any]:
        """Check health of registered services."""
        health = {
            "status": "healthy",
            "singleton_count": len(self._singletons),
            "factory_count": len(self._factories),
            "services": {}
        }

        for name in self.get_registered_services():
            try:
                instance = self.resolve(name)
                if hasattr(instance, 'health_check'):
                    service_health = instance.health_check()
                    health["services"][name] = service_health
                    if service_health.get("status") != "healthy":
                        health["status"] = "degraded"
                else:
                    health["services"][name] = {"status": "healthy"}
            except Exception as e:
                health["services"][name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"

        return health
