"""
Service Discovery Module
=======================
Provides service registration and discovery capabilities using Consul.
"""

from .registry import ServiceRegistry, ServiceMeta

__all__ = ['ServiceRegistry', 'ServiceMeta']
