"""
Service Discovery & Registry Implementation
Integrates with Consul for dynamic service management
"""

import os
import atexit
import socket
import time
import requests
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Environment configuration
CONSUL_ADDR = os.getenv("CONSUL_HTTP_ADDR", "http://consul:8500")
CONSUL_TOKEN = os.getenv("CONSUL_HTTP_TOKEN", None)
SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown")
APP_ENV = os.getenv("APP_ENV", "dev")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

@dataclass
class ServiceMeta:
    """Service metadata for Consul registration"""
    version: str
    capabilities: List[str]
    load: float = 0.0
    env: str = APP_ENV
    region: str = os.getenv("REGION", "local")
    instance_id: str = os.getenv("INSTANCE_ID", "1")

class ServiceRegistry:
    """Consul-based service registry for USDCOP trading system"""
    
    def __init__(self, name: str, port: int, meta: ServiceMeta,
                 service_id: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 health_path: str = "/health/overview",
                 interval: str = "10s",
                 timeout: str = "30s"):
        self.name = name
        self.port = port
        self.meta = meta
        self.tags = tags or []
        self.health_path = health_path
        self.interval = interval
        self.timeout = timeout
        self.addr = self._get_host_ip()
        self.service_id = service_id or f"{name}-{self.addr}-{port}"
        self._headers = {"X-Consul-Token": CONSUL_TOKEN} if CONSUL_TOKEN else {}
        self._registered = False
        
        # Auto-deregister on exit
        atexit.register(self.deregister)
        
        logger.info(f"ServiceRegistry initialized: {name} on {self.addr}:{port}")
    
    def _get_host_ip(self) -> str:
        """Get container IP address"""
        try:
            # Try to get container IP from environment
            if os.getenv("HOSTNAME"):
                return socket.gethostbyname(os.getenv("HOSTNAME"))
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
    
    def register(self) -> bool:
        """Register service with Consul"""
        try:
            payload = {
                "Name": self.name,
                "ID": self.service_id,
                "Address": self.addr,
                "Port": self.port,
                "Tags": self.tags,
                "Meta": asdict(self.meta),
                "Check": {
                    "HTTP": f"http://{self.addr}:{self.port}{self.health_path}",
                    "Method": "GET",
                    "Interval": self.interval,
                    "Timeout": self.timeout,
                    "DeregisterCriticalServiceAfter": "2m"
                }
            }
            
            response = requests.put(
                f"{CONSUL_ADDR}/v1/agent/service/register",
                json=payload,
                headers=self._headers,
                timeout=5
            )
            response.raise_for_status()
            
            self._registered = True
            logger.info(f"Service registered successfully: {self.name} ({self.service_id})")
            return True
            
        except Exception as e:
            logger.error(f"Service registration failed: {e}")
            return False
    
    def deregister(self) -> bool:
        """Deregister service from Consul"""
        if not self._registered:
            return False
            
        try:
            response = requests.put(
                f"{CONSUL_ADDR}/v1/agent/service/deregister/{self.service_id}",
                headers=self._headers,
                timeout=5
            )
            response.raise_for_status()
            
            logger.info(f"Service deregistered successfully: {self.service_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Service deregistration failed: {e}")
            return False
        finally:
            self._registered = False
    
    @staticmethod
    def discover(name: str) -> List[Dict[str, Any]]:
        """Discover healthy instances of a service"""
        try:
            headers = {"X-Consul-Token": CONSUL_TOKEN} if CONSUL_TOKEN else {}
            response = requests.get(
                f"{CONSUL_ADDR}/v1/health/service/{name}?passing=true",
                headers=headers,
                timeout=3
            )
            response.raise_for_status()
            
            services = response.json()
            return [
                {
                    "id": svc["Service"]["ID"],
                    "name": svc["Service"]["Service"],
                    "address": svc["Service"]["Address"],
                    "port": svc["Service"]["Port"],
                    "tags": svc["Service"].get("Tags", []),
                    "meta": svc["Service"].get("Meta", {}),
                    "health": svc["Checks"][0]["Status"]
                }
                for svc in services
            ]
            
        except Exception as e:
            logger.error(f"Service discovery failed for {name}: {e}")
            return []
    
    @staticmethod
    def pick_healthy_endpoint(name: str) -> Optional[str]:
        """Get a healthy endpoint for load balancing"""
        services = ServiceRegistry.discover(name)
        if not services:
            return None
        
        # Simple round-robin based on time
        selected = services[int(time.time()) % len(services)]
        return f"http://{selected['address']}:{selected['port']}"
    
    @staticmethod
    def get_service_health(name: str) -> Dict[str, Any]:
        """Get health status of a service"""
        try:
            headers = {"X-Consul-Token": CONSUL_TOKEN} if CONSUL_TOKEN else {}
            response = requests.get(
                f"{CONSUL_ADDR}/v1/health/service/{name}",
                headers=headers,
                timeout=3
            )
            response.raise_for_status()
            
            checks = response.json()
            healthy_count = sum(1 for check in checks if check["Status"] == "passing")
            total_count = len(checks)
            
            return {
                "service": name,
                "healthy_instances": healthy_count,
                "total_instances": total_count,
                "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 0,
                "instances": [
                    {
                        "id": check["ServiceID"],
                        "status": check["Status"],
                        "output": check.get("Output", ""),
                        "last_update": check.get("LastUpdate", "")
                    }
                    for check in checks
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get health for {name}: {e}")
            return {}
