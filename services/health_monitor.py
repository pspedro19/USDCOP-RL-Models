"""
Health Monitor Service
Monitors all services and provides consolidated health status
"""

import asyncio
import aiohttp
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthMonitor:
    """Monitors health of all services"""
    
    def __init__(self):
        self.services = {
            'postgres': 'http://postgres:5432',  # Will use pg_isready
            'redis': 'http://redis:6379',        # Will use redis-cli ping
            'realtime-data': 'http://realtime-data-service:8080/health',
            'websocket': 'http://websocket-service:8080/health',
            'dashboard': 'http://dashboard:3000/api/health',
            'airflow-webserver': 'http://airflow-webserver:8080/health',
            'grafana': 'http://grafana:3000/api/health'
        }
        self.last_check = {}
        self.health_status = {}
        
    async def check_service_health(self, service_name: str, endpoint: str) -> Dict[str, Any]:
        """Check health of a single service"""
        try:
            if service_name in ['postgres', 'redis']:
                # Special handling for infrastructure services
                return await self.check_infrastructure_service(service_name)
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = datetime.now()
                async with session.get(endpoint) as response:
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'service': service_name,
                            'status': 'healthy',
                            'response_time_ms': round(response_time, 2),
                            'details': data,
                            'checked_at': datetime.now().isoformat()
                        }
                    else:
                        return {
                            'service': service_name,
                            'status': 'unhealthy',
                            'response_time_ms': round(response_time, 2),
                            'error': f"HTTP {response.status}",
                            'checked_at': datetime.now().isoformat()
                        }
                        
        except Exception as e:
            return {
                'service': service_name,
                'status': 'unhealthy',
                'error': str(e),
                'checked_at': datetime.now().isoformat()
            }
    
    async def check_infrastructure_service(self, service_name: str) -> Dict[str, Any]:
        """Check infrastructure services (postgres, redis)"""
        try:
            if service_name == 'postgres':
                # Check postgres using docker exec
                process = await asyncio.create_subprocess_exec(
                    'docker', 'exec', 'usdcop-postgres-timescale', 'pg_isready', '-U', 'admin',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    return {
                        'service': service_name,
                        'status': 'healthy',
                        'checked_at': datetime.now().isoformat()
                    }
                else:
                    return {
                        'service': service_name,
                        'status': 'unhealthy',
                        'error': stderr.decode() if stderr else 'Connection failed',
                        'checked_at': datetime.now().isoformat()
                    }
                    
            elif service_name == 'redis':
                # Check redis using docker exec
                process = await asyncio.create_subprocess_exec(
                    'docker', 'exec', 'usdcop-redis', 'redis-cli', 'ping',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0 and b'PONG' in stdout:
                    return {
                        'service': service_name,
                        'status': 'healthy',
                        'checked_at': datetime.now().isoformat()
                    }
                else:
                    return {
                        'service': service_name,
                        'status': 'unhealthy',
                        'error': stderr.decode() if stderr else 'Ping failed',
                        'checked_at': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            return {
                'service': service_name,
                'status': 'unhealthy',
                'error': str(e),
                'checked_at': datetime.now().isoformat()
            }
    
    async def check_all_services(self) -> Dict[str, Any]:
        """Check health of all services"""
        tasks = []
        for service_name, endpoint in self.services.items():
            task = self.check_service_health(service_name, endpoint)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        healthy_services = []
        unhealthy_services = []
        service_details = {}
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error checking service: {result}")
                continue
                
            service_name = result['service']
            service_details[service_name] = result
            
            if result['status'] == 'healthy':
                healthy_services.append(service_name)
            else:
                unhealthy_services.append(service_name)
        
        overall_status = 'healthy' if len(unhealthy_services) == 0 else 'degraded' if len(healthy_services) > len(unhealthy_services) else 'unhealthy'
        
        return {
            'overall_status': overall_status,
            'total_services': len(self.services),
            'healthy_count': len(healthy_services),
            'unhealthy_count': len(unhealthy_services),
            'healthy_services': healthy_services,
            'unhealthy_services': unhealthy_services,
            'service_details': service_details,
            'checked_at': datetime.now().isoformat()
        }

# Global health monitor
health_monitor = HealthMonitor()

# FastAPI application
app = FastAPI(
    title="USDCOP Health Monitor",
    description="Health monitoring service for all USDCOP trading system components",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check for the monitor itself"""
    return JSONResponse({
        "status": "healthy",
        "service": "health-monitor",
        "timestamp": datetime.now().isoformat()
    })

@app.get("/health/all")
async def check_all_services():
    """Check health of all services"""
    try:
        health_status = await health_monitor.check_all_services()
        return JSONResponse(health_status)
    except Exception as e:
        logger.error(f"Error checking all services: {e}")
        return JSONResponse({
            "error": "Failed to check service health",
            "details": str(e)
        }, status_code=500)

@app.get("/health/service/{service_name}")
async def check_specific_service(service_name: str):
    """Check health of a specific service"""
    if service_name not in health_monitor.services:
        return JSONResponse({
            "error": f"Service '{service_name}' not found",
            "available_services": list(health_monitor.services.keys())
        }, status_code=404)
    
    try:
        endpoint = health_monitor.services[service_name]
        health_status = await health_monitor.check_service_health(service_name, endpoint)
        return JSONResponse(health_status)
    except Exception as e:
        logger.error(f"Error checking service {service_name}: {e}")
        return JSONResponse({
            "error": f"Failed to check {service_name} health",
            "details": str(e)
        }, status_code=500)

@app.get("/health/summary")
async def health_summary():
    """Get a quick health summary"""
    try:
        health_status = await health_monitor.check_all_services()
        return JSONResponse({
            "overall_status": health_status["overall_status"],
            "healthy_count": health_status["healthy_count"],
            "total_services": health_status["total_services"],
            "unhealthy_services": health_status["unhealthy_services"],
            "checked_at": health_status["checked_at"]
        })
    except Exception as e:
        logger.error(f"Error getting health summary: {e}")
        return JSONResponse({
            "error": "Failed to get health summary",
            "details": str(e)
        }, status_code=500)

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )