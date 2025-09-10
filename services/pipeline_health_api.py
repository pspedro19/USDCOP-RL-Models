"""
Pipeline Health Monitoring API
==============================

FastAPI service providing REST endpoints for pipeline health monitoring dashboard.
Serves real-time metrics, historical data, and system health indicators.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime, timedelta
import yaml
from contextlib import asynccontextmanager

from pipeline_health_monitor import PipelineHealthMonitor, PipelineStageHealth, DataFlowMetrics, SystemHealthIndicators

# Configuration
class PipelineHealthConfig:
    def __init__(self, config_path: str = "config/pipeline_health_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

# Pydantic models for API responses
class PipelineStageHealthResponse(BaseModel):
    stage: str
    status: str
    last_run: Optional[datetime]
    processing_time: Optional[float]
    records_processed: int
    error_count: int
    error_rate: float
    data_completeness: float
    quality_score: float
    next_scheduled: Optional[datetime]
    dag_status: str

class DataFlowMetricsResponse(BaseModel):
    source_stage: str
    target_stage: str
    records_transferred: int
    transfer_time: float
    data_quality_delta: float
    last_transfer: datetime

class SystemHealthResponse(BaseModel):
    overall_status: str
    pipeline_availability: float
    average_processing_time: float
    total_error_rate: float
    data_freshness: float
    storage_usage: Dict[str, float]
    active_dags: int
    failed_dags: int

class AlertRule(BaseModel):
    id: str
    stage: str
    metric: str
    condition: str  # gt, lt, eq
    threshold: float
    severity: str  # ERROR, WARNING, INFO
    enabled: bool

class HistoricalDataRequest(BaseModel):
    stage: str
    hours: int = 24
    metric: Optional[str] = None

# Global monitoring instance
monitor: Optional[PipelineHealthMonitor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global monitor
    
    # Startup
    config = PipelineHealthConfig().config
    monitor = PipelineHealthMonitor(config)
    
    try:
        await monitor.initialize()
        
        # Start monitoring in background
        import asyncio
        monitoring_task = asyncio.create_task(monitor.start_monitoring())
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
        logging.info("Pipeline health monitoring API started")
        yield
        
    finally:
        # Shutdown
        monitoring_task.cancel()
        cleanup_task.cancel()
        await monitor.close()
        logging.info("Pipeline health monitoring API stopped")

# Initialize FastAPI app
app = FastAPI(
    title="Pipeline Health Monitoring API",
    description="Real-time monitoring and metrics for USDCOP trading pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get monitor instance
def get_monitor() -> PipelineHealthMonitor:
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitoring service not initialized")
    return monitor

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Pipeline Health Monitoring API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "pipeline-health-monitor"
    }

@app.get("/api/v1/pipeline/health", response_model=List[PipelineStageHealthResponse])
async def get_pipeline_health(monitor: PipelineHealthMonitor = Depends(get_monitor)):
    """Get current health status for all pipeline stages"""
    try:
        health_data = await monitor.get_pipeline_health()
        return [PipelineStageHealthResponse(**stage.__dict__) for stage in health_data]
    except Exception as e:
        logging.error(f"Error getting pipeline health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipeline/health/{stage}", response_model=PipelineStageHealthResponse)
async def get_stage_health(stage: str, monitor: PipelineHealthMonitor = Depends(get_monitor)):
    """Get health status for a specific pipeline stage"""
    try:
        # Validate stage name
        valid_stages = ['L0_ACQUIRE', 'L1_STANDARDIZE', 'L2_PREPARE', 
                       'L3_FEATURE', 'L4_RLREADY', 'L5_SERVING', 'L6_BACKTEST']
        
        if stage.upper() not in valid_stages:
            raise HTTPException(status_code=400, detail=f"Invalid stage: {stage}")
        
        health_data = await monitor.get_pipeline_health()
        stage_health = next((h for h in health_data if h.stage == stage.upper()), None)
        
        if not stage_health:
            raise HTTPException(status_code=404, detail=f"Stage {stage} not found")
        
        return PipelineStageHealthResponse(**stage_health.__dict__)
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting health for stage {stage}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipeline/dataflow", response_model=List[DataFlowMetricsResponse])
async def get_data_flow_metrics(monitor: PipelineHealthMonitor = Depends(get_monitor)):
    """Get data flow metrics between pipeline stages"""
    try:
        flow_data = await monitor.get_data_flow_metrics()
        return [DataFlowMetricsResponse(**flow.__dict__) for flow in flow_data]
    except Exception as e:
        logging.error(f"Error getting data flow metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/system/health", response_model=SystemHealthResponse)
async def get_system_health(monitor: PipelineHealthMonitor = Depends(get_monitor)):
    """Get overall system health indicators"""
    try:
        health_data = await monitor.get_system_health_indicators()
        return SystemHealthResponse(**health_data.__dict__)
    except Exception as e:
        logging.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipeline/historical/{stage}")
async def get_historical_data(
    stage: str, 
    hours: int = 24,
    monitor: PipelineHealthMonitor = Depends(get_monitor)
):
    """Get historical metrics for a pipeline stage"""
    try:
        # Validate stage name
        valid_stages = ['L0_ACQUIRE', 'L1_STANDARDIZE', 'L2_PREPARE', 
                       'L3_FEATURE', 'L4_RLREADY', 'L5_SERVING', 'L6_BACKTEST']
        
        if stage.upper() not in valid_stages:
            raise HTTPException(status_code=400, detail=f"Invalid stage: {stage}")
        
        if hours <= 0 or hours > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        
        historical_data = await monitor.get_historical_metrics(stage.upper(), hours)
        return {
            "stage": stage.upper(),
            "hours": hours,
            "data_points": len(historical_data),
            "data": historical_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting historical data for {stage}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics/summary")
async def get_metrics_summary(monitor: PipelineHealthMonitor = Depends(get_monitor)):
    """Get summarized metrics across all pipeline stages"""
    try:
        # Get current data
        pipeline_health = await monitor.get_pipeline_health()
        system_health = await monitor.get_system_health_indicators()
        data_flow = await monitor.get_data_flow_metrics()
        
        # Calculate summary statistics
        total_stages = len(pipeline_health)
        healthy_stages = sum(1 for stage in pipeline_health if stage.status == 'HEALTHY')
        warning_stages = sum(1 for stage in pipeline_health if stage.status == 'WARNING')
        error_stages = sum(1 for stage in pipeline_health if stage.status == 'ERROR')
        
        # Processing statistics
        total_records = sum(stage.records_processed for stage in pipeline_health)
        total_errors = sum(stage.error_count for stage in pipeline_health)
        
        # Quality statistics
        avg_quality_score = sum(stage.quality_score for stage in pipeline_health) / total_stages if total_stages > 0 else 0
        avg_completeness = sum(stage.data_completeness for stage in pipeline_health) / total_stages if total_stages > 0 else 0
        
        # Data flow statistics
        total_transfers = sum(flow.records_transferred for flow in data_flow)
        avg_transfer_time = sum(flow.transfer_time for flow in data_flow) / len(data_flow) if data_flow else 0
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": system_health.overall_status,
            "pipeline_summary": {
                "total_stages": total_stages,
                "healthy_stages": healthy_stages,
                "warning_stages": warning_stages,
                "error_stages": error_stages,
                "availability_percentage": system_health.pipeline_availability
            },
            "processing_summary": {
                "total_records_processed": total_records,
                "total_errors": total_errors,
                "overall_error_rate": system_health.total_error_rate,
                "average_processing_time": system_health.average_processing_time
            },
            "quality_summary": {
                "average_quality_score": round(avg_quality_score, 1),
                "average_completeness": round(avg_completeness * 100, 1),
                "data_freshness_hours": system_health.data_freshness
            },
            "data_flow_summary": {
                "total_records_transferred": total_transfers,
                "average_transfer_time": round(avg_transfer_time, 2),
                "active_flows": len(data_flow)
            },
            "infrastructure_summary": {
                "active_dags": system_health.active_dags,
                "failed_dags": system_health.failed_dags,
                "storage_usage_mb": system_health.storage_usage
            }
        }
    
    except Exception as e:
        logging.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/airflow/dags")
async def get_airflow_dag_status():
    """Get Airflow DAG status information"""
    try:
        # This would integrate with Airflow API
        # For now, return mock data structure
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_dags": 7,
            "active_dags": 6,
            "paused_dags": 1,
            "dags": [
                {
                    "dag_id": "usdcop_m5__l0_acquire",
                    "is_active": True,
                    "is_paused": False,
                    "last_run_state": "success",
                    "next_dagrun": "2025-01-09T10:00:00Z"
                },
                # Add other DAGs...
            ]
        }
    except Exception as e:
        logging.error(f"Error getting Airflow DAG status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/metrics/refresh")
async def refresh_metrics(
    background_tasks: BackgroundTasks,
    monitor: PipelineHealthMonitor = Depends(get_monitor)
):
    """Trigger manual refresh of metrics"""
    try:
        background_tasks.add_task(monitor.record_metrics)
        return {
            "message": "Metrics refresh triggered",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logging.error(f"Error triggering metrics refresh: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/alerts/rules")
async def get_alert_rules():
    """Get configured alert rules"""
    # This would typically come from a configuration file or database
    sample_rules = [
        {
            "id": "l0_data_freshness",
            "stage": "L0_ACQUIRE",
            "metric": "data_freshness", 
            "condition": "gt",
            "threshold": 2.0,  # hours
            "severity": "WARNING",
            "enabled": True
        },
        {
            "id": "pipeline_error_rate",
            "stage": "ALL",
            "metric": "error_rate",
            "condition": "gt", 
            "threshold": 0.05,  # 5%
            "severity": "ERROR",
            "enabled": True
        }
    ]
    
    return {
        "rules": sample_rules,
        "total_rules": len(sample_rules)
    }

@app.get("/api/v1/alerts/active")
async def get_active_alerts(monitor: PipelineHealthMonitor = Depends(get_monitor)):
    """Get currently active alerts"""
    try:
        # Get current metrics
        pipeline_health = await monitor.get_pipeline_health()
        system_health = await monitor.get_system_health_indicators()
        
        active_alerts = []
        
        # Check for various alert conditions
        for stage in pipeline_health:
            
            # Error status alert
            if stage.status == 'ERROR':
                active_alerts.append({
                    "id": f"error_{stage.stage}_{int(time.time())}",
                    "stage": stage.stage,
                    "severity": "ERROR",
                    "message": f"Pipeline stage {stage.stage} is in ERROR state",
                    "metric": "status",
                    "value": stage.status,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # High error rate alert
            if stage.error_rate > 0.05:  # 5%
                active_alerts.append({
                    "id": f"error_rate_{stage.stage}_{int(time.time())}",
                    "stage": stage.stage,
                    "severity": "WARNING",
                    "message": f"High error rate in {stage.stage}: {stage.error_rate:.2%}",
                    "metric": "error_rate",
                    "value": stage.error_rate,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Low data completeness alert
            if stage.data_completeness < 0.8:  # 80%
                active_alerts.append({
                    "id": f"completeness_{stage.stage}_{int(time.time())}",
                    "stage": stage.stage,
                    "severity": "WARNING", 
                    "message": f"Low data completeness in {stage.stage}: {stage.data_completeness:.1%}",
                    "metric": "data_completeness",
                    "value": stage.data_completeness,
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # System-level alerts
        if system_health.data_freshness > 2:  # 2 hours
            active_alerts.append({
                "id": f"data_freshness_{int(time.time())}",
                "stage": "SYSTEM",
                "severity": "WARNING",
                "message": f"Data is stale: {system_health.data_freshness:.1f} hours old",
                "metric": "data_freshness", 
                "value": system_health.data_freshness,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return {
            "alerts": active_alerts,
            "total_alerts": len(active_alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logging.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def periodic_cleanup():
    """Periodic cleanup of old metrics"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            if monitor:
                await monitor.cleanup_old_metrics()
        except Exception as e:
            logging.error(f"Error in periodic cleanup: {e}")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the API server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8002,
        log_level="info"
    )