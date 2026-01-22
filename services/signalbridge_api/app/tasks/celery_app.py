"""
Celery application configuration.
"""

from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "signalbridge",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.tasks.signal_tasks",
        "app.tasks.execution_tasks",
    ],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=300,  # 5 minutes max per task
    task_soft_time_limit=240,  # 4 minutes soft limit

    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=4,

    # Result backend
    result_expires=3600,  # 1 hour

    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,

    # Beat schedule for periodic tasks
    beat_schedule={
        "process-pending-signals": {
            "task": "app.tasks.signal_tasks.process_pending_signals",
            "schedule": 30.0,  # Every 30 seconds
        },
        "retry-failed-executions": {
            "task": "app.tasks.execution_tasks.retry_failed_executions",
            "schedule": 300.0,  # Every 5 minutes
        },
        "cleanup-old-data": {
            "task": "app.tasks.maintenance_tasks.cleanup_old_data",
            "schedule": crontab(hour=0, minute=0),  # Daily at midnight
        },
    },
)
