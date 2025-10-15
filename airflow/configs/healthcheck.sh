#!/bin/bash
# Simple healthcheck script for Airflow Scheduler

echo "CUSTOM HEALTHCHECK SCRIPT RUNNING"

# Check if any process has "airflow scheduler" in its command line
if find /proc -maxdepth 2 -name cmdline -exec grep -l "airflow.*scheduler" {} \; 2>/dev/null | head -1 | grep -q .; then
    echo "Scheduler process found - healthy"
    exit 0
else
    echo "Scheduler process not found - unhealthy"
    exit 1
fi