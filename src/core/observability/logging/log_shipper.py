"""
Log Shipper
===========
Sends logs to centralized storage (Loki, ELK, etc.).
"""
import json
import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime

class LogShipper:
    """Base class for shipping logs to centralized storage."""
    
    def __init__(self, endpoint: str, service_name: str):
        self.endpoint = endpoint
        self.service_name = service_name
        self.logger = logging.getLogger(__name__)
    
    def ship_log(self, log_entry: Dict[str, Any]) -> bool:
        """Ship a log entry to the centralized storage."""
        raise NotImplementedError("Subclasses must implement ship_log")

class LokiLogShipper(LogShipper):
    """Log shipper for Grafana Loki."""
    
    def __init__(self, loki_url: str, service_name: str, labels: Optional[Dict[str, str]] = None):
        super().__init__(loki_url, service_name)
        self.loki_url = loki_url.rstrip('/') + '/loki/api/v1/push'
        self.labels = labels or {}
        self.session = requests.Session()
    
    def ship_log(self, log_entry: Dict[str, Any]) -> bool:
        """Ship log to Loki using the push API."""
        try:
            # Prepare Loki payload
            timestamp_ns = int(datetime.utcnow().timestamp() * 1e9)
            
            # Extract labels from log entry
            labels = {
                "service": self.service_name,
                "level": log_entry.get("level", "INFO"),
                "module": log_entry.get("module", "unknown"),
                **self.labels
            }
            
            # Add trace context as labels if available
            if log_entry.get("trace_id"):
                labels["trace_id"] = log_entry["trace_id"]
            if log_entry.get("correlation_id"):
                labels["correlation_id"] = log_entry["correlation_id"]
            
            # Create Loki stream
            stream = {
                "stream": labels,
                "values": [
                    [str(timestamp_ns), json.dumps(log_entry)]
                ]
            }
            
            payload = {"streams": [stream]}
            
            # Send to Loki
            response = self.session.post(
                self.loki_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 204:
                return True
            else:
                self.logger.warning(f"Failed to ship log to Loki: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error shipping log to Loki: {e}")
            return False
    
    def close(self):
        """Close the session."""
        self.session.close()

class ElasticsearchLogShipper(LogShipper):
    """Log shipper for Elasticsearch."""
    
    def __init__(self, es_url: str, service_name: str, index_prefix: str = "logs"):
        super().__init__(es_url, service_name)
        self.es_url = es_url.rstrip('/')
        self.index_prefix = index_prefix
        self.session = requests.Session()
    
    def ship_log(self, log_entry: Dict[str, Any]) -> bool:
        """Ship log to Elasticsearch."""
        try:
            # Create index name with date
            date_str = datetime.utcnow().strftime("%Y.%m.%d")
            index_name = f"{self.index_prefix}-{self.service_name}-{date_str}"
            
            # Prepare document
            document = {
                "@timestamp": log_entry.get("timestamp", datetime.utcnow().isoformat()),
                "service": self.service_name,
                "level": log_entry.get("level", "INFO"),
                "message": log_entry.get("message", ""),
                "logger": log_entry.get("logger", ""),
                "module": log_entry.get("module", ""),
                "function": log_entry.get("function", ""),
                "line": log_entry.get("line", 0),
                "trace_id": log_entry.get("trace_id"),
                "span_id": log_entry.get("span_id"),
                "correlation_id": log_entry.get("correlation_id"),
                "request_id": log_entry.get("request_id"),
                "exception": log_entry.get("exception"),
                **{k: v for k, v in log_entry.items() 
                   if k not in ["timestamp", "service", "level", "message", "logger", 
                               "module", "function", "line", "trace_id", "span_id", 
                               "correlation_id", "request_id", "exception"]}
            }
            
            # Send to Elasticsearch
            url = f"{self.es_url}/{index_name}/_doc"
            response = self.session.post(
                url,
                json=document,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code in [200, 201]:
                return True
            else:
                self.logger.warning(f"Failed to ship log to Elasticsearch: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error shipping log to Elasticsearch: {e}")
            return False
    
    def close(self):
        """Close the session."""
        self.session.close()

class FileLogShipper(LogShipper):
    """Log shipper that writes to files (for testing/fallback)."""
    
    def __init__(self, file_path: str, service_name: str):
        super().__init__(file_path, service_name)
        self.file_path = file_path
    
    def ship_log(self, log_entry: Dict[str, Any]) -> bool:
        """Write log to file."""
        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            return True
        except Exception as e:
            self.logger.error(f"Error writing log to file: {e}")
            return False
