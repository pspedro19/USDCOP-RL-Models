# USDCOP RL Trading System - Backend API Specification

## Overview
This document specifies the REST API endpoints required to support the frontend dashboard components. All endpoints follow RESTful conventions and return JSON responses with consistent error handling.

## Base Configuration
- **Base URL**: `http://localhost:8000/api/v1`
- **Authentication**: Bearer token authentication
- **Content-Type**: `application/json`
- **Rate Limiting**: 1000 requests/hour per API key

## Global Response Format
```json
{
  "success": boolean,
  "data": object | array,
  "timestamp": "ISO8601",
  "meta": {
    "total": number,
    "page": number,
    "limit": number
  }
}
```

## Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message",
    "details": object
  },
  "timestamp": "ISO8601"
}
```

---

## 1. Executive Overview Endpoints

### GET /executive/kpis
**Description**: Real-time KPI metrics for executive dashboard
**Response**:
```json
{
  "success": true,
  "data": {
    "production_gate_l4": {
      "status": "PASS" | "FAIL" | "WARNING",
      "score": 0.956,
      "threshold": 0.95,
      "last_updated": "2025-01-15T10:30:00Z"
    },
    "sortino_ratio": {
      "current": 2.34,
      "target": 2.0,
      "trend": "UP" | "DOWN" | "STABLE"
    },
    "calmar_ratio": {
      "current": 1.82,
      "target": 1.5,
      "trend": "UP"
    },
    "daily_pnl": {
      "amount": 15420.50,
      "currency": "USD",
      "percentage": 0.0234
    },
    "model_accuracy": {
      "ppo_lstm": 0.847,
      "qr_dqn": 0.823,
      "ensemble": 0.865
    },
    "active_positions": 12,
    "total_trades_today": 847
  }
}
```

### GET /executive/performance-chart
**Description**: Historical performance data for charts
**Query Parameters**: 
- `period`: 1d, 7d, 30d, 90d, 1y
- `metric`: pnl, sortino, calmar, sharpe

**Response**:
```json
{
  "success": true,
  "data": {
    "series": [
      {
        "timestamp": "2025-01-15T09:00:00Z",
        "value": 15420.50
      }
    ],
    "statistics": {
      "min": 12000.00,
      "max": 18500.00,
      "avg": 15200.00,
      "volatility": 0.0245
    }
  }
}
```

---

## 2. Live Trading Terminal Endpoints

### GET /trading/real-time-data
**Description**: Real-time USD/COP price feed with technical indicators
**Response**:
```json
{
  "success": true,
  "data": {
    "symbol": "USDCOP",
    "price": 4285.50,
    "change": 12.50,
    "change_percent": 0.29,
    "volume": 1250000,
    "bid": 4285.25,
    "ask": 4285.75,
    "spread": 0.50,
    "timestamp": "2025-01-15T10:30:15Z",
    "technical_indicators": {
      "sma_20": 4280.15,
      "ema_12": 4283.20,
      "rsi": 67.5,
      "macd": {
        "macd": 2.15,
        "signal": 1.80,
        "histogram": 0.35
      },
      "bollinger": {
        "upper": 4295.50,
        "middle": 4285.00,
        "lower": 4274.50
      }
    }
  }
}
```

### GET /trading/historical-data
**Description**: Historical OHLCV data for charting
**Query Parameters**:
- `symbol`: USDCOP
- `interval`: 1m, 5m, 15m, 1h, 4h, 1d
- `start_time`: ISO8601 timestamp
- `end_time`: ISO8601 timestamp
- `limit`: max 1000

**Response**:
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2025-01-15T10:30:00Z",
      "open": 4280.00,
      "high": 4290.00,
      "low": 4275.00,
      "close": 4285.50,
      "volume": 125000
    }
  ]
}
```

### GET /trading/rl-actions
**Description**: Current RL model discrete actions and confidence
**Response**:
```json
{
  "success": true,
  "data": {
    "current_action": "HOLD",
    "confidence": 0.78,
    "action_probabilities": {
      "SELL": 0.12,
      "HOLD": 0.78,
      "BUY": 0.10
    },
    "model_state": {
      "ppo_recommendation": "HOLD",
      "qr_dqn_recommendation": "BUY",
      "ensemble_decision": "HOLD"
    },
    "next_decision_time": "2025-01-15T10:35:00Z"
  }
}
```

### POST /trading/manual-override
**Description**: Manual trading override (emergency stop/start)
**Request Body**:
```json
{
  "action": "STOP" | "START" | "EMERGENCY_STOP",
  "reason": "string",
  "duration_minutes": 60
}
```

---

## 3. RL Model Health Endpoints

### GET /models/health-summary
**Description**: Overall health status of all RL models
**Response**:
```json
{
  "success": true,
  "data": {
    "ppo_lstm": {
      "status": "HEALTHY" | "WARNING" | "CRITICAL",
      "last_training": "2025-01-15T06:00:00Z",
      "performance_score": 0.847,
      "prediction_accuracy": 0.823,
      "training_loss": 0.0234
    },
    "qr_dqn": {
      "status": "HEALTHY",
      "last_training": "2025-01-15T06:00:00Z",
      "performance_score": 0.812,
      "prediction_accuracy": 0.798,
      "training_loss": 0.0298
    },
    "ensemble": {
      "status": "HEALTHY",
      "combined_score": 0.865,
      "agreement_rate": 0.78
    }
  }
}
```

### GET /models/training-metrics
**Description**: Detailed training metrics and convergence data
**Query Parameters**:
- `model`: ppo_lstm, qr_dqn, ensemble
- `period`: 1d, 7d, 30d

**Response**:
```json
{
  "success": true,
  "data": {
    "training_history": [
      {
        "epoch": 1250,
        "loss": 0.0234,
        "reward": 0.78,
        "accuracy": 0.823,
        "timestamp": "2025-01-15T06:00:00Z"
      }
    ],
    "convergence_metrics": {
      "is_converged": true,
      "convergence_epoch": 1180,
      "stability_score": 0.92
    },
    "hyperparameters": {
      "learning_rate": 0.0003,
      "batch_size": 64,
      "epsilon": 0.2,
      "gamma": 0.99
    }
  }
}
```

### GET /models/action-distribution
**Description**: RL action distribution heatmap data
**Response**:
```json
{
  "success": true,
  "data": {
    "hourly_distribution": [
      {
        "hour": 9,
        "SELL": 0.15,
        "HOLD": 0.70,
        "BUY": 0.15
      }
    ],
    "market_condition_distribution": {
      "volatile": {"SELL": 0.25, "HOLD": 0.60, "BUY": 0.15},
      "trending": {"SELL": 0.10, "HOLD": 0.70, "BUY": 0.20},
      "stable": {"SELL": 0.12, "HOLD": 0.80, "BUY": 0.08}
    }
  }
}
```

---

## 4. Risk Management Endpoints

### GET /risk/var-analysis
**Description**: Value at Risk and Conditional VaR analysis
**Query Parameters**:
- `confidence_level`: 95, 99
- `time_horizon`: 1, 5, 10 (days)

**Response**:
```json
{
  "success": true,
  "data": {
    "var_95": {
      "1_day": -125000.00,
      "5_day": -280000.00,
      "10_day": -395000.00
    },
    "cvar_95": {
      "1_day": -156000.00,
      "5_day": -348000.00,
      "10_day": -495000.00
    },
    "portfolio_value": 5000000.00,
    "var_percentage": {
      "1_day": 0.025,
      "5_day": 0.056,
      "10_day": 0.079
    }
  }
}
```

### GET /risk/stress-tests
**Description**: Stress testing scenarios and results
**Response**:
```json
{
  "success": true,
  "data": {
    "scenarios": [
      {
        "name": "COVID-19 Like Event",
        "description": "Market volatility +300%, correlation breakdown",
        "probability": 0.02,
        "expected_loss": -890000.00,
        "time_to_recovery": "45 days",
        "status": "PASS" | "FAIL"
      }
    ],
    "stress_test_summary": {
      "worst_case_loss": -1200000.00,
      "probability_of_ruin": 0.003,
      "capital_adequacy": "SUFFICIENT" | "MARGINAL" | "INSUFFICIENT"
    }
  }
}
```

### GET /risk/exposure-analysis
**Description**: Portfolio exposure breakdown
**Response**:
```json
{
  "success": true,
  "data": {
    "currency_exposure": {
      "USD": 0.65,
      "COP": 0.35
    },
    "position_concentration": {
      "max_single_position": 0.08,
      "herfindahl_index": 0.12
    },
    "leverage": {
      "current": 2.3,
      "maximum_allowed": 3.0,
      "utilization": 0.77
    }
  }
}
```

---

## 5. Data Pipeline Quality Endpoints

### GET /pipeline/quality-gates
**Description**: L0→L4 data pipeline quality gate status
**Response**:
```json
{
  "success": true,
  "data": {
    "l0_raw_data": {
      "status": "PASS",
      "quality_score": 0.98,
      "checks": {
        "completeness": 0.99,
        "timeliness": 0.97,
        "format_validity": 1.0
      },
      "last_check": "2025-01-15T10:25:00Z"
    },
    "l1_cleaned": {
      "status": "PASS",
      "quality_score": 0.96,
      "checks": {
        "outlier_detection": 0.94,
        "missing_data": 0.98,
        "consistency": 0.96
      }
    },
    "l2_features": {
      "status": "WARNING",
      "quality_score": 0.89,
      "checks": {
        "feature_correlation": 0.85,
        "distribution_stability": 0.92,
        "technical_indicators": 0.91
      }
    },
    "l3_model_ready": {
      "status": "PASS",
      "quality_score": 0.94,
      "checks": {
        "normalization": 0.96,
        "feature_engineering": 0.93,
        "data_leakage": 1.0
      }
    },
    "l4_production": {
      "status": "PASS",
      "quality_score": 0.96,
      "checks": {
        "model_validation": 0.94,
        "prediction_quality": 0.97,
        "latency": 0.98
      }
    }
  }
}
```

### GET /pipeline/anti-leakage-checks
**Description**: Data leakage prevention monitoring
**Response**:
```json
{
  "success": true,
  "data": {
    "temporal_leakage": {
      "status": "CLEAN",
      "future_data_detected": false,
      "last_scan": "2025-01-15T10:00:00Z"
    },
    "feature_leakage": {
      "status": "CLEAN", 
      "suspicious_correlations": [],
      "information_coefficient": 0.12
    },
    "model_leakage": {
      "status": "CLEAN",
      "target_correlation": 0.05,
      "feature_importance_drift": 0.02
    }
  }
}
```

### GET /pipeline/system-resources
**Description**: System resource utilization
**Response**:
```json
{
  "success": true,
  "data": {
    "airflow_cluster": {
      "cpu_usage": 0.65,
      "memory_usage": 0.78,
      "disk_usage": 0.45,
      "active_tasks": 12,
      "queued_tasks": 3
    },
    "minio_storage": {
      "total_capacity": "10TB",
      "used_capacity": "3.2TB",
      "utilization": 0.32,
      "iops": 2500
    },
    "model_inference": {
      "avg_latency_ms": 45,
      "throughput_rps": 150,
      "gpu_utilization": 0.82
    }
  }
}
```

---

## 6. Audit & Compliance Endpoints

### GET /audit/traceability
**Description**: Data and model traceability with SHA256 hashes
**Query Parameters**:
- `entity_type`: data, model, trade, decision
- `entity_id`: specific ID to trace
- `start_date`: ISO8601
- `end_date`: ISO8601

**Response**:
```json
{
  "success": true,
  "data": {
    "traceability_chain": [
      {
        "entity_id": "trade_20250115_001",
        "entity_type": "trade",
        "hash": "a1b2c3d4e5f6...",
        "parent_hash": "f6e5d4c3b2a1...",
        "timestamp": "2025-01-15T10:30:00Z",
        "metadata": {
          "model_version": "v2.1.3",
          "data_version": "20250115_0900",
          "decision_logic": "ensemble_weighted"
        }
      }
    ],
    "integrity_status": "VERIFIED" | "CORRUPTED",
    "chain_completeness": 0.999
  }
}
```

### GET /audit/regulatory-compliance
**Description**: Regulatory compliance status (SFC Colombia, Basel III)
**Response**:
```json
{
  "success": true,
  "data": {
    "sfc_colombia": {
      "status": "COMPLIANT",
      "last_assessment": "2025-01-10T00:00:00Z",
      "requirements": {
        "risk_management": "PASS",
        "model_governance": "PASS",
        "operational_controls": "PASS",
        "reporting": "PASS"
      }
    },
    "basel_iii": {
      "status": "COMPLIANT",
      "capital_ratios": {
        "tier1_ratio": 0.124,
        "total_capital_ratio": 0.156,
        "leverage_ratio": 0.045
      },
      "liquidity_ratios": {
        "lcr": 1.25,
        "nsfr": 1.18
      }
    },
    "model_risk_management": {
      "validation_frequency": "monthly",
      "last_validation": "2025-01-01T00:00:00Z",
      "independent_validation": true,
      "backtesting_results": "PASS"
    }
  }
}
```

### GET /audit/security-compliance
**Description**: Security and cybersecurity compliance status
**Response**:
```json
{
  "success": true,
  "data": {
    "security_frameworks": {
      "iso_27001": {
        "status": "CERTIFIED",
        "certificate_expiry": "2025-12-31",
        "last_audit": "2024-06-15"
      },
      "nist_framework": {
        "status": "COMPLIANT",
        "maturity_level": "4",
        "last_assessment": "2024-11-20"
      }
    },
    "vulnerability_management": {
      "critical_vulnerabilities": 0,
      "high_vulnerabilities": 2,
      "medium_vulnerabilities": 8,
      "last_scan": "2025-01-14T02:00:00Z"
    },
    "access_controls": {
      "mfa_enabled": true,
      "privileged_access_managed": true,
      "session_monitoring": true
    }
  }
}
```

### GET /audit/audit-history
**Description**: Historical audit logs and compliance reports
**Query Parameters**:
- `audit_type`: security, compliance, model, operational
- `start_date`: ISO8601
- `end_date`: ISO8601
- `limit`: max 100

**Response**:
```json
{
  "success": true,
  "data": [
    {
      "audit_id": "AUD_20250115_001",
      "audit_type": "model_validation",
      "auditor": "Internal Risk Team",
      "status": "COMPLETED",
      "findings": 3,
      "recommendations": 2,
      "risk_rating": "LOW",
      "completion_date": "2025-01-15T16:00:00Z",
      "report_url": "/reports/AUD_20250115_001.pdf"
    }
  ]
}
```

---

## 7. Authentication & Authorization

### POST /auth/login
**Description**: User authentication
**Request Body**:
```json
{
  "username": "string",
  "password": "string",
  "mfa_token": "string"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "access_token": "jwt_token",
    "refresh_token": "refresh_token",
    "expires_in": 3600,
    "user": {
      "id": "user_id",
      "username": "username",
      "role": "admin" | "trader" | "analyst" | "viewer"
    }
  }
}
```

### POST /auth/refresh
**Description**: Refresh access token
**Request Body**:
```json
{
  "refresh_token": "string"
}
```

---

## 8. WebSocket Endpoints

### WS /ws/real-time-feed
**Description**: Real-time data feed for live updates
**Subscription Topics**:
- `price_feed`: USD/COP price updates
- `rl_decisions`: RL model decisions
- `risk_alerts`: Risk management alerts
- `pipeline_status`: Data pipeline status updates

**Message Format**:
```json
{
  "topic": "price_feed",
  "data": {
    "symbol": "USDCOP",
    "price": 4285.50,
    "timestamp": "2025-01-15T10:30:15Z"
  }
}
```

---

## Implementation Notes

1. **Rate Limiting**: Implement Redis-based rate limiting
2. **Caching**: Use Redis for caching frequently accessed data (5-minute TTL)
3. **Database**: PostgreSQL for transactional data, InfluxDB for time-series data
4. **Security**: JWT authentication, HTTPS only, API key validation
5. **Monitoring**: Implement comprehensive logging and metrics collection
6. **Error Handling**: Consistent error codes and messages across all endpoints
7. **Versioning**: API versioning in URL path (/api/v1/)
8. **Documentation**: OpenAPI 3.0 specification for interactive documentation

## Status Codes
- 200: Success
- 201: Created
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error
- 503: Service Unavailable