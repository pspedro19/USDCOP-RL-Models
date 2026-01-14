# API Versioning Policy

## Overview

This document describes the API versioning strategy for USDCOP Trading System services.

## Versioning Scheme

All API endpoints use URL path versioning:

```
/api/v{major}/endpoint
```

**Current Version:** `v1`

### Examples

```
GET  /api/v1/health
POST /api/v1/backtest
GET  /api/v1/models
POST /api/v1/inference/{model_id}
```

## Version Lifecycle

### Supported Versions

| Version | Status     | Support Until | Notes                |
|---------|------------|---------------|----------------------|
| v1      | Current    | Indefinite    | Active development   |

### Version States

- **Current**: Actively developed, receives new features
- **Maintained**: Receives bug fixes and security patches only
- **Deprecated**: Scheduled for removal, avoid using
- **Removed**: No longer available

## Deprecation Policy

1. **Announcement**: Deprecation announced minimum 90 days before removal
2. **Header Warning**: Deprecated endpoints return `Deprecation` header
3. **Documentation**: Deprecated endpoints marked in OpenAPI spec
4. **Migration Guide**: Published with deprecation notice

### Deprecation Headers

```http
Deprecation: Sun, 01 Mar 2026 00:00:00 GMT
Sunset: Sun, 01 Jun 2026 00:00:00 GMT
Link: </api/v2/endpoint>; rel="successor-version"
```

## Breaking Changes

The following are considered **breaking changes** and require a major version bump:

1. Removing an endpoint
2. Removing a required request field
3. Removing a response field
4. Changing field types
5. Changing authentication requirements
6. Changing error response format

### Non-Breaking Changes

These can be made without version bump:

1. Adding new optional request fields
2. Adding new response fields
3. Adding new endpoints
4. Adding new enum values (when client handles unknown values)
5. Fixing bugs in documented behavior
6. Performance improvements

## Error Responses

All versions use consistent error format:

```json
{
  "error": "error_code",
  "message": "Human readable message",
  "details": {},
  "request_id": "correlation-id"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `validation_error` | 400 | Invalid request parameters |
| `unauthorized` | 401 | Missing or invalid auth |
| `forbidden` | 403 | Insufficient permissions |
| `not_found` | 404 | Resource not found |
| `rate_limit_exceeded` | 429 | Too many requests |
| `internal_error` | 500 | Server error |

## Security Headers

All responses include:

```http
X-Request-ID: correlation-id
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642694400
```

## Rate Limiting

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Health checks | Unlimited | - |
| Read (GET) | 100 req | 1 minute |
| Write (POST) | 60 req | 1 minute |
| Backtest | 10 req | 1 minute |

## Migration Guides

### v0 to v1

The initial public release. No migration needed.

### Future v1 to v2

When v2 is released, a detailed migration guide will be published here.

## Client Integration

### Python

```python
import httpx

client = httpx.Client(
    base_url="http://localhost:8000/api/v1",
    headers={"X-Request-ID": "custom-correlation-id"}
)

response = client.get("/health")
```

### TypeScript

```typescript
const API_BASE = '/api/v1';

async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`, {
    headers: {
      'X-Request-ID': crypto.randomUUID()
    }
  });
  return response.json();
}
```

### cURL

```bash
curl -H "X-Request-ID: test-123" http://localhost:8000/api/v1/health
```

## OpenAPI Specification

The OpenAPI spec is available at:

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **JSON Spec**: `/openapi.json`
- **Export**: `/openapi-export`

## Changelog

### v1.0.0 (2025-01-14)

- Initial release
- Rate limiting with Token Bucket algorithm
- Correlation ID tracking
- Structured request logging
- API versioning with `/api/v1/` prefix

## Contact

For API questions or issues:
- GitHub Issues: [USDCOP-RL-Models](https://github.com/your-repo/USDCOP-RL-Models/issues)
- Email: trading@example.com
