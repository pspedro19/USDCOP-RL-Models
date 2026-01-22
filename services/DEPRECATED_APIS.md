# Deprecated APIs

This document tracks deprecated API files in the services directory.

---

## Archived APIs

### multi_model_trading_api.py

- **Status**: Archived
- **Archived Location**: `archive/services-deprecated/multi_model_trading_api.py`
- **Replaced By**: `trading_models_api.py`
- **Archived Date**: 2026-01-17

#### Reason for Deprecation

Duplicate implementation of the multi-model trading API. The original file
(`multi_model_trading_api.py`) used synchronous patterns while
`trading_api_multi_model.py` (now `trading_models_api.py`) implements the
same functionality with modern async-first patterns.

Key differences:
- **Deprecated version**: Synchronous database calls, blocking I/O
- **New version**: Full async/await support with asyncpg and aioredis

#### Migration Path

1. Update all imports from `multi_model_trading_api` to `trading_models_api`
2. Ensure your client code handles async responses appropriately
3. The API endpoints remain compatible - no changes needed to API consumers

```python
# Before (deprecated)
from services.multi_model_trading_api import app

# After (current)
from services.trading_models_api import app
```

---

## Migration Checklist

- [ ] Update Docker Compose service definitions
- [ ] Update Airflow DAG imports (if applicable)
- [ ] Update any direct script references
- [ ] Verify CI/CD pipeline references

---

*Last Updated: 2026-01-17*
