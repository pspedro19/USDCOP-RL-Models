# 🚀 SPEC: Trading Execution SaaS Platform
## Spec-Driven Development + AI-Augmented TDD

**Proyecto**: SignalBridge (nombre tentativo)
**Dominio**: Trading Automatizado / FinTech
**Criticidad**: 🔴 ALTA (dinero real, APIs financieras)
**Versión Spec**: 1.0
**Fecha**: 2026-01-18

---

# PARTE 1: VISIÓN Y ALCANCE

## 1.1 Objetivo del Sistema

Plataforma SaaS que permite a usuarios:
1. Conectar sus cuentas de exchanges (MEXC, Binance)
2. Configurar pares de trading y parámetros de riesgo
3. Recibir señales de un modelo RL backend (ya existente)
4. Ejecutar trades automáticamente en múltiples exchanges
5. Monitorear posiciones y performance en tiempo real

## 1.2 Usuarios Target

| Usuario | Descripción | Necesidades |
|---------|-------------|-------------|
| Trader Individual | Tiene cuentas en MEXC/Binance | Automatizar ejecución de señales |
| El Operador (tú) | Administrador del sistema | Gestionar modelo, monitorear usuarios |

## 1.3 Fuera de Alcance (v1)

- [ ] Múltiples estrategias por usuario
- [ ] Backtesting en plataforma
- [ ] Social trading / copy trading
- [ ] Soporte para más de 2 exchanges
- [ ] Mobile app nativa

---

# PARTE 2: ARQUITECTURA POR CONTRATOS

## 2.1 Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SIGNALBRIDGE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     CT-AUTH      ┌──────────────┐                        │
│  │   Frontend   │◄───────────────►│  Auth Service │                        │
│  │   (React)    │                  │   (Supabase)  │                        │
│  └──────┬───────┘                  └───────┬──────┘                        │
│         │                                  │                                │
│         │ CT-API                           │ CT-USER                        │
│         ▼                                  ▼                                │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                      API GATEWAY (FastAPI)                   │           │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │           │
│  │  │ /users/*    │  │ /exchanges/*│  │ /signals/*          │  │           │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │           │
│  └─────────────────────────────────────────────────────────────┘           │
│         │                    │                    │                         │
│         │ CT-DB              │ CT-VAULT           │ CT-SIGNAL               │
│         ▼                    ▼                    ▼                         │
│  ┌────────────┐      ┌────────────┐      ┌────────────────────┐            │
│  │ PostgreSQL │      │   Vault    │      │  Signal Processor  │            │
│  │ (Supabase) │      │ (Secrets)  │      │    (Celery/Redis)  │            │
│  └────────────┘      └────────────┘      └─────────┬──────────┘            │
│                                                    │                        │
│                                                    │ CT-EXEC                │
│                                                    ▼                        │
│                              ┌──────────────────────────────────┐          │
│                              │      EXECUTION ENGINE            │          │
│                              │  ┌────────────┐ ┌────────────┐   │          │
│                              │  │MEXC Adapter│ │Binance Adpt│   │          │
│                              │  └────────────┘ └────────────┘   │          │
│                              └──────────────────────────────────┘          │
│                                          │                                  │
│                                          │ CT-EXCHANGE                      │
│                                          ▼                                  │
│                              ┌──────────────────────────────────┐          │
│                              │    EXTERNAL EXCHANGES            │          │
│                              │    (MEXC API / Binance API)      │          │
│                              └──────────────────────────────────┘          │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │              ML SIGNAL GENERATOR (Sistema Existente)         │          │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────────────────────┐   │          │
│  │  │ Airflow │───►│ Model   │───►│ Signal Publisher        │   │          │
│  │  │ DAGs    │    │ Inference│   │ (Webhook/Queue)         │   │          │
│  │  └─────────┘    └─────────┘    └─────────────────────────┘   │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                              │                                              │
│                              │ CT-ML-SIGNAL                                 │
│                              ▼                                              │
│                    (Conecta con Signal Processor arriba)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Inventario de Contratos

| ID | Contrato | Productor | Consumidor | Tipo |
|----|----------|-----------|------------|------|
| CT-AUTH | Auth Token/Session | Auth Service | Frontend, API | JWT |
| CT-USER | User Profile | Auth Service | API Gateway | JSON Schema |
| CT-API | REST Endpoints | API Gateway | Frontend | OpenAPI 3.0 |
| CT-DB | Data Models | PostgreSQL | API Gateway | SQLAlchemy |
| CT-VAULT | API Keys encrypted | Vault | Execution Engine | Encrypted blob |
| CT-SIGNAL | Trading Signal | Signal Processor | Execution Engine | Pydantic |
| CT-EXEC | Execution Request | Execution Engine | Exchange Adapters | Pydantic |
| CT-EXCHANGE | Exchange Response | MEXC/Binance | Execution Engine | JSON |
| CT-ML-SIGNAL | ML Prediction | ML System | Signal Processor | Webhook/Queue |

---

# PARTE 3: ESPECIFICACIÓN DE CONTRATOS

## 3.1 CT-AUTH: Contrato de Autenticación

### Schema

```python
# contracts/auth.py
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional
from enum import Enum

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"

class AuthToken(BaseModel):
    """JWT Token response."""
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    refresh_token: Optional[str] = None

class UserSession(BaseModel):
    """Decoded JWT payload."""
    user_id: str = Field(..., description="UUID del usuario")
    email: EmailStr
    role: UserRole
    iat: datetime  # issued at
    exp: datetime  # expiration
    
class SignUpRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    
class LoginRequest(BaseModel):
    email: EmailStr
    password: str
```

### Tests de Contrato

```python
# tests/contracts/test_auth_contract.py
import pytest
from contracts.auth import AuthToken, UserSession, LoginRequest

class TestAuthContract:
    """Tests que verifican el contrato de auth."""
    
    def test_auth_token_valid_structure(self):
        """Token debe tener estructura correcta."""
        token = AuthToken(
            access_token="eyJ...",
            expires_at=datetime.now() + timedelta(hours=1)
        )
        assert token.token_type == "bearer"
        assert token.access_token.startswith("eyJ")
    
    def test_user_session_from_jwt(self):
        """Session debe parsearse de JWT válido."""
        payload = {
            "user_id": "uuid-123",
            "email": "test@example.com",
            "role": "user",
            "iat": datetime.now(),
            "exp": datetime.now() + timedelta(hours=1)
        }
        session = UserSession(**payload)
        assert session.role == UserRole.USER
    
    def test_login_request_validation(self):
        """Password debe tener mínimo 8 caracteres."""
        with pytest.raises(ValidationError):
            LoginRequest(email="test@test.com", password="123")
```

---

## 3.2 CT-USER: Contrato de Perfil de Usuario

### Schema

```python
# contracts/user.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"  # max 1% per trade
    MODERATE = "moderate"          # max 2% per trade
    AGGRESSIVE = "aggressive"      # max 5% per trade

class UserProfile(BaseModel):
    """Perfil completo del usuario."""
    id: str
    email: str
    created_at: datetime
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    risk_profile: RiskProfile = RiskProfile.MODERATE
    max_daily_trades: int = Field(default=10, ge=1, le=100)
    max_position_size_pct: float = Field(default=0.02, ge=0.001, le=0.1)
    connected_exchanges: List[str] = []
    is_active: bool = True
    
class UserProfileUpdate(BaseModel):
    """Campos actualizables del perfil."""
    risk_profile: Optional[RiskProfile] = None
    max_daily_trades: Optional[int] = Field(None, ge=1, le=100)
    max_position_size_pct: Optional[float] = Field(None, ge=0.001, le=0.1)
```

### Reglas de Negocio

| Regla | Descripción | Validación |
|-------|-------------|------------|
| RN-U01 | Usuario FREE: máx 5 trades/día | Backend enforce |
| RN-U02 | Usuario FREE: solo 1 exchange | Backend enforce |
| RN-U03 | Position size no puede exceder risk profile | Backend enforce |
| RN-U04 | Email único en sistema | DB constraint |

---

## 3.3 CT-VAULT: Contrato de API Keys (Seguridad Crítica)

### Schema

```python
# contracts/vault.py
from pydantic import BaseModel, Field, SecretStr
from typing import Optional
from datetime import datetime
from enum import Enum

class ExchangeType(str, Enum):
    MEXC = "mexc"
    BINANCE = "binance"

class ExchangeCredentials(BaseModel):
    """Credenciales de exchange - NUNCA loggear."""
    exchange: ExchangeType
    api_key: SecretStr
    api_secret: SecretStr
    passphrase: Optional[SecretStr] = None  # Algunos exchanges lo requieren
    
    class Config:
        # Prevenir que secrets aparezcan en logs/repr
        json_encoders = {
            SecretStr: lambda v: "***REDACTED***"
        }

class StoredCredentials(BaseModel):
    """Representación en DB - encriptada."""
    id: str
    user_id: str
    exchange: ExchangeType
    encrypted_blob: bytes  # AES-256-GCM encrypted
    key_fingerprint: str   # Para identificar sin decriptar
    created_at: datetime
    last_used_at: Optional[datetime] = None
    is_valid: bool = True  # Se marca False si exchange rechaza

class CredentialsValidationResult(BaseModel):
    """Resultado de validar credenciales con exchange."""
    is_valid: bool
    exchange: ExchangeType
    permissions: List[str] = []  # ['spot', 'futures', 'withdraw']
    error_message: Optional[str] = None
```

### Reglas de Seguridad

| Regla | Descripción | Implementación |
|-------|-------------|----------------|
| SEC-01 | API keys encriptadas at-rest | AES-256-GCM |
| SEC-02 | Keys nunca en logs | SecretStr + middleware |
| SEC-03 | Keys nunca en response | Exclusión explícita |
| SEC-04 | Validar permisos mínimos | Solo SPOT, no WITHDRAW |
| SEC-05 | Rate limit por usuario | 10 req/min a exchanges |
| SEC-06 | Audit log de uso de keys | Tabla separada |

### Tests de Seguridad

```python
# tests/contracts/test_vault_security.py
import pytest
from contracts.vault import ExchangeCredentials, StoredCredentials

class TestVaultSecurity:
    """Tests de seguridad para manejo de API keys."""
    
    def test_credentials_never_in_repr(self):
        """Secrets no deben aparecer en repr/str."""
        creds = ExchangeCredentials(
            exchange="mexc",
            api_key="my-secret-key",
            api_secret="my-secret-secret"
        )
        repr_str = repr(creds)
        assert "my-secret-key" not in repr_str
        assert "my-secret-secret" not in repr_str
    
    def test_credentials_never_in_json(self):
        """Secrets no deben aparecer en JSON export."""
        creds = ExchangeCredentials(
            exchange="mexc",
            api_key="my-secret-key",
            api_secret="my-secret-secret"
        )
        json_str = creds.json()
        assert "my-secret-key" not in json_str
        assert "REDACTED" in json_str
    
    def test_stored_credentials_encrypted(self):
        """Verificar que blob está encriptado."""
        stored = StoredCredentials(
            id="uuid",
            user_id="user-uuid",
            exchange="mexc",
            encrypted_blob=b"encrypted_data",
            key_fingerprint="abc123",
            created_at=datetime.now()
        )
        # No debe contener plaintext
        assert b"api_key" not in stored.encrypted_blob
```

---

## 3.4 CT-SIGNAL: Contrato de Señales de Trading

### Schema

```python
# contracts/signal.py
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from enum import IntEnum

class TradingAction(IntEnum):
    """Acción de trading - SSOT."""
    SELL = 0
    HOLD = 1
    BUY = 2

class SignalSource(str, Enum):
    ML_MODEL = "ml_model"
    MANUAL = "manual"
    EXTERNAL = "external"

class TradingSignal(BaseModel):
    """Señal de trading del modelo ML."""
    signal_id: str = Field(..., description="UUID único de la señal")
    timestamp: datetime
    symbol: str = Field(..., pattern=r"^[A-Z]{3,10}/[A-Z]{3,10}$")  # e.g., "USD/COP"
    action: TradingAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    source: SignalSource = SignalSource.ML_MODEL
    
    # Metadata opcional
    features_snapshot: Optional[dict] = None
    
    class Config:
        schema_extra = {
            "example": {
                "signal_id": "sig-20260118-143000-001",
                "timestamp": "2026-01-18T14:30:00Z",
                "symbol": "USD/COP",
                "action": 2,
                "confidence": 0.85,
                "model_version": "v21",
                "source": "ml_model"
            }
        }

class SignalAcknowledgment(BaseModel):
    """Confirmación de recepción de señal."""
    signal_id: str
    received_at: datetime
    status: Literal["received", "queued", "processing", "rejected"]
    rejection_reason: Optional[str] = None
```

### Webhook Contract (ML → Signal Processor)

```yaml
# openapi/signal_webhook.yaml
openapi: 3.0.0
info:
  title: Signal Webhook API
  version: 1.0.0
paths:
  /webhook/signal:
    post:
      summary: Receive trading signal from ML system
      security:
        - ApiKeyAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TradingSignal'
      responses:
        '200':
          description: Signal acknowledged
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SignalAcknowledgment'
        '401':
          description: Invalid API key
        '422':
          description: Invalid signal format
```

---

## 3.5 CT-EXEC: Contrato de Ejecución

### Schema

```python
# contracts/execution.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum
from decimal import Decimal

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

class ExecutionRequest(BaseModel):
    """Request para ejecutar trade en exchange."""
    request_id: str
    user_id: str
    signal_id: str  # Referencia a señal origen
    exchange: ExchangeType
    symbol: str  # Exchange-specific format (USDTCOP, USDT_COP, etc.)
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal = Field(..., gt=0)
    price: Optional[Decimal] = None  # Solo para LIMIT
    stop_loss_pct: Optional[float] = Field(None, ge=0.001, le=0.1)
    take_profit_pct: Optional[float] = Field(None, ge=0.001, le=0.5)
    
class ExecutionResult(BaseModel):
    """Resultado de ejecución."""
    request_id: str
    exchange_order_id: str
    status: OrderStatus
    filled_quantity: Decimal
    filled_price: Decimal
    fees: Decimal
    fees_currency: str
    executed_at: datetime
    error_message: Optional[str] = None

class ExecutionBatch(BaseModel):
    """Batch de ejecuciones (multi-exchange)."""
    signal_id: str
    executions: List[ExecutionRequest]
    created_at: datetime
```

### State Machine de Órdenes

```
┌─────────┐     submit      ┌───────────┐
│ PENDING │────────────────►│ SUBMITTED │
└─────────┘                 └─────┬─────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
              ┌─────────┐  ┌───────────┐  ┌──────────┐
              │ PARTIAL │  │  FILLED   │  │ REJECTED │
              └────┬────┘  └───────────┘  └──────────┘
                   │
                   ▼
              ┌─────────┐
              │ FILLED  │
              └─────────┘
              
              (CANCELLED puede venir de PENDING o PARTIAL)
              (FAILED puede venir de cualquier estado)
```

---

## 3.6 CT-EXCHANGE: Adaptadores de Exchange

### Interface Común

```python
# contracts/exchange_adapter.py
from abc import ABC, abstractmethod
from typing import Optional
from decimal import Decimal

class ExchangeAdapter(ABC):
    """Interface que todos los adaptadores deben implementar."""
    
    @abstractmethod
    async def validate_credentials(self) -> CredentialsValidationResult:
        """Validar que las credenciales funcionan."""
        pass
    
    @abstractmethod
    async def get_balance(self, currency: str) -> Decimal:
        """Obtener balance de una moneda."""
        pass
    
    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Info del par (min_qty, price_precision, etc.)."""
        pass
    
    @abstractmethod
    async def place_market_order(
        self, 
        symbol: str, 
        side: OrderSide, 
        quantity: Decimal
    ) -> ExecutionResult:
        """Ejecutar orden de mercado."""
        pass
    
    @abstractmethod
    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal
    ) -> ExecutionResult:
        """Ejecutar orden límite."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancelar orden pendiente."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> ExecutionResult:
        """Obtener estado actual de orden."""
        pass

class SymbolInfo(BaseModel):
    """Info de un par de trading."""
    symbol: str
    base_currency: str
    quote_currency: str
    min_quantity: Decimal
    max_quantity: Decimal
    quantity_precision: int
    price_precision: int
    min_notional: Decimal  # Mínimo valor de orden
```

### Mapeo de Símbolos

```python
# contracts/symbol_mapping.py

SYMBOL_MAPPINGS = {
    "USD/COP": {
        "mexc": "USDTCOP",      # MEXC usa USDT como proxy
        "binance": "USDTCOP",   # Binance igual
    },
    # Agregar más pares según necesidad
}

def get_exchange_symbol(internal_symbol: str, exchange: ExchangeType) -> str:
    """Convierte símbolo interno a formato del exchange."""
    if internal_symbol not in SYMBOL_MAPPINGS:
        raise ValueError(f"Symbol {internal_symbol} not supported")
    return SYMBOL_MAPPINGS[internal_symbol][exchange.value]
```

---

# PARTE 4: ESPECIFICACIÓN DE APIs

## 4.1 API Gateway - OpenAPI Spec

```yaml
# openapi/api_gateway.yaml
openapi: 3.0.0
info:
  title: SignalBridge API
  version: 1.0.0
  description: Trading Execution SaaS Platform

servers:
  - url: https://api.signalbridge.io/v1
    description: Production
  - url: http://localhost:8000/v1
    description: Development

security:
  - BearerAuth: []

paths:
  # ═══════════════ USER ENDPOINTS ═══════════════
  /users/me:
    get:
      tags: [Users]
      summary: Get current user profile
      responses:
        '200':
          description: User profile
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserProfile'
    
    patch:
      tags: [Users]
      summary: Update user profile
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserProfileUpdate'
      responses:
        '200':
          description: Updated profile

  # ═══════════════ EXCHANGE ENDPOINTS ═══════════════
  /exchanges:
    get:
      tags: [Exchanges]
      summary: List connected exchanges
      responses:
        '200':
          description: List of connected exchanges
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/ConnectedExchange'
  
  /exchanges/{exchange}/connect:
    post:
      tags: [Exchanges]
      summary: Connect exchange API keys
      parameters:
        - name: exchange
          in: path
          required: true
          schema:
            $ref: '#/components/schemas/ExchangeType'
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ExchangeCredentialsInput'
      responses:
        '201':
          description: Exchange connected
        '400':
          description: Invalid credentials
  
  /exchanges/{exchange}/disconnect:
    delete:
      tags: [Exchanges]
      summary: Disconnect exchange
      parameters:
        - name: exchange
          in: path
          required: true
          schema:
            $ref: '#/components/schemas/ExchangeType'
      responses:
        '204':
          description: Disconnected

  /exchanges/{exchange}/validate:
    post:
      tags: [Exchanges]
      summary: Validate exchange connection
      responses:
        '200':
          description: Validation result
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CredentialsValidationResult'

  /exchanges/{exchange}/balance:
    get:
      tags: [Exchanges]
      summary: Get exchange balance
      responses:
        '200':
          description: Balance info
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ExchangeBalance'

  # ═══════════════ TRADING CONFIG ENDPOINTS ═══════════════
  /trading/config:
    get:
      tags: [Trading]
      summary: Get trading configuration
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TradingConfig'
    
    put:
      tags: [Trading]
      summary: Update trading configuration
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TradingConfigUpdate'
      responses:
        '200':
          description: Updated config

  /trading/toggle:
    post:
      tags: [Trading]
      summary: Enable/disable auto-trading
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                enabled:
                  type: boolean
      responses:
        '200':
          description: Trading toggled

  # ═══════════════ SIGNALS & EXECUTIONS ═══════════════
  /signals:
    get:
      tags: [Signals]
      summary: List recent signals
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 50
        - name: since
          in: query
          schema:
            type: string
            format: date-time
      responses:
        '200':
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/TradingSignal'

  /executions:
    get:
      tags: [Executions]
      summary: List trade executions
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 50
        - name: exchange
          in: query
          schema:
            $ref: '#/components/schemas/ExchangeType'
      responses:
        '200':
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/ExecutionResult'

  /executions/{execution_id}:
    get:
      tags: [Executions]
      summary: Get execution details
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ExecutionDetail'

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  schemas:
    # Referencias a los Pydantic models definidos arriba
    UserProfile:
      # ... (mapea a contracts/user.py)
    ExchangeType:
      type: string
      enum: [mexc, binance]
    # ... etc
```

---

# PARTE 5: MODELOS DE BASE DE DATOS

## 5.1 Schema PostgreSQL

```sql
-- migrations/001_initial_schema.sql

-- ═══════════════ USERS ═══════════════
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    subscription_tier VARCHAR(20) DEFAULT 'free',
    risk_profile VARCHAR(20) DEFAULT 'moderate',
    max_daily_trades INTEGER DEFAULT 10,
    max_position_size_pct DECIMAL(5,4) DEFAULT 0.02,
    is_active BOOLEAN DEFAULT TRUE,
    
    CONSTRAINT valid_tier CHECK (subscription_tier IN ('free', 'pro', 'enterprise')),
    CONSTRAINT valid_risk CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive'))
);

-- ═══════════════ EXCHANGE CREDENTIALS ═══════════════
CREATE TABLE exchange_credentials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    exchange VARCHAR(20) NOT NULL,
    encrypted_blob BYTEA NOT NULL,
    key_fingerprint VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    is_valid BOOLEAN DEFAULT TRUE,
    
    CONSTRAINT valid_exchange CHECK (exchange IN ('mexc', 'binance')),
    UNIQUE(user_id, exchange)
);

-- ═══════════════ TRADING CONFIG ═══════════════
CREATE TABLE trading_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    symbol VARCHAR(20) DEFAULT 'USD/COP',
    is_enabled BOOLEAN DEFAULT FALSE,
    stop_loss_pct DECIMAL(5,4) DEFAULT 0.02,
    take_profit_pct DECIMAL(5,4) DEFAULT 0.05,
    min_confidence DECIMAL(3,2) DEFAULT 0.70,
    execute_on_exchanges JSONB DEFAULT '["mexc", "binance"]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════ SIGNALS ═══════════════
CREATE TABLE signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id VARCHAR(50) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    action SMALLINT NOT NULL,
    confidence DECIMAL(4,3) NOT NULL,
    model_version VARCHAR(20),
    source VARCHAR(20) DEFAULT 'ml_model',
    features_snapshot JSONB,
    received_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_action CHECK (action IN (0, 1, 2))
);

CREATE INDEX idx_signals_timestamp ON signals(timestamp DESC);

-- ═══════════════ EXECUTIONS ═══════════════
CREATE TABLE executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(50) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id),
    signal_id UUID REFERENCES signals(id),
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) DEFAULT 'market',
    requested_quantity DECIMAL(20,8) NOT NULL,
    exchange_order_id VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending',
    filled_quantity DECIMAL(20,8),
    filled_price DECIMAL(20,8),
    fees DECIMAL(20,8),
    fees_currency VARCHAR(10),
    executed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_status CHECK (status IN 
        ('pending', 'submitted', 'partial', 'filled', 'cancelled', 'rejected', 'failed'))
);

CREATE INDEX idx_executions_user ON executions(user_id, created_at DESC);
CREATE INDEX idx_executions_signal ON executions(signal_id);

-- ═══════════════ AUDIT LOG ═══════════════
CREATE TABLE credential_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    exchange VARCHAR(20) NOT NULL,
    action VARCHAR(50) NOT NULL,  -- 'validate', 'use_for_trade', 'rotate'
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_user ON credential_audit_log(user_id, created_at DESC);
```

---

# PARTE 6: FLUJOS DE NEGOCIO

## 6.1 Flujo: Usuario Conecta Exchange

```
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌────────┐     ┌─────────┐
│ Frontend│     │   API    │     │  Vault  │     │Adapter │     │Exchange │
└────┬────┘     └────┬─────┘     └────┬────┘     └───┬────┘     └────┬────┘
     │               │                │              │               │
     │ POST /exchanges/mexc/connect   │              │               │
     │──────────────►│                │              │               │
     │               │                │              │               │
     │               │ encrypt(keys)  │              │               │
     │               │───────────────►│              │               │
     │               │                │              │               │
     │               │    blob        │              │               │
     │               │◄───────────────│              │               │
     │               │                │              │               │
     │               │ validate(keys) │              │               │
     │               │────────────────────────────►  │               │
     │               │                │              │ GET /account  │
     │               │                │              │──────────────►│
     │               │                │              │               │
     │               │                │              │   permissions │
     │               │                │              │◄──────────────│
     │               │                │              │               │
     │               │    ValidationResult           │               │
     │               │◄────────────────────────────  │               │
     │               │                │              │               │
     │               │ store(blob)    │              │               │
     │               │───────────────►│              │               │
     │               │                │              │               │
     │   201 Created │                │              │               │
     │◄──────────────│                │              │               │
     │               │                │              │               │
```

## 6.2 Flujo: Señal → Ejecución

```
┌────────┐    ┌─────────┐    ┌───────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ML Model│    │ Webhook │    │ Signal    │    │Execution│    │Exchange │    │ MEXC/   │
│        │    │Receiver │    │ Processor │    │ Engine  │    │ Adapter │    │ Binance │
└───┬────┘    └────┬────┘    └─────┬─────┘    └────┬────┘    └────┬────┘    └────┬────┘
    │              │               │               │              │              │
    │ POST /webhook/signal         │               │              │              │
    │─────────────►│               │               │              │              │
    │              │               │               │              │              │
    │              │ validate+queue│               │              │              │
    │              │──────────────►│               │              │              │
    │              │               │               │              │              │
    │    ack       │               │               │              │              │
    │◄─────────────│               │               │              │              │
    │              │               │               │              │              │
    │              │               │ for each user │              │              │
    │              │               │ with trading ON              │              │
    │              │               │───────────────────────────►  │              │
    │              │               │               │              │              │
    │              │               │               │ check config │              │
    │              │               │               │ (confidence, │              │
    │              │               │               │  risk limits)│              │
    │              │               │               │              │              │
    │              │               │               │ for each     │              │
    │              │               │               │ exchange     │              │
    │              │               │               │─────────────►│              │
    │              │               │               │              │              │
    │              │               │               │              │ place_order  │
    │              │               │               │              │─────────────►│
    │              │               │               │              │              │
    │              │               │               │              │   result     │
    │              │               │               │              │◄─────────────│
    │              │               │               │              │              │
    │              │               │               │ ExecutionResult              │
    │              │               │               │◄─────────────│              │
    │              │               │               │              │              │
    │              │               │               │ save to DB   │              │
    │              │               │               │              │              │
```

---

# PARTE 7: TESTS DE ACEPTACIÓN

## 7.1 User Stories & Acceptance Criteria

### US-01: Como usuario, quiero conectar mi cuenta de MEXC

```gherkin
Feature: Exchange Connection
  
  Scenario: Successfully connect MEXC account
    Given I am a logged in user
    And I have valid MEXC API credentials with SPOT permissions
    When I submit my API key and secret
    Then the system should validate credentials with MEXC
    And store encrypted credentials
    And show "MEXC Connected" status
    
  Scenario: Reject credentials without SPOT permission
    Given I am a logged in user
    And I have MEXC credentials with only FUTURES permission
    When I submit my credentials
    Then the system should reject with error "SPOT permission required"
    And not store any credentials
    
  Scenario: Reject invalid credentials
    Given I am a logged in user
    When I submit invalid API credentials
    Then the system should show error "Invalid API credentials"
    And not store any credentials
```

### US-02: Como usuario, quiero configurar mi estrategia

```gherkin
Feature: Trading Configuration
  
  Scenario: Configure stop-loss and take-profit
    Given I have connected exchanges
    When I set stop_loss to 2% and take_profit to 5%
    And set min_confidence to 0.75
    Then the system should save my configuration
    And apply it to future trades
    
  Scenario: Enforce risk profile limits
    Given I have risk_profile "conservative" (max 1% per trade)
    When I try to set stop_loss to 3%
    Then the system should reject with "Exceeds risk profile limit"
```

### US-03: Como usuario, quiero que mis trades se ejecuten automáticamente

```gherkin
Feature: Auto Execution
  
  Scenario: Execute BUY signal on both exchanges
    Given I have trading enabled
    And I have MEXC and Binance connected
    And min_confidence is 0.70
    When a BUY signal arrives with confidence 0.85
    Then the system should place BUY orders on both exchanges
    And record both executions
    And send me notification
    
  Scenario: Skip signal below confidence threshold
    Given I have trading enabled
    And min_confidence is 0.80
    When a BUY signal arrives with confidence 0.65
    Then the system should NOT execute any orders
    And log "Signal skipped: below confidence threshold"
    
  Scenario: Handle partial failure
    Given I have MEXC and Binance connected
    When a signal triggers execution
    And MEXC succeeds but Binance fails
    Then the system should:
      - Record MEXC execution as success
      - Record Binance execution as failed
      - Alert user about partial execution
      - NOT rollback MEXC order
```

---

# PARTE 8: EDGE CASES & ERROR HANDLING

## 8.1 Edge Cases Matrix

| ID | Escenario | Input | Expected Behavior |
|----|-----------|-------|-------------------|
| EC-01 | Signal durante mantenimiento de exchange | BUY signal | Queue signal, retry when exchange up |
| EC-02 | Balance insuficiente | BUY signal, 0 USDT | Skip exchange, log reason |
| EC-03 | Rate limit hit | Multiple signals rápidos | Queue + exponential backoff |
| EC-04 | API key expirada/revocada | Any order | Mark credentials invalid, alert user |
| EC-05 | Precio se movió significativamente | Market order | Accept slippage up to 1%, else reject |
| EC-06 | Señal duplicada | Same signal_id twice | Idempotent: skip second |
| EC-07 | Usuario desactiva trading mid-execution | Order in flight | Complete in-flight, don't start new |
| EC-08 | Network timeout | Order submission | Retry 3x, then mark FAILED |
| EC-09 | Exchange devuelve error desconocido | Unknown error code | Log full response, mark FAILED, alert |
| EC-10 | Cantidad menor al mínimo del exchange | Small order | Adjust to min or skip with log |

## 8.2 Error Response Contract

```python
# contracts/errors.py
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class ErrorCode(str, Enum):
    # Auth errors (1xxx)
    INVALID_TOKEN = "AUTH_1001"
    EXPIRED_TOKEN = "AUTH_1002"
    
    # Exchange errors (2xxx)
    INVALID_CREDENTIALS = "EXCH_2001"
    INSUFFICIENT_PERMISSIONS = "EXCH_2002"
    RATE_LIMITED = "EXCH_2003"
    EXCHANGE_UNAVAILABLE = "EXCH_2004"
    INSUFFICIENT_BALANCE = "EXCH_2005"
    
    # Trading errors (3xxx)
    BELOW_MINIMUM_ORDER = "TRADE_3001"
    EXCEEDS_RISK_LIMIT = "TRADE_3002"
    TRADING_DISABLED = "TRADE_3003"
    BELOW_CONFIDENCE = "TRADE_3004"
    
    # System errors (5xxx)
    INTERNAL_ERROR = "SYS_5001"
    DATABASE_ERROR = "SYS_5002"

class ErrorResponse(BaseModel):
    """Formato estándar de errores."""
    code: ErrorCode
    message: str
    details: Optional[dict] = None
    request_id: str  # Para debugging
    
    class Config:
        schema_extra = {
            "example": {
                "code": "EXCH_2005",
                "message": "Insufficient USDT balance on MEXC",
                "details": {
                    "required": "100.00",
                    "available": "45.50"
                },
                "request_id": "req-abc123"
            }
        }
```

---

# PARTE 9: STACK TECNOLÓGICO

## 9.1 Decisiones de Arquitectura

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| **Frontend** | React + TypeScript + Vite | Type safety, fast dev |
| **UI Components** | Shadcn/ui + Tailwind | Consistent, customizable |
| **State Management** | Zustand | Simple, performant |
| **Backend** | FastAPI + Python 3.11 | Async, Pydantic native |
| **Database** | PostgreSQL (Supabase) | Managed, Row Level Security |
| **Auth** | Supabase Auth | JWT, social login ready |
| **Secrets** | Supabase Vault o AWS Secrets Manager | Encryption at rest |
| **Queue** | Redis + Celery | Reliable task processing |
| **Monitoring** | Sentry + custom metrics | Error tracking |

## 9.2 Directory Structure

```
signalbridge/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── contracts/          # Zod schemas (mirrors backend)
│   │   ├── hooks/
│   │   ├── pages/
│   │   ├── services/           # API clients
│   │   └── stores/             # Zustand stores
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── users.py
│   │   │   │   ├── exchanges.py
│   │   │   │   ├── trading.py
│   │   │   │   └── signals.py
│   │   │   └── deps.py         # Dependencies (auth, db)
│   │   ├── contracts/          # Pydantic models (SSOT)
│   │   ├── services/
│   │   │   ├── vault.py        # Encryption service
│   │   │   ├── execution.py    # Trade execution
│   │   │   └── signal_processor.py
│   │   ├── adapters/
│   │   │   ├── base.py         # Abstract adapter
│   │   │   ├── mexc.py
│   │   │   └── binance.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── security.py
│   │   └── main.py
│   ├── tests/
│   │   ├── contracts/          # Contract tests
│   │   ├── integration/        # API tests
│   │   └── unit/
│   ├── migrations/
│   └── pyproject.toml
│
├── infra/
│   ├── docker-compose.yml
│   └── terraform/
│
└── docs/
    ├── api/                    # OpenAPI specs
    └── architecture/
```

---

# PARTE 10: ROADMAP DE IMPLEMENTACIÓN

## 10.1 Fases

### Fase 1: Foundation (Semana 1-2)
- [ ] Setup proyecto (monorepo structure)
- [ ] Contracts package (Pydantic models)
- [ ] Database schema + migrations
- [ ] Auth integration (Supabase)
- [ ] Basic API endpoints (users, health)
- [ ] Contract tests

### Fase 2: Exchange Integration (Semana 3-4)
- [ ] Vault service (encryption)
- [ ] Exchange adapter interface
- [ ] MEXC adapter implementation
- [ ] Binance adapter implementation
- [ ] Credential validation flow
- [ ] Integration tests con sandbox/testnet

### Fase 3: Signal Processing (Semana 5-6)
- [ ] Signal webhook receiver
- [ ] Signal processor service
- [ ] Execution engine
- [ ] Order state machine
- [ ] Queue setup (Redis + Celery)
- [ ] End-to-end tests

### Fase 4: Frontend (Semana 7-8)
- [ ] Auth pages (login, register)
- [ ] Dashboard layout
- [ ] Exchange connection UI
- [ ] Trading config UI
- [ ] Signals & executions view
- [ ] Frontend contract validation (Zod)

### Fase 5: Hardening (Semana 9-10)
- [ ] Error handling improvements
- [ ] Retry logic
- [ ] Monitoring & alerting
- [ ] Rate limiting
- [ ] Security audit
- [ ] Load testing

---

# PARTE 11: PROMPTS PARA IMPLEMENTACIÓN

## 11.1 Prompt: Backend Foundation

```markdown
# SPEC-DRIVEN TASK: Backend Foundation

## Contexto
Estamos construyendo SignalBridge, un SaaS de trading automatizado.
Ver spec completo en: TRADING_SAAS_SPEC_v1.md

## Task
Implementar la estructura base del backend FastAPI.

## Inputs
- Spec de contratos: contracts/auth.py, contracts/user.py
- Schema DB: migrations/001_initial_schema.sql

## Output Esperado
- /backend/app/main.py con FastAPI app
- /backend/app/api/v1/users.py con endpoints
- /backend/app/api/deps.py con auth dependency
- /backend/tests/contracts/test_auth.py

## Criterios de Aceptación
1. [ ] `pytest tests/contracts/` pasa
2. [ ] `/docs` muestra OpenAPI spec
3. [ ] Auth middleware valida JWT
4. [ ] Pydantic models match spec exactamente

## Constraints
- Python 3.11+
- Async everywhere
- Type hints obligatorios
- No secrets hardcodeados
```

## 11.2 Prompt: Exchange Adapter

```markdown
# SPEC-DRIVEN TASK: MEXC Exchange Adapter

## Contexto
Implementar adaptador para MEXC siguiendo interface CT-EXCHANGE.

## Inputs
- Interface: contracts/exchange_adapter.py
- MEXC API Docs: https://mexcdevelop.github.io/apidocs/

## Output Esperado
- /backend/app/adapters/mexc.py
- /backend/tests/integration/test_mexc_adapter.py

## Criterios de Aceptación
1. [ ] Implementa todos los métodos de ExchangeAdapter
2. [ ] Maneja errores de API correctamente
3. [ ] Rate limiting implementado
4. [ ] Tests pasan contra testnet
5. [ ] Secrets nunca en logs

## Edge Cases a Manejar
- Rate limit (429)
- Invalid signature
- Insufficient balance
- Network timeout
- Unknown error codes
```

---

# APÉNDICE A: SECURITY CHECKLIST

- [ ] API keys encriptadas con AES-256-GCM
- [ ] Keys nunca en logs (SecretStr)
- [ ] Keys nunca en responses
- [ ] HTTPS only
- [ ] JWT con expiration corto (1h)
- [ ] Refresh tokens rotados
- [ ] Rate limiting por usuario
- [ ] Rate limiting por IP
- [ ] SQL injection prevention (SQLAlchemy)
- [ ] XSS prevention (React escaping)
- [ ] CSRF tokens
- [ ] Row Level Security en DB
- [ ] Audit log de credential usage
- [ ] Validar que keys solo tienen SPOT permission
- [ ] Alertas si credential usage anómalo

---

*Spec v1.0 - SignalBridge Trading Execution Platform*
*Metodología: Spec-Driven Development + AI-Augmented TDD*
*Arquitectura: Contract-First Design*
# SIGNALBRIDGE SPEC v2.0 - PARTE 2
## Continuación: Fases, Prompts, Código de Referencia y Apéndices

---

# PARTE 14: ROADMAP DE IMPLEMENTACIÓN (Continuación)

### 📅 Fase 5: Hardening (Semana 9-10)

**Objetivo**: Production-ready

| Task | Prioridad | Estimado |
|------|-----------|----------|
| Error handling improvements | P0 | 4h |
| Retry logic refinement | P0 | 4h |
| Monitoring setup (Sentry) | P0 | 4h |
| Alerting rules | P0 | 4h |
| Rate limiting | P0 | 4h |
| Security audit | P0 | 8h |
| Load testing | P1 | 4h |
| Documentation | P1 | 4h |
| Runbooks | P1 | 4h |

**Deliverable**: Sistema listo para producción

---

# PARTE 15: PROMPTS PARA IMPLEMENTACIÓN

## 15.1 Prompt: Backend Foundation

```markdown
# SPEC-DRIVEN TASK: Backend Foundation

## Contexto
Estamos construyendo SignalBridge, un SaaS de trading automatizado.
Ver spec completo en: SIGNALBRIDGE_SPEC_v2.md

## Task
Implementar la estructura base del backend FastAPI.

## Output Esperado
- backend/app/main.py con FastAPI app
- backend/app/api/v1/users.py con endpoints
- backend/app/api/deps.py con auth dependency
- backend/app/contracts/ con Pydantic models
- backend/tests/contracts/test_auth.py

## Criterios de Aceptación
1. pytest tests/contracts/ pasa (>90% coverage)
2. /docs muestra OpenAPI spec
3. Auth middleware valida JWT
4. Pydantic models match spec exactamente
5. Async everywhere
```

## 15.2 Prompt: Exchange Adapters

```markdown
# SPEC-DRIVEN TASK: Exchange Adapters

## Contexto
Implementar adapters para Binance y MEXC siguiendo interface CT-EXCHANGE.

## Output Esperado
- backend/app/adapters/base.py (ExchangeAdapter ABC)
- backend/app/adapters/binance_adapter.py
- backend/app/adapters/mexc_adapter.py
- backend/app/adapters/factory.py

## Criterios de Aceptación
1. Ambos adapters implementan ExchangeAdapter interface
2. validate_credentials() detecta permisos correctamente
3. place_market_order() ejecuta trades
4. Error handling robusto
5. Secrets NUNCA en logs
```

## 15.3 Prompt: Signal Processor

```markdown
# SPEC-DRIVEN TASK: Signal Processor

## Task
Implementar el servicio que recibe señales del ML y las procesa.

## Flujo
1. Webhook recibe TradingSignal
2. Validar + guardar en DB
3. Query usuarios elegibles
4. Ejecutar en cada exchange
5. Guardar ExecutionResult
6. Notificar via WebSocket
```

## 15.4 Prompt: Frontend Dashboard

```markdown
# SPEC-DRIVEN TASK: Frontend Dashboard

## Task
Implementar dashboard con React + TypeScript + Zod

## Componentes
- Auth pages (login, register)
- Exchange connection UI
- Trading config UI
- Signals & executions views
- WebSocket integration
```

---

# PARTE 16: CÓDIGO DE REFERENCIA

## 16.1 Enums y Dataclasses (SSOT)

```python
from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Optional, Dict

class TradingAction(IntEnum):
    """Acción de trading - SSOT"""
    SELL = 0
    HOLD = 1
    BUY = 2

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class TimeInForce(Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"

@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    status: Optional[str] = None
    error_message: Optional[str] = None
```

## 16.2 Signature Generation

```python
import hmac
import hashlib
import time
from urllib.parse import urlencode

def generate_signature(api_secret: str, params: dict) -> str:
    params['timestamp'] = int(time.time() * 1000)
    query_string = urlencode(params)
    return hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
```

## 16.3 URLs de Referencia

| Exchange | API Producción | API Testnet |
|----------|----------------|-------------|
| Binance | https://api.binance.com | https://testnet.binance.vision |
| MEXC | https://api.mexc.com | N/A (usar /order/test) |

## 16.4 Unified Trader (CCXT)

```python
import ccxt

class UnifiedTrader:
    SUPPORTED = ['binance', 'mexc']
    
    def __init__(self, exchange: str, api_key: str, secret: str, testnet: bool = False):
        config = {
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }
        
        if exchange == 'binance':
            self.exchange = ccxt.binance(config)
            if testnet:
                self.exchange.set_sandbox_mode(True)
        elif exchange == 'mexc':
            self.exchange = ccxt.mexc(config)
    
    def market_buy(self, symbol: str, amount: float):
        return self.exchange.create_market_buy_order(symbol, amount)
    
    def market_sell(self, symbol: str, amount: float):
        return self.exchange.create_market_sell_order(symbol, amount)
    
    def get_balance(self, asset: str = None):
        balance = self.exchange.fetch_balance()
        if asset:
            return balance[asset]
        return balance
```

---

# APÉNDICE A: CHECKLIST DE LANZAMIENTO

## Seguridad
- [ ] API keys encryption verificada (AES-256-GCM)
- [ ] RLS policies testeadas
- [ ] Rate limiting configurado
- [ ] HTTPS enforced
- [ ] Penetration testing

## Infraestructura
- [ ] Database backups
- [ ] Monitoring/alerting activo
- [ ] CI/CD pipeline
- [ ] Rollback plan

## Testing
- [ ] Unit tests >80% coverage
- [ ] Integration tests
- [ ] E2E tests
- [ ] Load testing

---

# APÉNDICE B: GLOSARIO

| Término | Definición |
|---------|------------|
| **Adapter** | Clase que implementa interface para un exchange |
| **Contract** | Schema que define comunicación entre componentes |
| **SSOT** | Single Source of Truth |
| **Signal** | Predicción del modelo (BUY/HOLD/SELL) |
| **Execution** | Resultado de ejecutar orden en exchange |
| **Vault** | Almacenamiento seguro de API keys |

---

# APÉNDICE C: RECURSOS

| Recurso | URL |
|---------|-----|
| Binance Docs | https://developers.binance.com |
| MEXC Docs | https://mexcdevelop.github.io/apidocs |
| CCXT Docs | https://docs.ccxt.com |
| Binance Testnet | https://testnet.binance.vision |

---

*SIGNALBRIDGE SPEC v2.0 - Part 2*
*Metodología: Spec-Driven Development + AI-Augmented TDD*
*Última actualización: 2026-01-18*

---

# PARTE 17: DETALLES TÉCNICOS DE EXCHANGES (De spot_trading_guide)

## 17.1 Configuración de API Keys

### Binance
1. Ir a: https://www.binance.com/en/my/settings/api-management
2. Crear nueva API Key
3. Habilitar SOLO: ✅ Enable Spot & Margin Trading
4. NUNCA habilitar: ❌ Enable Withdrawals
5. Configurar IP Whitelist (RECOMENDADO)

### MEXC
1. Ir a: https://www.mexc.com/user/openapi
2. Permisos requeridos:
   - ✅ SPOT_ACCOUNT_READ
   - ✅ SPOT_DEAL_WRITE
   - ❌ SPOT_ACCOUNT_TRANSFER (NUNCA)

## 17.2 Endpoints Principales

### Binance Spot API v3
| Endpoint | Método | Signed | Descripción |
|----------|--------|--------|-------------|
| /api/v3/account | GET | ✅ | Info de cuenta |
| /api/v3/order | POST | ✅ | Crear orden |
| /api/v3/order | DELETE | ✅ | Cancelar orden |
| /api/v3/order/test | POST | ✅ | Test order |
| /api/v3/ticker/price | GET | ❌ | Precio actual |

### MEXC Spot API v3
| Endpoint | Método | Signed | Descripción |
|----------|--------|--------|-------------|
| /api/v3/account | GET | ✅ | Info de cuenta |
| /api/v3/order | POST | ✅ | Crear orden |
| /api/v3/order | DELETE | ✅ | Cancelar orden |
| /api/v3/ticker/price | GET | ❌ | Precio actual |

## 17.3 Formato de Órdenes

### Market Order
```json
{
    "symbol": "BTCUSDT",
    "side": "BUY",
    "type": "MARKET",
    "quantity": "0.001"
}
```

### Market Order con Quote
```json
{
    "symbol": "BTCUSDT",
    "side": "BUY",
    "type": "MARKET",
    "quoteOrderQty": "100"
}
```

### Limit Order
```json
{
    "symbol": "BTCUSDT",
    "side": "BUY",
    "type": "LIMIT",
    "timeInForce": "GTC",
    "quantity": "0.001",
    "price": "50000"
}
```

## 17.4 Códigos de Error Comunes

| Exchange | Código | Mensaje | Acción |
|----------|--------|---------|--------|
| Binance | -1013 | MIN_NOTIONAL | Aumentar cantidad |
| Binance | -2010 | Insufficient balance | Skip, log |
| Binance | -1015 | Too many requests | Backoff |
| MEXC | 30002 | Insufficient balance | Skip, log |
| MEXC | 429 | Rate limit | Backoff |

## 17.5 Rate Limits

| Exchange | Límite |
|----------|--------|
| Binance | 1200 req/min |
| MEXC | 500 req/min |

---

# PARTE 18: IMPLEMENTACIÓN DE ADAPTERS (De spot_trading_module.py)

## 18.1 Binance Adapter Nativo

```python
class BinanceSpotAPI:
    BASE_URL_PROD = "https://api.binance.com"
    BASE_URL_TEST = "https://testnet.binance.vision"
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.BASE_URL_TEST if testnet else self.BASE_URL_PROD
        self.session = requests.Session()
        self.session.headers.update({'X-MBX-APIKEY': self.api_key})
    
    def _generate_signature(self, query_string: str) -> str:
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def new_order(self, symbol, side, order_type, quantity=None, 
                  quote_order_qty=None, price=None) -> OrderResult:
        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': order_type.upper()
        }
        if quantity:
            params['quantity'] = quantity
        elif quote_order_qty:
            params['quoteOrderQty'] = quote_order_qty
        if price:
            params['price'] = price
            params['timeInForce'] = 'GTC'
        
        # Add timestamp and signature
        params['timestamp'] = int(time.time() * 1000)
        query = urlencode(params)
        signature = self._generate_signature(query)
        
        response = self.session.post(
            f"{self.base_url}/api/v3/order?{query}&signature={signature}"
        )
        return self._parse_response(response)
```

## 18.2 MEXC Adapter Nativo

```python
class MEXCSpotAPI:
    BASE_URL = "https://api.mexc.com"
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({'X-MEXC-APIKEY': self.api_key})
    
    def _generate_signature(self, query_string: str) -> str:
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def new_order(self, symbol, side, order_type, quantity=None,
                  quote_order_qty=None, price=None) -> OrderResult:
        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': order_type.upper()
        }
        if quantity:
            params['quantity'] = quantity
        elif quote_order_qty:
            params['quoteOrderQty'] = quote_order_qty
        if price:
            params['price'] = price
            params['timeInForce'] = 'GTC'
        
        params['timestamp'] = int(time.time() * 1000)
        query = urlencode(params)
        signature = self._generate_signature(query)
        
        response = self.session.post(
            f"{self.BASE_URL}/api/v3/order",
            data=f"{query}&signature={signature}"
        )
        return self._parse_response(response)
```

## 18.3 Manejo de Errores Robusto

```python
def safe_order(exchange, symbol, side, order_type, amount, price=None):
    """Ejecutar orden con manejo de errores."""
    try:
        if order_type == 'market':
            if side == 'buy':
                order = exchange.create_market_buy_order(symbol, amount)
            else:
                order = exchange.create_market_sell_order(symbol, amount)
        else:
            order = exchange.create_limit_order(symbol, side, amount, price)
        return {'success': True, 'order': order}
        
    except ccxt.InsufficientFunds as e:
        return {'success': False, 'error': 'INSUFFICIENT_BALANCE', 'details': str(e)}
    except ccxt.InvalidOrder as e:
        return {'success': False, 'error': 'INVALID_ORDER', 'details': str(e)}
    except ccxt.NetworkError as e:
        return {'success': False, 'error': 'NETWORK_ERROR', 'details': str(e)}
    except ccxt.ExchangeError as e:
        return {'success': False, 'error': 'EXCHANGE_ERROR', 'details': str(e)}
```

## 18.4 Rate Limiter

```python
import time

class RateLimiter:
    def __init__(self, calls_per_second=10):
        self.calls_per_second = calls_per_second
        self.last_call = 0
        
    def wait(self):
        elapsed = time.time() - self.last_call
        min_interval = 1.0 / self.calls_per_second
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_call = time.time()

# Uso
limiter = RateLimiter(calls_per_second=5)
for symbol in symbols:
    limiter.wait()
    ticker = exchange.fetch_ticker(symbol)
```

---

# PARTE 19: DEPENDENCIAS E INSTALACIÓN

## 19.1 Backend (Python)

```bash
pip install fastapi uvicorn sqlalchemy[asyncio] asyncpg
pip install httpx pydantic cryptography python-jose passlib
pip install celery[redis] structlog tenacity
pip install ccxt  # Para prototipos rápidos
```

## 19.2 Frontend (Node.js)

```bash
npm install react react-dom react-router-dom
npm install @tanstack/react-query zustand zod axios
npm install tailwindcss lucide-react date-fns
```

## 19.3 Exchange Libraries

```bash
# CCXT (unificada - recomendada)
pip install ccxt

# Binance específica
pip install binance-connector  # Oficial
pip install python-binance     # No oficial pero popular

# MEXC específica
pip install pymexc
```

---

*SIGNALBRIDGE SPEC v2.0 COMPLETO*
*Fusión de: TRADING_SAAS_SPEC_v1.md + spot_trading_guide.md + spot_trading_module.py*
*Metodología: Spec-Driven Development + AI-Augmented TDD*
*Arquitectura: Contract-First Design*
*Última actualización: 2026-01-18*
