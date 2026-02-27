"""
SQLAlchemy models for SignalBridge API.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID, JSON, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "sb_users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, nullable=True)
    last_login = Column(DateTime, nullable=True)

    trading_config = relationship("TradingConfig", back_populates="user", uselist=False)
    credentials = relationship("ExchangeCredential", back_populates="user")
    signals = relationship("Signal", back_populates="user")
    executions = relationship("Execution", back_populates="user")


class TradingConfig(Base):
    __tablename__ = "sb_trading_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("sb_users.id"), nullable=False, unique=True)
    trading_enabled = Column(Boolean, default=False, nullable=False)
    default_exchange = Column(String(50), nullable=True)
    max_position_size = Column(Float, default=0.1)
    stop_loss_percent = Column(Float, default=5.0)
    take_profit_percent = Column(Float, default=10.0)
    use_trailing_stop = Column(Boolean, default=False)
    trailing_stop_percent = Column(Float, default=2.0)
    allowed_symbols = Column(ARRAY(String), default=list)
    blocked_symbols = Column(ARRAY(String), default=list)
    max_daily_trades = Column(Integer, default=50)
    max_concurrent_positions = Column(Integer, default=5)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="trading_config")


class ExchangeCredential(Base):
    __tablename__ = "sb_exchange_credentials"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("sb_users.id"), nullable=False, index=True)
    exchange = Column(String(50), nullable=False)
    label = Column(String(100), nullable=False)
    encrypted_api_key = Column(Text, nullable=False)
    encrypted_api_secret = Column(Text, nullable=False)
    encrypted_passphrase = Column(Text, nullable=True)
    key_version = Column(String(50), nullable=False)
    is_testnet = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    is_valid = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, nullable=True)
    last_used = Column(DateTime, nullable=True)
    last_validated = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="credentials")
    audit_logs = relationship("CredentialAuditLog", back_populates="credential")


class CredentialAuditLog(Base):
    __tablename__ = "sb_credential_audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    credential_id = Column(UUID(as_uuid=True), ForeignKey("sb_exchange_credentials.id"), nullable=False, index=True)
    action = Column(String(50), nullable=False)
    actor_id = Column(UUID(as_uuid=True), nullable=False)
    ip_address = Column(String(45), nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    credential = relationship("ExchangeCredential", back_populates="audit_logs")


class Signal(Base):
    __tablename__ = "sb_signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("sb_users.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(Integer, nullable=False)
    price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    source = Column(String(50), default="api")
    signal_metadata = Column("signal_metadata", JSON, default=dict)

    @property
    def metadata_dict(self):
        return self.signal_metadata or {}

    def __init__(self, **kwargs):
        if "metadata" in kwargs:
            kwargs["signal_metadata"] = kwargs.pop("metadata")
        super().__init__(**kwargs)
    is_processed = Column(Boolean, default=False, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    execution_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="signals")


class Execution(Base):
    __tablename__ = "sb_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("sb_users.id"), nullable=False, index=True)
    signal_id = Column(UUID(as_uuid=True), nullable=True)
    exchange = Column(String(50), nullable=False)
    credential_id = Column(UUID(as_uuid=True), ForeignKey("sb_exchange_credentials.id"), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    order_type = Column(String(30), nullable=False, default="market")
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    exchange_order_id = Column(String(100), nullable=True)
    filled_quantity = Column(Float, default=0.0)
    average_price = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    commission_asset = Column(String(20), nullable=True)
    executed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    raw_response = Column(JSON, nullable=True)
    execution_metadata = Column("execution_metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="executions")

    def __init__(self, **kwargs):
        if "metadata" in kwargs:
            kwargs["execution_metadata"] = kwargs.pop("metadata")
        super().__init__(**kwargs)
