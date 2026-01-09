# USD/COP Trading System - Refactoring with SOLID & Design Patterns

## Version 3.0.0 - December 17, 2025

This document describes the complete refactoring of the `src/` codebase applying **SOLID principles** and **Design Patterns** to create a professional, maintainable, and testable architecture.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [SOLID Principles Applied](#solid-principles-applied)
3. [Design Patterns Implemented](#design-patterns-implemented)
4. [Directory Structure](#directory-structure)
5. [Usage Examples](#usage-examples)
6. [Migration Guide](#migration-guide)
7. [Testing](#testing)

---

## Architecture Overview

The refactored architecture separates concerns into distinct layers:

```
src/
├── core/                          # Business logic
│   ├── interfaces/                # Abstract interfaces (DIP)
│   ├── factories/                 # Factory Pattern
│   ├── calculators/               # Feature calculators (SRP)
│   ├── normalizers/               # Strategy Pattern
│   ├── builders/                  # Builder Pattern
│   └── services/                  # Orchestration layer
├── shared/                        # Cross-cutting concerns
│   ├── config_loader.py           # Configuration management
│   ├── config_loader_adapter.py   # Adapter Pattern
│   └── exceptions.py              # Custom exceptions
```

---

## SOLID Principles Applied

### 1. Single Responsibility Principle (SRP)

Each class has ONE reason to change:

- **RSICalculator**: Only calculates RSI
- **ATRCalculator**: Only calculates ATR
- **ZScoreNormalizer**: Only handles z-score normalization
- **ObservationBuilder**: Only builds observation vectors
- **FeatureBuilderRefactored**: Only orchestrates (delegates actual work)

### 2. Open/Closed Principle (OCP)

Open for extension, closed for modification:

```python
# Add new calculator WITHOUT modifying existing code
class CustomCalculator(BaseFeatureCalculator):
    def compute(self, data: pd.DataFrame) -> pd.Series:
        # Custom logic here
        pass

# Register with factory
FeatureCalculatorFactory.register('custom', CustomCalculator)

# Use it
calc = FeatureCalculatorFactory.create('custom', param=value)
```

### 3. Liskov Substitution Principle (LSP)

All calculators implement `IFeatureCalculator` and are interchangeable:

```python
def process_feature(calculator: IFeatureCalculator, data: pd.DataFrame):
    return calculator.calculate(data)

# Any calculator works
process_feature(RSICalculator(period=9), df)
process_feature(ATRCalculator(period=10), df)
process_feature(ReturnsCalculator(periods=1), df)
```

### 4. Interface Segregation Principle (ISP)

Small, focused interfaces instead of one large interface:

- `IFeatureCalculator`: Only calculation methods
- `INormalizer`: Only normalization methods
- `IObservationBuilder`: Only building methods
- `IConfigLoader`: Only configuration methods

### 5. Dependency Inversion Principle (DIP)

Depend on abstractions, not concretions:

```python
class FeatureBuilderRefactored:
    def __init__(self, config: IConfigLoader):  # Depends on interface
        self._config = config
        self._calculators: Dict[str, IFeatureCalculator] = {}
        self._obs_builder: IObservationBuilder = ObservationBuilder(...)
```

---

## Design Patterns Implemented

### 1. Factory Pattern

**Purpose**: Create objects without specifying exact classes

**Location**: `src/core/factories/`

**Example**:
```python
from src.core.factories import FeatureCalculatorFactory

# Create calculators by type
rsi_calc = FeatureCalculatorFactory.create('rsi', period=9)
atr_calc = FeatureCalculatorFactory.create('atr', period=10, as_percentage=True)
ret_calc = FeatureCalculatorFactory.create('returns', periods=12, name='log_ret_1h')

# Register custom calculators
FeatureCalculatorFactory.register('my_indicator', MyIndicatorCalculator)
```

### 2. Strategy Pattern

**Purpose**: Define family of algorithms, make them interchangeable

**Location**: `src/core/normalizers/`

**Example**:
```python
from src.core.normalizers import ZScoreNormalizer, ClipNormalizer, CompositeNormalizer

# Different normalization strategies
zscore = ZScoreNormalizer(mean=100.0, std=10.0)
clip = ClipNormalizer(min_val=-4.0, max_val=4.0)

# Combine strategies
composite = CompositeNormalizer(normalizers=[zscore, clip])

# Use any strategy
normalized = zscore.normalize(value)
normalized = composite.normalize(value)  # z-score then clip
```

### 3. Builder Pattern

**Purpose**: Construct complex objects step by step with fluent interface

**Location**: `src/core/builders/observation_builder.py`

**Example**:
```python
from src.core.builders import ObservationBuilder

builder = ObservationBuilder(
    feature_order=['log_ret_5m', 'rsi_9', ...],
    obs_dim=15
)

# Fluent interface
obs = (builder
       .with_features(features_dict)
       .with_position(0.5)
       .with_time_normalized(30, 60)
       .build())
```

### 4. Template Method Pattern

**Purpose**: Define algorithm skeleton, let subclasses override steps

**Location**: `src/core/calculators/base_calculator.py`

**Example**:
```python
class BaseFeatureCalculator(IFeatureCalculator):
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        # Template method defines algorithm structure
        self.validate_input(data)      # Step 1
        result = self.compute(data)    # Step 2 (abstract - must override)
        result = self._normalize(result)  # Step 3
        result = self._clip(result)    # Step 4
        return result

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        # Subclasses implement this
        pass

class RSICalculator(BaseFeatureCalculator):
    def compute(self, data: pd.DataFrame) -> pd.Series:
        # Only implement calculation logic
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
```

### 5. Adapter Pattern

**Purpose**: Convert interface of class to another interface clients expect

**Location**: `src/shared/config_loader_adapter.py`

**Example**:
```python
# Original ConfigLoader doesn't implement IConfigLoader interface
# Adapter makes it compatible

config = ConfigLoaderAdapter()  # Adapts ConfigLoader
builder = FeatureBuilderRefactored(config=config)  # Works with interface
```

---

## Directory Structure

```
src/
├── core/
│   ├── interfaces/                    # SOLID: Dependency Inversion
│   │   ├── __init__.py
│   │   ├── feature_calculator.py      # IFeatureCalculator
│   │   ├── normalizer.py              # INormalizer
│   │   ├── observation_builder.py     # IObservationBuilder
│   │   └── config_loader.py           # IConfigLoader
│   │
│   ├── factories/                     # Factory Pattern
│   │   ├── __init__.py
│   │   ├── feature_calculator_factory.py
│   │   └── normalizer_factory.py
│   │
│   ├── calculators/                   # Template Method + SRP
│   │   ├── __init__.py
│   │   ├── base_calculator.py         # Template Method
│   │   ├── rsi_calculator.py          # RSI implementation
│   │   ├── atr_calculator.py          # ATR implementation
│   │   ├── adx_calculator.py          # ADX implementation
│   │   ├── returns_calculator.py      # Log returns
│   │   ├── macro_zscore_calculator.py # Macro z-scores
│   │   └── macro_change_calculator.py # Macro changes
│   │
│   ├── normalizers/                   # Strategy Pattern
│   │   ├── __init__.py
│   │   ├── zscore_normalizer.py       # Z-score strategy
│   │   ├── clip_normalizer.py         # Clipping strategy
│   │   ├── noop_normalizer.py         # No-op strategy
│   │   └── composite_normalizer.py    # Composite strategy
│   │
│   ├── builders/                      # Builder Pattern
│   │   ├── __init__.py
│   │   └── observation_builder.py     # Fluent interface
│   │
│   └── services/
│       ├── __init__.py
│       ├── feature_builder.py         # Original (backward compat)
│       └── feature_builder_refactored.py  # Refactored with DI
│
└── shared/
    ├── __init__.py
    ├── config_loader.py               # Original config loader
    ├── config_loader_adapter.py       # Adapter Pattern
    └── exceptions.py                  # Custom exceptions
```

---

## Usage Examples

### Basic Usage (Backward Compatible)

```python
# Old way still works
from src import FeatureBuilder

builder = FeatureBuilder()
features_df = builder.build_batch(ohlcv_df, macro_df)
obs = builder.build_observation(features_dict, position=0.0, bar_number=30)
```

### New Refactored Usage

```python
from src import FeatureBuilderRefactored, ConfigLoaderAdapter

# With dependency injection
config = ConfigLoaderAdapter('config/')
builder = FeatureBuilderRefactored(config=config)

# Same API
features_df = builder.build_batch(ohlcv_df, macro_df)
obs = builder.build_observation(features_dict, position=0.0, bar_number=30)
```

### Using Individual Components

```python
from src.core.calculators import RSICalculator
from src.core.normalizers import ZScoreNormalizer, ClipNormalizer, CompositeNormalizer
from src.core.builders import ObservationBuilder

# 1. Create calculator with normalization
normalizer = CompositeNormalizer(normalizers=[
    ZScoreNormalizer(mean=50.0, std=10.0),
    ClipNormalizer(min_val=-4.0, max_val=4.0)
])

rsi_calc = RSICalculator(period=9, normalizer=normalizer)
rsi_values = rsi_calc.calculate(df)

# 2. Build observations
obs_builder = ObservationBuilder(
    feature_order=['log_ret_5m', 'rsi_9', 'atr_pct', ...],
    obs_dim=15
)

obs = (obs_builder
       .with_features(features_dict)
       .with_position(0.5)
       .with_time_normalized(30, 60)
       .build())
```

### Creating Custom Calculators

```python
from src.core.calculators import BaseFeatureCalculator
from src.core.factories import FeatureCalculatorFactory

class BBandsCalculator(BaseFeatureCalculator):
    """Custom Bollinger Bands calculator"""

    def __init__(self, period=20, std_dev=2):
        super().__init__(
            name=f'bbands_{period}',
            dependencies=['close']
        )
        self.period = period
        self.std_dev = std_dev

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        sma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = sma + (std * self.std_dev)
        lower = sma - (std * self.std_dev)
        # Return position within bands
        return (close - lower) / (upper - lower + 1e-10)

# Register with factory
FeatureCalculatorFactory.register('bbands', BBandsCalculator)

# Use it
bbands = FeatureCalculatorFactory.create('bbands', period=20, std_dev=2)
values = bbands.calculate(df)
```

---

## Migration Guide

### Step 1: Install (No changes needed)

The refactored code is **100% backward compatible**. Existing code continues to work.

### Step 2: Gradual Migration (Optional)

```python
# OLD (still works)
from src import FeatureBuilder
builder = FeatureBuilder()

# NEW (recommended for new code)
from src import FeatureBuilderRefactored
builder = FeatureBuilderRefactored()
```

### Step 3: Leverage New Features

```python
# Access individual calculators
calculators = builder.get_calculators()
rsi_calc = calculators['rsi_9']

# Create custom calculators
from src.core.factories import FeatureCalculatorFactory
custom_calc = FeatureCalculatorFactory.create('custom', ...)

# Use different normalization strategies
from src.core.normalizers import ZScoreNormalizer
normalizer = ZScoreNormalizer(mean=0, std=1)
```

---

## Testing

### Unit Tests for Calculators

```python
import pytest
from src.core.calculators import RSICalculator

def test_rsi_calculator():
    # Create test data
    df = pd.DataFrame({
        'close': [100, 102, 101, 103, 105, 104, 106, 108]
    })

    # Test calculator
    calc = RSICalculator(period=9)
    rsi = calc.calculate(df)

    # Verify
    assert not rsi.isna().all()
    assert (rsi >= 0).all() and (rsi <= 100).all()
```

### Integration Tests

```python
def test_feature_builder_refactored():
    from src import FeatureBuilderRefactored

    builder = FeatureBuilderRefactored()

    # Test batch processing
    features_df = builder.build_batch(ohlcv_df, macro_df)
    assert len(features_df.columns) >= 13

    # Test observation building
    features_dict = features_df.iloc[-1].to_dict()
    obs = builder.build_observation(features_dict, 0.5, 30)
    assert obs.shape == (15,)
```

### Mocking for Tests (Dependency Injection)

```python
from unittest.mock import Mock

def test_with_mock_config():
    # Create mock config
    mock_config = Mock(spec=IConfigLoader)
    mock_config.get_feature_order.return_value = ['log_ret_5m', 'rsi_9']
    mock_config.get_obs_dim.return_value = 15

    # Inject mock
    builder = FeatureBuilderRefactored(config=mock_config)

    # Test with mock
    assert builder.feature_order == ['log_ret_5m', 'rsi_9']
```

---

## Benefits

1. **Maintainability**: Each class has single responsibility
2. **Testability**: Dependencies can be mocked/injected
3. **Extensibility**: New calculators via Factory, no code modification
4. **Flexibility**: Swap normalization strategies at runtime
5. **Readability**: Clear separation of concerns
6. **Reusability**: Individual components can be used standalone
7. **Type Safety**: Interfaces define clear contracts

---

## Summary

This refactoring transforms the codebase from a monolithic design to a professional, enterprise-grade architecture following industry best practices:

- **SOLID principles** for maintainability
- **Design Patterns** for flexibility and extensibility
- **Dependency Injection** for testability
- **100% backward compatibility** for smooth migration

All existing functionality is preserved while enabling future enhancements through clean abstractions.

---

**Author**: Pedro @ Lean Tech Solutions
**Version**: 3.0.0
**Date**: 2025-12-17
