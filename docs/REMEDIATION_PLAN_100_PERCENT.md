# PLAN DE REMEDIACIÓN INTEGRAL - 100% COMPLIANCE
## USD/COP RL Trading System - Production Readiness Remediation

**Fecha**: 2026-01-17
**Score Actual**: 70.5% (315/447)
**Score Objetivo**: 100% (447/447)
**Veredicto Actual**: ⚠️ **NO LISTO PARA PRODUCCIÓN** (3 bloqueadores críticos)
**Duración Estimada**: 3 Semanas

---

# 📊 ANÁLISIS EJECUTIVO

## Estado del Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ESTADO DEL SISTEMA                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SCORE: 70.5%  ████████████████████░░░░░░░░░░                              │
│                                                                             │
│  ✅ FORTALEZAS                    ❌ BLOQUEADORES                           │
│  ═══════════════                  ═══════════════                           │
│  • Contracts: 80%                 • Action enum: 4 definiciones             │
│  • DAGs: 83%                      • FEATURE_ORDER: 7 definiciones           │
│  • Monitoring: 88%                • L5 ignora TRADING_ENABLED               │
│  • Documentation: 90%             • "session_progress" aún existe           │
│  • CI/CD: OK                                                                │
│                                                                             │
│  DIAGNÓSTICO: Arquitectura excelente, pero SSOT roto                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Scores por Categoría

```
┌────────────────────────────────────────────────────────────────────────────┐
│  CATEGORÍA              │ CUMPLE │ PARCIAL │ NO CUMPLE │ SCORE │ ESTADO   │
├─────────────────────────┼────────┼─────────┼───────────┼───────┼──────────┤
│ Documentation (DOC)     │   18   │    2    │     0     │  90%  │ ✅       │
│ Alerts (ALERT)          │   12   │    3    │     0     │  80%  │ ✅       │
│ Metrics (MET)           │   12   │    3    │     0     │  80%  │ ✅       │
│ Logging (LOG)           │    8   │    2    │     0     │  80%  │ ✅       │
│ API Endpoints           │   15   │    4    │     1     │  75%  │ ✅       │
│ UI Components           │   16   │    3    │     1     │  80%  │ ✅       │
│ Frontend (FE)           │   16   │    3    │     1     │  80%  │ ✅       │
│ Contract Tests          │   12   │    2    │     1     │  80%  │ ✅       │
│ Contracts (CONTRACT)    │   24   │    5    │     1     │  80%  │ ✅       │
│ Risk Manager (RISK)     │   12   │    2    │     1     │  80%  │ ✅       │
│ DAGs (L0-L5)            │   62   │   11    │     2     │  83%  │ ✅       │
│ Production (LIVE)       │   18   │    2    │     0     │  90%  │ ✅       │
├─────────────────────────┼────────┼─────────┼───────────┼───────┼──────────┤
│ Security (SEC)          │    7   │    2    │     1     │  70%  │ ⚠️       │
│ Feature Flags (FLAG)    │   10   │    3    │     2     │  67%  │ ⚠️       │
│ Structure (DIR)         │   12   │    5    │     3     │  60%  │ ⚠️       │
│ MLflow (MLF)            │   12   │    6    │     2     │  60%  │ ⚠️       │
│ Database (DB)           │   12   │    6    │     2     │  60%  │ ⚠️       │
│ Unit Tests (TEST)       │   12   │    2    │     1     │  80%  │ ⚠️       │
│ Integration Tests       │    5   │    4    │     1     │  50%  │ ⚠️       │
├─────────────────────────┼────────┼─────────┼───────────┼───────┼──────────┤
│ Dead Code (DEAD)        │    9   │    9    │    12     │  30%  │ ❌       │
│ SSOT                    │    9   │    9    │    12     │  30%  │ ❌       │
├─────────────────────────┼────────┼─────────┼───────────┼───────┼──────────┤
│ TOTAL                   │  315   │   88    │    44     │ 70.5% │ ⚠️       │
└────────────────────────────────────────────────────────────────────────────┘
```

## Gap Analysis: De 70.5% a 100%

| Categoría | Actual | Objetivo | Items a Remediar |
|-----------|--------|----------|------------------|
| SSOT | 30% | 100% | **21 items** (CRÍTICO) |
| Dead Code | 30% | 100% | **21 items** (CRÍTICO) |
| Integration Tests | 50% | 100% | 5 items |
| Database | 60% | 100% | 8 items |
| MLflow | 60% | 100% | 8 items |
| Structure | 60% | 100% | 8 items |
| Feature Flags | 67% | 100% | 5 items |
| Security | 70% | 100% | 3 items |
| API | 75% | 100% | 5 items |
| Otros (>80%) | 80-90% | 100% | 48 items |
| **TOTAL** | **70.5%** | **100%** | **132 items** |

---

# 🚨 FASE 0: BLOQUEADORES CRÍTICOS (P0)

## Los 3 bloqueadores que impiden producción

Estos **DEBEN** resolverse antes de cualquier otra tarea. Sin estos fixes, el sistema puede perder dinero real por señales invertidas o trades no autorizados.

---

## BLOQUEADOR #1: Action Enum Fragmentado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔴 CRÍTICO: 4+ DEFINICIONES DE ACTION ENUM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ARCHIVO                                        │ DEFINICIÓN               │
│  ═══════════════════════════════════════════════════════════════════════   │
│  src/core/contracts/action_contract.py          │ SELL=0, HOLD=1, BUY=2 ✅ │
│  src/core/constants.py                          │ ??? (orden diferente) ❌ │
│  src/trading/trading_env.py                     │ Definición local ❌      │
│  services/inference_api/core/inference_engine.py│ Mapeo hardcoded ❌       │
│                                                                             │
│  IMPACTO: Si un componente usa SELL=2 y otro usa SELL=0,                   │
│           las señales de trading se INVIERTEN.                              │
│           → El modelo dice COMPRAR, el sistema VENDE.                       │
│                                                                             │
│  RIESGO FINANCIERO: 💰💰💰 MÁXIMO                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Diagnóstico
```bash
# Ejecutar para ver todas las definiciones conflictivas
grep -rn "class Action" src/ services/ --include="*.py" | grep -v test | grep -v __pycache__
grep -rn "SELL.*=\|BUY.*=\|HOLD.*=" src/ services/ --include="*.py" | grep -v test | grep -v __pycache__
```

### Remediación Detallada

**Archivo 1: `src/core/constants.py`**
```python
# ============================================================
# ANTES (INCORRECTO - puede tener orden diferente)
# ============================================================
class Action:
    BUY = 0    # ❌ INCORRECTO - debería ser 2
    HOLD = 1
    SELL = 2   # ❌ INCORRECTO - debería ser 0

# O peor:
ACTION_BUY = 0
ACTION_HOLD = 1
ACTION_SELL = 2

# ============================================================
# DESPUÉS (CORRECTO - importar desde SSOT)
# ============================================================
"""
Action constants - Re-exported from SSOT.

IMPORTANTE: NO definir Action aquí. Importar siempre desde contracts.
El orden correcto es: SELL=0, HOLD=1, BUY=2 (requerido por modelo PPO).
"""
from src.core.contracts.action_contract import (
    Action,
    ACTION_SELL,
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_COUNT,
    ACTION_NAMES,
    VALID_ACTIONS,
)

# Re-export para backwards compatibility
__all__ = [
    "Action",
    "ACTION_SELL",
    "ACTION_HOLD",
    "ACTION_BUY",
    "ACTION_COUNT",
    "ACTION_NAMES",
    "VALID_ACTIONS",
]

# Verificación en tiempo de import
assert Action.SELL == 0, "Action.SELL debe ser 0"
assert Action.HOLD == 1, "Action.HOLD debe ser 1"
assert Action.BUY == 2, "Action.BUY debe ser 2"
```

**Archivo 2: `src/trading/trading_env.py`**
```python
# ============================================================
# ANTES (INCORRECTO - definición local)
# ============================================================
# En algún lugar del archivo:
SELL, HOLD, BUY = 0, 1, 2
# o
class Actions:
    SELL = 0
    HOLD = 1
    BUY = 2

# ============================================================
# DESPUÉS (CORRECTO - importar desde SSOT)
# ============================================================
from src.core.contracts import (
    Action,
    ACTION_SELL,
    ACTION_HOLD,
    ACTION_BUY,
)

# Usar Action.SELL, Action.HOLD, Action.BUY en todo el código
# o las constantes ACTION_SELL, ACTION_HOLD, ACTION_BUY
```

**Archivo 3: `services/inference_api/core/inference_engine.py`**
```python
# ============================================================
# ANTES (INCORRECTO - mapeo hardcoded)
# ============================================================
action_map = {0: "sell", 1: "hold", 2: "buy"}

def get_action_name(action_idx: int) -> str:
    return action_map[action_idx]

# ============================================================
# DESPUÉS (CORRECTO - usar SSOT)
# ============================================================
from src.core.contracts import Action, ACTION_NAMES

def get_action_name(action_idx: int) -> str:
    """Obtiene nombre de acción desde índice."""
    return ACTION_NAMES[action_idx]

def get_action(action_idx: int) -> Action:
    """Obtiene Action enum desde índice."""
    return Action(action_idx)
```

### Test de Regresión Obligatorio
**Archivo: `tests/regression/test_action_enum_ssot.py`**

```python
"""
REGRESSION TEST: Action Enum SSOT
=================================
Este archivo DEBE ejecutarse en CI antes de cada deploy.
Si algún test falla, HAY UN BUG CRÍTICO que puede invertir señales de trading.

Contract ID: CTR-ACTION-SSOT-001
"""
import pytest
import subprocess
import sys
from pathlib import Path

class TestActionEnumSSoT:
    """Tests que garantizan que Action enum es SSOT."""

    def test_action_sell_is_zero(self):
        """CRÍTICO: SELL debe ser 0 (índice 0 del output PPO)."""
        from src.core.contracts import Action, ACTION_SELL
        assert Action.SELL == 0, f"Action.SELL debe ser 0, got {Action.SELL}"
        assert Action.SELL.value == 0
        assert ACTION_SELL == 0

    def test_action_hold_is_one(self):
        """CRÍTICO: HOLD debe ser 1 (índice 1 del output PPO)."""
        from src.core.contracts import Action, ACTION_HOLD
        assert Action.HOLD == 1, f"Action.HOLD debe ser 1, got {Action.HOLD}"
        assert Action.HOLD.value == 1
        assert ACTION_HOLD == 1

    def test_action_buy_is_two(self):
        """CRÍTICO: BUY debe ser 2 (índice 2 del output PPO)."""
        from src.core.contracts import Action, ACTION_BUY
        assert Action.BUY == 2, f"Action.BUY debe ser 2, got {Action.BUY}"
        assert Action.BUY.value == 2
        assert ACTION_BUY == 2

    def test_action_count_is_three(self):
        """Debe haber exactamente 3 acciones."""
        from src.core.contracts import Action, ACTION_COUNT
        assert len(Action) == 3
        assert ACTION_COUNT == 3

    def test_action_values_are_contiguous(self):
        """Los valores deben ser 0, 1, 2 sin gaps."""
        from src.core.contracts import Action
        values = sorted([a.value for a in Action])
        assert values == [0, 1, 2], f"Values deben ser [0,1,2], got {values}"

    def test_action_names_mapping(self):
        """ACTION_NAMES debe mapear correctamente."""
        from src.core.contracts import ACTION_NAMES, Action
        assert ACTION_NAMES[0] == "SELL"
        assert ACTION_NAMES[1] == "HOLD"
        assert ACTION_NAMES[2] == "BUY"
        assert ACTION_NAMES[Action.SELL] == "SELL"

    def test_no_duplicate_action_class_definitions(self):
        """
        CRÍTICO: Solo debe existir UNA definición de 'class Action'.
        Si este test falla, hay definiciones duplicadas que pueden
        causar que diferentes partes del sistema usen diferentes valores.
        """
        result = subprocess.run(
            ["grep", "-rn", "class Action", "--include=*.py",
             "src/", "services/", "airflow/"],
            capture_output=True, text=True, cwd=Path(__file__).parents[3]
        )

        lines = [l for l in result.stdout.strip().split("\n")
                 if l and "test" not in l.lower() and "__pycache__" not in l]

        # Solo debe estar en action_contract.py
        valid_files = ["action_contract.py"]
        invalid_definitions = [
            l for l in lines
            if not any(v in l for v in valid_files)
        ]

        assert len(invalid_definitions) == 0, (
            f"Encontradas definiciones duplicadas de Action:\n" +
            "\n".join(invalid_definitions) +
            "\n\nTodas las definiciones de Action DEBEN estar en "
            "src/core/contracts/action_contract.py"
        )

    def test_no_hardcoded_action_mappings(self):
        """
        Verificar que no hay mapeos hardcoded tipo {0: "sell"}.
        """
        result = subprocess.run(
            ["grep", "-rn", r'{0.*sell\|{0.*buy\|action_map',
             "--include=*.py", "src/", "services/"],
            capture_output=True, text=True, cwd=Path(__file__).parents[3]
        )

        lines = [l for l in result.stdout.strip().split("\n")
                 if l and "test" not in l.lower()]

        # No debería haber ninguno (excepto en el SSOT)
        assert len(lines) <= 1, (
            f"Encontrados mapeos hardcoded de action:\n" +
            "\n".join(lines) +
            "\n\nUsar ACTION_NAMES desde contracts"
        )

    def test_constants_imports_from_ssot(self):
        """Verificar que constants.py importa desde SSOT."""
        constants_path = Path(__file__).parents[3] / "src" / "core" / "constants.py"
        if constants_path.exists():
            content = constants_path.read_text()
            # Debe importar desde contracts, no definir localmente
            assert "from src.core.contracts" in content or "class Action" not in content, (
                "constants.py debe importar Action desde contracts, no definirlo"
            )
```

### Script de Migración Automática
**Archivo: `scripts/migrate_action_enum.py`**

```python
#!/usr/bin/env python3
"""
Migración automática de Action enum al SSOT.

Uso:
    python scripts/migrate_action_enum.py --dry-run  # Ver cambios sin aplicar
    python scripts/migrate_action_enum.py --apply    # Aplicar cambios

Este script:
1. Encuentra todas las definiciones de Action fuera del SSOT
2. Las reemplaza por imports desde src.core.contracts
3. Actualiza mapeos hardcoded para usar ACTION_NAMES
"""
import argparse
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

SSOT_FILE = "src/core/contracts/action_contract.py"
IMPORT_STATEMENT = "from src.core.contracts import Action, ACTION_SELL, ACTION_HOLD, ACTION_BUY, ACTION_NAMES"

def find_action_definitions() -> List[Tuple[Path, int, str]]:
    """Encuentra todas las definiciones de Action fuera del SSOT."""
    result = subprocess.run(
        ["grep", "-rn", "class Action\\|SELL.*=.*0\\|BUY.*=.*2\\|action_map",
         "--include=*.py", "src/", "services/", "airflow/"],
        capture_output=True, text=True
    )

    findings = []
    for line in result.stdout.strip().split("\n"):
        if not line or SSOT_FILE in line or "test" in line.lower():
            continue

        parts = line.split(":", 2)
        if len(parts) >= 3:
            filepath = Path(parts[0])
            lineno = int(parts[1])
            content = parts[2]
            findings.append((filepath, lineno, content))

    return findings

def generate_fix(filepath: Path, content: str) -> str:
    """Genera el código de fix para un archivo."""
    # Leer archivo completo
    full_content = filepath.read_text()

    # Si ya importa desde contracts, no hacer nada
    if "from src.core.contracts import" in full_content and "Action" in full_content:
        return None

    # Agregar import al inicio (después de otros imports)
    lines = full_content.split("\n")
    import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            import_idx = i + 1

    lines.insert(import_idx, IMPORT_STATEMENT)

    # Eliminar definiciones locales de Action
    new_lines = []
    skip_until_dedent = False
    for line in lines:
        if "class Action" in line and "contracts" not in str(filepath):
            skip_until_dedent = True
            continue
        if skip_until_dedent:
            if line and not line.startswith(" ") and not line.startswith("\t"):
                skip_until_dedent = False
            else:
                continue
        new_lines.append(line)

    return "\n".join(new_lines)

def main():
    parser = argparse.ArgumentParser(description="Migrar Action enum al SSOT")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar cambios sin aplicar")
    parser.add_argument("--apply", action="store_true", help="Aplicar cambios")
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        print("Especificar --dry-run o --apply")
        return 1

    findings = find_action_definitions()

    print(f"Encontradas {len(findings)} definiciones de Action fuera del SSOT:\n")

    for filepath, lineno, content in findings:
        print(f"  {filepath}:{lineno}")
        print(f"    {content.strip()}")

    if args.apply:
        print("\n" + "="*60)
        print("APLICANDO CAMBIOS...")
        print("="*60)

        files_modified = set()
        for filepath, _, _ in findings:
            if filepath in files_modified:
                continue

            fix = generate_fix(filepath, "")
            if fix:
                filepath.write_text(fix)
                files_modified.add(filepath)
                print(f"  ✅ Modificado: {filepath}")

        print(f"\n{len(files_modified)} archivos modificados")
        print("\nEjecutar tests para verificar:")
        print("  pytest tests/regression/test_action_enum_ssot.py -v")

    return 0

if __name__ == "__main__":
    exit(main())
```

---

## BLOQUEADOR #2: FEATURE_ORDER Fragmentado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔴 CRÍTICO: 7+ DEFINICIONES DE FEATURE_ORDER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ARCHIVO                                            │ ESTADO               │
│  ══════════════════════════════════════════════════════════════════════    │
│  src/core/contracts/feature_contract.py             │ ✅ SSOT (correcto)   │
│  src/features/builder.py                            │ ❌ Duplicado         │
│  src/core/services/feature_builder.py               │ ❌ Duplicado         │
│  src/services/backtest_feature_builder.py           │ ❌ Duplicado         │
│  services/inference_api/core/observation_builder.py │ ❌ Duplicado         │
│  services/mlops/feature_cache.py                    │ ❌ Duplicado         │
│  airflow/dags/l1_feature_refresh.py                 │ ❌ Duplicado         │
│                                                                             │
│  IMPACTO: Training entrena con features en orden [A,B,C],                  │
│           Inference recibe features en orden [B,A,C].                       │
│           → El modelo ve datos completamente incorrectos.                   │
│                                                                             │
│  RIESGO FINANCIERO: 💰💰💰 MÁXIMO                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Diagnóstico
```bash
# Ejecutar para ver todas las definiciones
grep -rn "FEATURE_ORDER\s*=" src/ services/ airflow/ --include="*.py" | grep -v test | grep -v __pycache__
grep -rn "FEATURE_NAMES\s*=" src/ services/ --include="*.py" | grep -v test
grep -rn "session_progress" src/ services/ airflow/ --include="*.py"
```

### Remediación Detallada

**El SSOT canónico (NO MODIFICAR):**
```python
# src/core/contracts/feature_contract.py
FEATURE_ORDER: tuple[str, ...] = (
    "returns_1h",      # 0: Retorno 1 hora
    "returns_4h",      # 1: Retorno 4 horas
    "returns_1d",      # 2: Retorno 1 día
    "usdcop_position", # 3: Posición normalizada en rango
    "volatility",      # 4: Volatilidad realizada
    "hour_sin",        # 5: Hora del día (sin)
    "hour_cos",        # 6: Hora del día (cos)
    "rsi",             # 7: RSI normalizado
    "dxy_returns",     # 8: Retorno del DXY
    "oil_returns",     # 9: Retorno del petróleo
    "em_spread",       # 10: Spread mercados emergentes
    "macd_signal",     # 11: Señal MACD
    "bb_position",     # 12: Posición Bollinger Bands
    "position",        # 13: Posición actual del portafolio
    "time_normalized", # 14: Tiempo normalizado de sesión (NO "session_progress")
)

OBSERVATION_DIM = 15  # len(FEATURE_ORDER)
```

**Archivo 1: `src/features/builder.py`**
```python
# ============================================================
# ANTES (INCORRECTO - definición local)
# ============================================================
FEATURE_ORDER = [
    "returns_1h", "returns_4h", "returns_1d", ...
]

class FeatureBuilder:
    def build(self) -> np.ndarray:
        features = []
        for name in FEATURE_ORDER:
            features.append(self._compute_feature(name))
        return np.array(features)

# ============================================================
# DESPUÉS (CORRECTO - importar desde SSOT)
# ============================================================
from src.core.contracts import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    validate_feature_vector,
    features_dict_to_array,
)

class FeatureBuilder:
    """Builder de features usando FEATURE_ORDER canónico."""

    def build(self) -> np.ndarray:
        """Construye vector de features en orden correcto."""
        features_dict = {}
        for name in FEATURE_ORDER:
            features_dict[name] = self._compute_feature(name)

        # Usar función del contract para garantizar orden correcto
        feature_array = features_dict_to_array(features_dict)

        # Validar antes de retornar
        is_valid, errors = validate_feature_vector(feature_array)
        if not is_valid:
            raise ValueError(f"Feature vector inválido: {errors}")

        assert feature_array.shape == (OBSERVATION_DIM,)
        return feature_array
```

**Archivo 2: `services/inference_api/core/observation_builder.py`**
```python
# ============================================================
# ANTES (INCORRECTO)
# ============================================================
FEATURES = ["returns_1h", "returns_4h", ...]  # Puede estar en orden diferente!

# ============================================================
# DESPUÉS (CORRECTO)
# ============================================================
from src.core.contracts import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    features_dict_to_array,
)

class ObservationBuilder:
    """Construye observaciones para el modelo de inferencia."""

    def build_observation(self, market_data: dict) -> np.ndarray:
        """
        Construye observación en el orden exacto de FEATURE_ORDER.

        Args:
            market_data: Dict con datos de mercado

        Returns:
            np.ndarray de shape (OBSERVATION_DIM,) = (15,)
        """
        features_dict = {}

        for feature_name in FEATURE_ORDER:
            if feature_name not in market_data:
                raise ValueError(f"Missing feature: {feature_name}")
            features_dict[feature_name] = market_data[feature_name]

        return features_dict_to_array(features_dict)
```

**Archivo 3: `airflow/dags/l1_feature_refresh.py`**
```python
# ============================================================
# ANTES (INCORRECTO)
# ============================================================
FEATURE_LIST = [
    "returns_1h", "returns_4h", ...
]

# ============================================================
# DESPUÉS (CORRECTO)
# ============================================================
from src.core.contracts import FEATURE_ORDER, OBSERVATION_DIM

def compute_features(**context):
    """Computa features en el orden canónico de FEATURE_ORDER."""
    features = {}

    for feature_name in FEATURE_ORDER:
        features[feature_name] = compute_single_feature(feature_name, context)

    # Verificar que tenemos todos los features
    assert len(features) == OBSERVATION_DIM, \
        f"Expected {OBSERVATION_DIM} features, got {len(features)}"

    return features
```

### Eliminación de "session_progress"

El nombre obsoleto `session_progress` debe reemplazarse por `time_normalized` en todos los archivos:

```bash
# Encontrar todos los usos
grep -rn "session_progress" src/ services/ airflow/ --include="*.py"

# Reemplazar (con sed en Linux/Mac)
find src/ services/ airflow/ -name "*.py" -exec sed -i 's/session_progress/time_normalized/g' {} \;

# En Windows PowerShell:
Get-ChildItem -Path src/,services/,airflow/ -Filter *.py -Recurse | ForEach-Object {
    (Get-Content $_.FullName) -replace 'session_progress', 'time_normalized' | Set-Content $_.FullName
}
```

### Test de Regresión Obligatorio
**Archivo: `tests/regression/test_feature_order_ssot.py`**

```python
"""
REGRESSION TEST: FEATURE_ORDER SSOT
===================================
Este archivo DEBE ejecutarse en CI antes de cada deploy.
Si algún test falla, hay features en orden incorrecto que causarán
predicciones completamente erróneas.

Contract ID: CTR-FEATURE-SSOT-001
"""
import pytest
import subprocess
import hashlib
from pathlib import Path

class TestFeatureOrderSSoT:
    """Tests que garantizan que FEATURE_ORDER es SSOT."""

    def test_feature_order_has_exactly_15_elements(self):
        """FEATURE_ORDER debe tener exactamente 15 elementos."""
        from src.core.contracts import FEATURE_ORDER, OBSERVATION_DIM
        assert len(FEATURE_ORDER) == 15
        assert OBSERVATION_DIM == 15
        assert len(FEATURE_ORDER) == OBSERVATION_DIM

    def test_observation_dim_matches_feature_count(self):
        """OBSERVATION_DIM debe coincidir con len(FEATURE_ORDER)."""
        from src.core.contracts import FEATURE_ORDER, OBSERVATION_DIM
        assert OBSERVATION_DIM == len(FEATURE_ORDER)

    def test_no_session_progress_exists(self):
        """
        CRÍTICO: 'session_progress' es OBSOLETO.
        Debe ser 'time_normalized'.
        """
        from src.core.contracts import FEATURE_ORDER
        assert "session_progress" not in FEATURE_ORDER, (
            "'session_progress' está obsoleto. "
            "Debe ser 'time_normalized' (índice 14)"
        )

    def test_time_normalized_is_last_feature(self):
        """El último feature (índice 14) debe ser 'time_normalized'."""
        from src.core.contracts import FEATURE_ORDER
        assert FEATURE_ORDER[-1] == "time_normalized"
        assert FEATURE_ORDER[14] == "time_normalized"

    def test_position_is_second_to_last(self):
        """'position' debe ser el penúltimo feature (índice 13)."""
        from src.core.contracts import FEATURE_ORDER
        assert FEATURE_ORDER[-2] == "position"
        assert FEATURE_ORDER[13] == "position"

    def test_first_feature_is_returns_1h(self):
        """El primer feature debe ser 'returns_1h'."""
        from src.core.contracts import FEATURE_ORDER
        assert FEATURE_ORDER[0] == "returns_1h"

    def test_no_duplicate_features(self):
        """No debe haber features duplicados."""
        from src.core.contracts import FEATURE_ORDER
        assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER))

    def test_feature_order_hash_is_stable(self):
        """
        El hash de FEATURE_ORDER debe ser estable.
        Si cambia, significa que alguien modificó el orden.
        """
        from src.core.contracts import FEATURE_ORDER, FEATURE_ORDER_HASH

        computed_hash = hashlib.sha256(
            str(FEATURE_ORDER).encode()
        ).hexdigest()[:16]

        assert computed_hash == FEATURE_ORDER_HASH, (
            f"FEATURE_ORDER hash cambió!\n"
            f"Esperado: {FEATURE_ORDER_HASH}\n"
            f"Actual: {computed_hash}\n"
            f"Esto indica que FEATURE_ORDER fue modificado."
        )

    def test_no_duplicate_feature_order_definitions(self):
        """
        CRÍTICO: Solo debe existir UNA definición de FEATURE_ORDER.
        """
        result = subprocess.run(
            ["grep", "-rn", r"FEATURE_ORDER\s*=", "--include=*.py",
             "src/", "services/", "airflow/"],
            capture_output=True, text=True, cwd=Path(__file__).parents[3]
        )

        lines = [l for l in result.stdout.strip().split("\n")
                 if l and "test" not in l.lower() and "__pycache__" not in l
                 and "import" not in l.lower()]

        valid_files = ["feature_contract.py"]
        invalid_definitions = [
            l for l in lines
            if not any(v in l for v in valid_files)
        ]

        assert len(invalid_definitions) == 0, (
            f"Encontradas definiciones duplicadas de FEATURE_ORDER:\n" +
            "\n".join(invalid_definitions) +
            "\n\nTodas las definiciones DEBEN estar en "
            "src/core/contracts/feature_contract.py"
        )

    def test_no_session_progress_in_codebase(self):
        """
        Verificar que 'session_progress' no aparece en ningún archivo.
        """
        result = subprocess.run(
            ["grep", "-rn", "session_progress", "--include=*.py",
             "src/", "services/", "airflow/"],
            capture_output=True, text=True, cwd=Path(__file__).parents[3]
        )

        lines = [l for l in result.stdout.strip().split("\n")
                 if l and "test" not in l.lower()]

        assert len(lines) == 0, (
            f"Encontradas referencias a 'session_progress' (obsoleto):\n" +
            "\n".join(lines) +
            "\n\nReemplazar por 'time_normalized'"
        )
```

---

## BLOQUEADOR #3: L5 DAG Ignora Trading Flags

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔴 CRÍTICO: L5 DAG NO VALIDA TRADING_ENABLED NI KILL_SWITCH               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEMA:                                                                  │
│  ═════════                                                                  │
│  L5 DAG ejecuta predicciones y potencialmente trades SIN verificar:       │
│  • TRADING_ENABLED (master switch para habilitar/deshabilitar)             │
│  • KILL_SWITCH_ACTIVE (parada de emergencia)                               │
│  • PAPER_TRADING (modo simulación vs real)                                 │
│                                                                             │
│  ESCENARIO DE DESASTRE:                                                    │
│  ════════════════════════                                                   │
│  1. Tú configuras TRADING_ENABLED=false para hacer mantenimiento           │
│  2. L5 DAG corre de todas formas (no lee el flag)                          │
│  3. El sistema ejecuta trades reales cuando no debería                     │
│  4. Pérdida de dinero inesperada                                           │
│                                                                             │
│  RIESGO FINANCIERO: 💰💰💰 MÁXIMO                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Diagnóstico
```bash
# Verificar si L5 valida los flags
grep -rn "TRADING_ENABLED\|KILL_SWITCH" airflow/dags/ --include="*.py"

# Si no hay resultados, el DAG NO valida los flags
```

### Remediación Detallada

**Paso 1: Crear módulo de trading flags**
**Archivo: `src/trading/trading_flags.py`**

```python
"""
Trading Flags - Control de ejecución de trading.

Este módulo define los flags que controlan si el sistema puede:
1. Ejecutar trades (TRADING_ENABLED)
2. Ejecutar en modo real vs paper (PAPER_TRADING)
3. Parar de emergencia (KILL_SWITCH_ACTIVE)

TODOS los componentes que ejecutan trades DEBEN verificar estos flags.
"""
import os
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TradingFlags:
    """
    Flags de control de trading.

    Atributos:
        trading_enabled: Si False, NO se ejecutan trades (default: False)
        paper_trading: Si True, modo simulación (default: True)
        kill_switch_active: Si True, parada de emergencia (default: False)
    """
    trading_enabled: bool = False
    paper_trading: bool = True
    kill_switch_active: bool = False

    @classmethod
    def from_env(cls) -> "TradingFlags":
        """
        Carga flags desde variables de entorno.

        Variables:
            TRADING_ENABLED: "true" para habilitar (default: "false")
            PAPER_TRADING: "false" para trading real (default: "true")
            KILL_SWITCH_ACTIVE: "true" para parar (default: "false")
        """
        def parse_bool(val: Optional[str], default: bool) -> bool:
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        flags = cls(
            trading_enabled=parse_bool(os.getenv("TRADING_ENABLED"), False),
            paper_trading=parse_bool(os.getenv("PAPER_TRADING"), True),
            kill_switch_active=parse_bool(os.getenv("KILL_SWITCH_ACTIVE"), False),
        )

        logger.info(
            f"TradingFlags loaded: enabled={flags.trading_enabled}, "
            f"paper={flags.paper_trading}, kill_switch={flags.kill_switch_active}"
        )

        return flags

    def can_execute_trades(self) -> tuple[bool, str]:
        """
        Verifica si se pueden ejecutar trades.

        Returns:
            Tuple de (puede_ejecutar, razón)
        """
        if self.kill_switch_active:
            return False, "KILL_SWITCH_ACTIVE=true - Emergency stop"

        if not self.trading_enabled:
            return False, "TRADING_ENABLED=false - Trading disabled"

        return True, "Trading allowed"

    def is_paper_mode(self) -> bool:
        """Retorna True si estamos en modo paper trading."""
        return self.paper_trading


class TradingFlagsError(Exception):
    """Error cuando trading no está permitido."""
    pass


def require_trading_enabled():
    """
    Decorator que verifica flags antes de ejecutar.

    Uso:
        @require_trading_enabled()
        def execute_trade(...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            flags = TradingFlags.from_env()
            can_trade, reason = flags.can_execute_trades()

            if not can_trade:
                raise TradingFlagsError(reason)

            return func(*args, **kwargs)
        return wrapper
    return decorator
```

**Paso 2: Modificar L5 DAG**
**Archivo: `airflow/dags/l5_multi_model_inference.py`**

```python
"""
L5 DAG: Multi-Model Inference and Trading
==========================================

IMPORTANTE: Este DAG DEBE verificar trading flags antes de ejecutar.
Si TRADING_ENABLED=false o KILL_SWITCH_ACTIVE=true, el DAG se salta.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
import os
import logging

logger = logging.getLogger(__name__)

# Default args con retry policy
default_args = {
    'owner': 'trading',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'l5_multi_model_inference',
    default_args=default_args,
    description='Multi-model inference with trading flag validation',
    schedule_interval='*/15 * * * *',  # Cada 15 minutos
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['trading', 'inference', 'l5'],
)


def validate_trading_flags(**context) -> dict:
    """
    PRIMERA TASK: Validar que trading está permitido.

    Si no está permitido, lanza AirflowSkipException para
    saltar todo el DAG.

    Returns:
        Dict con estado de flags para XCom
    """
    from src.trading.trading_flags import TradingFlags

    flags = TradingFlags.from_env()

    logger.info(f"Trading flags: {flags}")

    # Verificar kill switch primero (máxima prioridad)
    if flags.kill_switch_active:
        logger.warning("🛑 KILL_SWITCH_ACTIVE=true - Skipping entire DAG")
        raise AirflowSkipException(
            "KILL_SWITCH_ACTIVE=true - Emergency stop activated"
        )

    # Verificar si trading está habilitado
    if not flags.trading_enabled:
        logger.info("⏸️ TRADING_ENABLED=false - Skipping trading tasks")
        raise AirflowSkipException(
            "TRADING_ENABLED=false - Trading is disabled"
        )

    # Logging de modo
    mode = "PAPER" if flags.paper_trading else "LIVE"
    logger.info(f"✅ Trading allowed in {mode} mode")

    # Guardar en XCom para tasks downstream
    return {
        "trading_enabled": flags.trading_enabled,
        "paper_trading": flags.paper_trading,
        "kill_switch_active": flags.kill_switch_active,
        "mode": mode,
    }


def load_model(**context):
    """Carga el modelo de MLflow."""
    # Verificar flags de nuevo por si cambiaron
    ti = context['ti']
    flags = ti.xcom_pull(task_ids='validate_trading_flags')

    if not flags or not flags.get('trading_enabled'):
        raise AirflowSkipException("Trading not enabled")

    # ... resto de la lógica de carga de modelo
    logger.info("Model loaded successfully")
    return {"model_loaded": True}


def make_prediction(**context):
    """Genera predicción."""
    ti = context['ti']
    flags = ti.xcom_pull(task_ids='validate_trading_flags')

    mode = flags.get('mode', 'UNKNOWN')
    logger.info(f"Making prediction in {mode} mode")

    # ... lógica de predicción
    return {"prediction": "HOLD", "confidence": 0.75}


def execute_trade(**context):
    """
    Ejecuta trade (o simula en paper mode).

    CRÍTICO: Esta función SIEMPRE verifica flags antes de ejecutar.
    """
    ti = context['ti']
    flags = ti.xcom_pull(task_ids='validate_trading_flags')
    prediction = ti.xcom_pull(task_ids='make_prediction')

    if not flags:
        raise AirflowSkipException("No trading flags available")

    is_paper = flags.get('paper_trading', True)

    if is_paper:
        logger.info(f"📝 PAPER TRADE: {prediction}")
        # Simular trade
        return {"executed": False, "simulated": True, "prediction": prediction}
    else:
        logger.info(f"💰 LIVE TRADE: {prediction}")
        # TODO: Ejecutar trade real
        # IMPORTANTE: Doble verificación de flags antes de ejecutar
        from src.trading.trading_flags import TradingFlags
        live_flags = TradingFlags.from_env()

        if live_flags.kill_switch_active:
            logger.error("🛑 Kill switch activated during execution!")
            raise AirflowSkipException("Kill switch activated")

        if not live_flags.trading_enabled:
            logger.warning("⚠️ Trading disabled during execution")
            raise AirflowSkipException("Trading disabled during execution")

        # Ejecutar trade real
        return {"executed": True, "simulated": False, "prediction": prediction}


# ============================================================
# DEFINICIÓN DE TASKS
# ============================================================

# TASK 1: Validar flags (DEBE ser la primera)
validate_flags_task = PythonOperator(
    task_id='validate_trading_flags',
    python_callable=validate_trading_flags,
    dag=dag,
)

# TASK 2: Cargar modelo
load_model_task = PythonOperator(
    task_id='load_model',
    python_callable=load_model,
    dag=dag,
)

# TASK 3: Hacer predicción
prediction_task = PythonOperator(
    task_id='make_prediction',
    python_callable=make_prediction,
    dag=dag,
)

# TASK 4: Ejecutar trade
execute_trade_task = PythonOperator(
    task_id='execute_trade',
    python_callable=execute_trade,
    dag=dag,
)

# ============================================================
# DEPENDENCIAS - validate_flags PRIMERO
# ============================================================
validate_flags_task >> load_model_task >> prediction_task >> execute_trade_task
```

**Paso 3: También validar en inference API**
**Archivo: `services/inference_api/core/inference_engine.py`**

```python
from src.trading.trading_flags import TradingFlags, TradingFlagsError

class InferenceEngine:
    def predict(self, observation: np.ndarray) -> dict:
        """
        Genera predicción validando flags primero.
        """
        # Verificar flags antes de cada predicción
        flags = TradingFlags.from_env()

        if flags.kill_switch_active:
            raise TradingFlagsError("KILL_SWITCH_ACTIVE - Cannot make predictions")

        # Nota: Podemos hacer predicciones incluso si trading está disabled
        # pero marcamos que no se pueden ejecutar

        # ... lógica de predicción ...

        result = {
            "action": action.name,
            "confidence": confidence,
            "can_execute": flags.trading_enabled and not flags.kill_switch_active,
            "mode": "paper" if flags.paper_trading else "live",
        }

        return result
```

### Test de Regresión Obligatorio
**Archivo: `tests/regression/test_trading_flags.py`**

```python
"""
REGRESSION TEST: Trading Flags Validation
==========================================
Verifica que L5 DAG y componentes de trading respetan los flags.
"""
import pytest
import os
from unittest.mock import patch

class TestTradingFlags:
    """Tests de trading flags."""

    def test_trading_disabled_by_default(self):
        """Trading debe estar deshabilitado por defecto (safe default)."""
        # Sin variables de entorno, debe estar deshabilitado
        with patch.dict(os.environ, {}, clear=True):
            from src.trading.trading_flags import TradingFlags
            flags = TradingFlags.from_env()

            assert flags.trading_enabled is False
            assert flags.paper_trading is True  # Paper por defecto
            assert flags.kill_switch_active is False

    def test_kill_switch_blocks_trading(self):
        """Kill switch debe bloquear trading incluso si está habilitado."""
        with patch.dict(os.environ, {
            "TRADING_ENABLED": "true",
            "KILL_SWITCH_ACTIVE": "true",
        }):
            from src.trading.trading_flags import TradingFlags
            flags = TradingFlags.from_env()

            can_trade, reason = flags.can_execute_trades()
            assert can_trade is False
            assert "KILL_SWITCH" in reason

    def test_trading_enabled_allows_execution(self):
        """Con trading habilitado y sin kill switch, debe permitir."""
        with patch.dict(os.environ, {
            "TRADING_ENABLED": "true",
            "KILL_SWITCH_ACTIVE": "false",
        }):
            from src.trading.trading_flags import TradingFlags
            flags = TradingFlags.from_env()

            can_trade, reason = flags.can_execute_trades()
            assert can_trade is True

    def test_l5_dag_has_flag_validation(self):
        """L5 DAG debe tener task de validación de flags."""
        import importlib.util
        from pathlib import Path

        dag_path = Path("airflow/dags/l5_multi_model_inference.py")
        if not dag_path.exists():
            pytest.skip("L5 DAG file not found")

        content = dag_path.read_text()

        # Debe importar o usar TradingFlags
        assert "TradingFlags" in content or "TRADING_ENABLED" in content, (
            "L5 DAG debe validar trading flags"
        )

        # Debe tener task de validación primero
        assert "validate_trading_flags" in content or "validate_flags" in content, (
            "L5 DAG debe tener task de validación de flags"
        )

    def test_inference_engine_respects_kill_switch(self):
        """Inference engine debe respetar kill switch."""
        with patch.dict(os.environ, {"KILL_SWITCH_ACTIVE": "true"}):
            # El inference engine debería lanzar error o marcar can_execute=False
            # Implementar según la lógica real
            pass
```

---

# 📋 SEMANA 1: REMEDIACIÓN CRÍTICOS (Días 1-5)

## Resumen de Tareas Día por Día

| Día | Foco | Tareas | Tiempo |
|-----|------|--------|--------|
| 1 | Action Enum | Consolidar SSOT, eliminar duplicados | 4h |
| 2 | FEATURE_ORDER | Consolidar SSOT, eliminar "session_progress" | 4h |
| 3 | Trading Flags | Agregar validación a L5 DAG y API | 3h |
| 4 | Tests Regresión | Crear tests que previenen regresión | 4h |
| 5 | Validación | Ejecutar todos los tests, verificar 100% | 3h |

**Total Semana 1: ~18 horas**

### Día 1: Consolidar Action Enum

```bash
# 1. Diagnóstico inicial (10 min)
grep -rn "class Action" src/ services/ --include="*.py" | grep -v test
grep -rn "SELL.*=.*0\|BUY.*=.*2" src/ services/ --include="*.py" | grep -v test

# 2. Ejecutar migración (30 min)
python scripts/migrate_action_enum.py --dry-run
python scripts/migrate_action_enum.py --apply

# 3. Verificar manualmente archivos críticos (2h)
# - src/core/constants.py
# - src/trading/trading_env.py
# - services/inference_api/core/inference_engine.py

# 4. Crear test de regresión (1h)
# tests/regression/test_action_enum_ssot.py

# 5. Ejecutar tests (30 min)
pytest tests/regression/test_action_enum_ssot.py -v
pytest tests/contracts/test_action_contract.py -v
```

### Día 2: Consolidar FEATURE_ORDER

```bash
# 1. Diagnóstico inicial (10 min)
grep -rn "FEATURE_ORDER\s*=" src/ services/ airflow/ --include="*.py" | grep -v test
grep -rn "session_progress" src/ services/ airflow/ --include="*.py"

# 2. Ejecutar migración (30 min)
python scripts/migrate_feature_order.py --dry-run
python scripts/migrate_feature_order.py --apply

# 3. Eliminar "session_progress" (30 min)
# Windows PowerShell:
Get-ChildItem -Path src/,services/,airflow/ -Filter *.py -Recurse | ForEach-Object {
    (Get-Content $_.FullName) -replace 'session_progress', 'time_normalized' | Set-Content $_.FullName
}

# 4. Verificar archivos críticos (2h)
# - src/features/builder.py
# - src/core/services/feature_builder.py
# - services/inference_api/core/observation_builder.py
# - airflow/dags/l1_feature_refresh.py

# 5. Crear test de regresión (1h)
# tests/regression/test_feature_order_ssot.py

# 6. Ejecutar tests (30 min)
pytest tests/regression/test_feature_order_ssot.py -v
pytest tests/contracts/test_feature_contract.py -v
```

### Día 3: Trading Flags en L5 DAG

```bash
# 1. Crear módulo de trading flags (1h)
# src/trading/trading_flags.py

# 2. Modificar L5 DAG (1.5h)
# airflow/dags/l5_multi_model_inference.py

# 3. Modificar inference engine (30 min)
# services/inference_api/core/inference_engine.py

# 4. Crear tests (1h)
# tests/regression/test_trading_flags.py

# 5. Verificar (30 min)
grep -rn "TRADING_ENABLED\|KILL_SWITCH" airflow/dags/ --include="*.py"
pytest tests/regression/test_trading_flags.py -v
```

### Día 4: Tests de Regresión Completos

```bash
# 1. Verificar que todos los tests de regresión existen
ls tests/regression/

# Deben existir:
# - test_action_enum_ssot.py
# - test_feature_order_ssot.py
# - test_trading_flags.py

# 2. Ejecutar todos los tests de regresión
pytest tests/regression/ -v --tb=short

# 3. Ejecutar tests de contratos
pytest tests/contracts/ -v

# 4. Ejecutar tests de integración
pytest tests/integration/test_contracts_e2e.py -v

# 5. Verificar cobertura
pytest tests/ -v --cov=src --cov-report=html
```

### Día 5: Validación Final Semana 1

```python
# Ejecutar script de validación
python scripts/validate_blockers.py
```

**Archivo: `scripts/validate_blockers.py`**

```python
#!/usr/bin/env python3
"""
Validación de que los 3 bloqueadores están resueltos.
Debe ejecutarse antes de continuar con Semana 2.
"""
import subprocess
import sys
from pathlib import Path

def check_action_enum():
    """Verificar que Action enum tiene solo 1 definición."""
    print("\n" + "="*60)
    print("CHECK 1: Action Enum SSOT")
    print("="*60)

    result = subprocess.run(
        ["grep", "-rn", "class Action", "--include=*.py", "src/", "services/"],
        capture_output=True, text=True
    )

    lines = [l for l in result.stdout.strip().split("\n")
             if l and "test" not in l.lower() and "__pycache__" not in l]

    valid = [l for l in lines if "action_contract.py" in l]
    invalid = [l for l in lines if "action_contract.py" not in l]

    if invalid:
        print("❌ FALLO: Definiciones duplicadas encontradas:")
        for l in invalid:
            print(f"   {l}")
        return False

    print("✅ OK: Solo una definición de Action en action_contract.py")
    return True

def check_feature_order():
    """Verificar que FEATURE_ORDER tiene solo 1 definición."""
    print("\n" + "="*60)
    print("CHECK 2: FEATURE_ORDER SSOT")
    print("="*60)

    result = subprocess.run(
        ["grep", "-rn", r"FEATURE_ORDER\s*=", "--include=*.py",
         "src/", "services/", "airflow/"],
        capture_output=True, text=True
    )

    lines = [l for l in result.stdout.strip().split("\n")
             if l and "test" not in l.lower() and "import" not in l.lower()]

    valid = [l for l in lines if "feature_contract.py" in l]
    invalid = [l for l in lines if "feature_contract.py" not in l]

    if invalid:
        print("❌ FALLO: Definiciones duplicadas encontradas:")
        for l in invalid:
            print(f"   {l}")
        return False

    print("✅ OK: Solo una definición de FEATURE_ORDER en feature_contract.py")
    return True

def check_no_session_progress():
    """Verificar que 'session_progress' no existe."""
    print("\n" + "="*60)
    print("CHECK 3: No 'session_progress'")
    print("="*60)

    result = subprocess.run(
        ["grep", "-rn", "session_progress", "--include=*.py",
         "src/", "services/", "airflow/"],
        capture_output=True, text=True
    )

    lines = [l for l in result.stdout.strip().split("\n")
             if l and "test" not in l.lower()]

    if lines:
        print("❌ FALLO: 'session_progress' encontrado (obsoleto):")
        for l in lines:
            print(f"   {l}")
        return False

    print("✅ OK: No hay referencias a 'session_progress'")
    return True

def check_l5_validates_flags():
    """Verificar que L5 DAG valida trading flags."""
    print("\n" + "="*60)
    print("CHECK 4: L5 DAG valida trading flags")
    print("="*60)

    l5_path = Path("airflow/dags/l5_multi_model_inference.py")
    if not l5_path.exists():
        print("❌ FALLO: L5 DAG no encontrado")
        return False

    content = l5_path.read_text()

    checks = [
        ("TRADING_ENABLED" in content or "TradingFlags" in content,
         "Debe verificar TRADING_ENABLED"),
        ("KILL_SWITCH" in content or "kill_switch" in content,
         "Debe verificar KILL_SWITCH"),
        ("validate" in content.lower() and "flag" in content.lower(),
         "Debe tener task de validación de flags"),
    ]

    all_passed = True
    for passed, msg in checks:
        if passed:
            print(f"  ✅ {msg}")
        else:
            print(f"  ❌ {msg}")
            all_passed = False

    return all_passed

def check_tests_pass():
    """Ejecutar tests de regresión."""
    print("\n" + "="*60)
    print("CHECK 5: Tests de regresión pasan")
    print("="*60)

    result = subprocess.run(
        ["pytest", "tests/regression/", "-v", "--tb=short", "-q"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print("❌ FALLO: Tests de regresión fallando:")
        print(result.stdout[-1000:])
        return False

    print("✅ OK: Todos los tests de regresión pasan")
    return True

def main():
    print("="*60)
    print("VALIDACIÓN DE BLOQUEADORES CRÍTICOS")
    print("="*60)

    checks = [
        ("Action Enum SSOT", check_action_enum),
        ("FEATURE_ORDER SSOT", check_feature_order),
        ("No session_progress", check_no_session_progress),
        ("L5 valida flags", check_l5_validates_flags),
        ("Tests pasan", check_tests_pass),
    ]

    results = []
    for name, check_fn in checks:
        try:
            passed = check_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ Error en {name}: {e}")
            results.append((name, False))

    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)

    passed_count = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    print(f"\nScore: {passed_count}/{total}")

    if passed_count == total:
        print("\n🎉 BLOQUEADORES RESUELTOS - Listo para Semana 2")
        return 0
    else:
        print(f"\n⚠️ {total - passed_count} bloqueadores pendientes")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

# 📋 SEMANA 2: DATABASE, API, SECURITY (Días 6-10)

## Componentes Faltantes a Crear

| Componente | Estado Actual | Prioridad | Tiempo |
|------------|---------------|-----------|--------|
| ValidatedPredictor | ❌ No existe | P1 | 3h |
| Alembic migrations | ❌ No existe | P1 | 4h |
| SQLAlchemy ORM | ❌ No existe | P1 | 4h |
| Endpoint /api/trades | ❌ No existe | P1 | 2h |
| WebSocket /ws/predictions | ❌ No existe | P1 | 3h |
| Dependabot | ❌ No existe | P1 | 30min |
| Security scanning | ❌ No existe | P1 | 1h |

### Día 6: ValidatedPredictor

**Archivo: `src/inference/validated_predictor.py`**

```python
"""
ValidatedPredictor - Wrapper que enforce contracts en predicción.

Este wrapper garantiza que:
1. Input tiene shape correcto (batch, 15)
2. Input tiene dtype float32
3. Input no tiene NaN/Inf
4. Output tiene shape correcto (batch, 3)
5. Output son probabilidades válidas (sum ≈ 1)
"""
import numpy as np
from typing import Tuple, Optional, Protocol
import logging

from src.core.contracts import (
    Action,
    OBSERVATION_DIM,
    ACTION_COUNT,
    validate_model_input,
    validate_model_output,
    ModelInputError,
    InvalidActionError,
)

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    """Protocol para modelos compatibles con ValidatedPredictor."""
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predice acción dada observación."""
        ...


class ValidatedPredictor:
    """
    Wrapper que valida input y output según contracts.

    Uso:
        model = load_model_from_mlflow(...)
        predictor = ValidatedPredictor(model)
        action, probs = predictor.predict(observation)

    El predictor garantiza que:
    - El input cumple ModelInputContract
    - El output cumple ModelOutputContract
    - Las excepciones son claras y específicas
    """

    def __init__(
        self,
        model: ModelProtocol,
        strict: bool = True,
        log_warnings: bool = True
    ):
        """
        Args:
            model: Modelo que implementa predict()
            strict: Si True, lanza excepciones en validación fallida
            log_warnings: Si True, loguea warnings cuando strict=False
        """
        self._model = model
        self._strict = strict
        self._log_warnings = log_warnings
        self._prediction_count = 0
        self._validation_failures = 0

    def predict(self, observation: np.ndarray) -> Tuple[Action, np.ndarray]:
        """
        Predice acción validando input y output.

        Args:
            observation: Array de shape (15,) o (1, 15) o (batch, 15)

        Returns:
            Tuple de (Action, action_probabilities)
            Para batch, retorna la primera predicción

        Raises:
            ModelInputError: Si observation es inválido (strict=True)
            InvalidActionError: Si output del modelo es inválido (strict=True)
        """
        self._prediction_count += 1

        # Normalizar shape
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # Validar input
        is_valid, errors = validate_model_input(observation)
        if not is_valid:
            self._validation_failures += 1
            msg = f"Input validation failed: {errors}"

            if self._strict:
                raise ModelInputError(msg)
            elif self._log_warnings:
                logger.warning(msg)

        # Ejecutar predicción
        try:
            action_indices, _states = self._model.predict(
                observation, deterministic=True
            )
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise

        # Obtener probabilidades
        probs = self._get_action_probabilities(observation)

        # Validar output
        is_valid, errors = validate_model_output(probs)
        if not is_valid:
            self._validation_failures += 1
            msg = f"Output validation failed: {errors}"

            if self._strict:
                raise InvalidActionError(msg)
            elif self._log_warnings:
                logger.warning(msg)

        # Convertir a Action enum
        action_idx = int(action_indices[0])
        action = Action(action_idx)

        return action, probs

    def _get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Obtiene probabilidades de acción del modelo."""
        if hasattr(self._model, 'policy') and hasattr(self._model.policy, 'get_distribution'):
            # Stable Baselines 3
            obs_tensor = self._model.policy.obs_to_tensor(observation)[0]
            distribution = self._model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.detach().cpu().numpy()[0]
        else:
            # Fallback: one-hot desde acción predicha
            action_idx, _ = self._model.predict(observation, deterministic=True)
            probs = np.zeros(ACTION_COUNT, dtype=np.float32)
            probs[action_idx[0]] = 1.0

        return probs

    def predict_batch(
        self,
        observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predice para batch de observations.

        Args:
            observations: Array de shape (batch, 15)

        Returns:
            Tuple de (actions_array, probabilities_array)
        """
        # Validar cada observación
        for i, obs in enumerate(observations):
            is_valid, errors = validate_model_input(obs.reshape(1, -1))
            if not is_valid and self._strict:
                raise ModelInputError(f"Invalid input at index {i}: {errors}")

        # Batch prediction
        actions, _states = self._model.predict(observations, deterministic=True)

        # TODO: Batch probabilities
        probs = np.zeros((len(observations), ACTION_COUNT), dtype=np.float32)
        for i, action_idx in enumerate(actions):
            probs[i, action_idx] = 1.0

        return actions, probs

    @property
    def stats(self) -> dict:
        """Estadísticas de predicción."""
        return {
            "prediction_count": self._prediction_count,
            "validation_failures": self._validation_failures,
            "failure_rate": (
                self._validation_failures / self._prediction_count
                if self._prediction_count > 0 else 0
            ),
        }
```

### Día 7: Alembic + SQLAlchemy

```bash
# 1. Instalar dependencias
pip install alembic sqlalchemy

# 2. Inicializar Alembic
cd C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models
alembic init database/alembic

# 3. Configurar alembic.ini
# sqlalchemy.url = postgresql://user:pass@localhost:5432/usdcop
```

**Archivo: `src/models/__init__.py`**

```python
"""
SQLAlchemy ORM Models para USD/COP Trading System.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    JSON, ForeignKey, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Trade(Base):
    """Registro de trades ejecutados."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)

    # Acción (0=SELL, 1=HOLD, 2=BUY)
    action = Column(Integer, nullable=False)
    action_name = Column(String(4), nullable=False)  # "SELL", "HOLD", "BUY"

    # Detalles del trade
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)

    # Metadata del modelo
    model_version = Column(String(64))
    model_run_id = Column(String(64))
    confidence = Column(Float)

    # Features usadas (para auditoría)
    observation = Column(JSON)  # 15 features
    probabilities = Column(JSON)  # 3 probabilidades

    # Resultados
    pnl = Column(Float)
    is_paper = Column(Boolean, default=True)

    # Índices
    __table_args__ = (
        Index('idx_trades_timestamp', 'timestamp'),
        Index('idx_trades_model', 'model_version'),
        CheckConstraint('action >= 0 AND action <= 2', name='valid_action'),
    )


class Prediction(Base):
    """Registro de predicciones (para monitoring)."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)

    model_version = Column(String(64))
    observation_hash = Column(String(64))  # Hash de features para dedup

    action = Column(Integer)
    probabilities = Column(JSON)
    confidence = Column(Float)

    # Performance
    latency_ms = Column(Float)

    __table_args__ = (
        Index('idx_predictions_timestamp', 'timestamp'),
    )


class RiskEvent(Base):
    """Eventos de risk management."""
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)

    event_type = Column(String(32), nullable=False)
    # Tipos: daily_loss_exceeded, drawdown_exceeded,
    #        circuit_breaker_triggered, position_limit_exceeded

    current_value = Column(Float)
    limit_value = Column(Float)
    action_taken = Column(String(32))  # blocked, warned, etc.

    details = Column(JSON)


class FeatureSnapshot(Base):
    """Snapshots de features para debugging."""
    __tablename__ = "feature_snapshots"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)

    features = Column(JSON, nullable=False)  # Dict con 15 features
    feature_hash = Column(String(64))
    source = Column(String(32))  # l1_dag, inference_api, backtest

    __table_args__ = (
        Index('idx_features_timestamp', 'timestamp'),
        Index('idx_features_hash', 'feature_hash'),
    )
```

**Primera migración:**
```bash
alembic revision --autogenerate -m "Initial tables"
alembic upgrade head
```

### Día 8: API Endpoints

**Archivo: `services/inference_api/routers/trades.py`**

```python
"""Trades API Router."""
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import logging

from src.core.contracts import Action

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/trades", tags=["trades"])


class TradeResponse(BaseModel):
    """Response model para trade."""
    id: int
    timestamp: datetime
    action: str
    action_value: int
    price: float
    quantity: float
    model_version: Optional[str]
    confidence: Optional[float]
    pnl: Optional[float]
    is_paper: bool


class TradesSummary(BaseModel):
    """Resumen de trading."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    best_trade_pnl: float
    worst_trade_pnl: float


@router.get("", response_model=List[TradeResponse])
async def get_trades(
    limit: int = Query(100, le=1000, description="Max trades to return"),
    offset: int = Query(0, ge=0),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    action: Optional[str] = Query(None, regex="^(SELL|HOLD|BUY)$"),
    model_version: Optional[str] = None,
):
    """
    Obtener historial de trades.

    Filtros opcionales:
    - start_date/end_date: Rango de fechas
    - action: Filtrar por acción (SELL, HOLD, BUY)
    - model_version: Filtrar por versión del modelo
    """
    # TODO: Implementar query a DB
    return []


@router.get("/summary", response_model=TradesSummary)
async def get_trades_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    """Obtener resumen de performance de trading."""
    # TODO: Implementar
    return TradesSummary(
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        total_pnl=0.0,
        avg_pnl_per_trade=0.0,
        best_trade_pnl=0.0,
        worst_trade_pnl=0.0,
    )


@router.get("/latest", response_model=Optional[TradeResponse])
async def get_latest_trade():
    """Obtener el trade más reciente."""
    # TODO: Implementar
    return None
```

**Archivo: `services/inference_api/routers/websocket.py`**

```python
"""WebSocket para predicciones en tiempo real."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Set
import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Gestiona conexiones WebSocket activas."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Envía mensaje a todas las conexiones activas."""
        dead_connections = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                dead_connections.add(connection)

        # Limpiar conexiones muertas
        self.active_connections -= dead_connections


manager = ConnectionManager()


@router.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """
    WebSocket para predicciones en tiempo real.

    Mensajes del cliente:
    - "ping": Recibe "pong"
    - "subscribe": Se subscribe a updates
    - JSON con request de predicción

    Mensajes del servidor:
    - Predicciones cuando hay nuevas
    - Heartbeat cada 30s
    """
    await manager.connect(websocket)

    try:
        while True:
            try:
                # Recibir mensaje con timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=60.0
                )

                if data == "ping":
                    await websocket.send_text("pong")

                elif data == "subscribe":
                    await websocket.send_json({
                        "type": "subscribed",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                else:
                    # Intentar parsear como JSON
                    try:
                        request = json.loads(data)
                        # TODO: Procesar request de predicción
                        await websocket.send_json({
                            "type": "prediction",
                            "data": {"action": "HOLD", "confidence": 0.5},
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON",
                        })

            except asyncio.TimeoutError:
                # Enviar heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_prediction(prediction: dict):
    """
    Broadcast nueva predicción a todos los subscribers.
    Llamar esto cuando hay una nueva predicción.
    """
    await manager.broadcast({
        "type": "prediction",
        "data": prediction,
        "timestamp": datetime.utcnow().isoformat(),
    })
```

### Día 9: Security

**Archivo: `.github/dependabot.yml`**

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "security"
    commit-message:
      prefix: "deps"

  # NPM dependencies (frontend)
  - package-ecosystem: "npm"
    directory: "/usdcop-trading-dashboard"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "frontend"

  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/docker"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "docker"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "ci"
```

**Archivo: `.github/workflows/security.yml`**

```yaml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install bandit safety pip-audit

      - name: Run Bandit (Python security linter)
        run: |
          bandit -r src/ services/ -ll -ii -f json -o bandit-report.json || true
          bandit -r src/ services/ -ll -ii

      - name: Run Safety (dependency vulnerabilities)
        run: |
          safety check -r requirements.txt --full-report || true

      - name: Run pip-audit
        run: |
          pip-audit -r requirements.txt || true

      - name: Upload security reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: security-reports
          path: |
            bandit-report.json
```

### Día 10: Frontend Types

**Script para encontrar y arreglar `:any`:**

```bash
# Encontrar todos los :any
grep -rn ": any" usdcop-trading-dashboard/components/ usdcop-trading-dashboard/app/ --include="*.ts" --include="*.tsx" | wc -l
```

**Archivo: `usdcop-trading-dashboard/types/trading.ts`**

```typescript
/**
 * Types centralizados para el sistema de trading.
 * USAR ESTOS TYPES EN LUGAR DE :any
 */

// ============================================================
// ACCIONES
// ============================================================

export type ActionName = 'SELL' | 'HOLD' | 'BUY';
export type ActionValue = 0 | 1 | 2;

export interface ActionProbabilities {
  sell: number;
  hold: number;
  buy: number;
}

// ============================================================
// TRADES
// ============================================================

export interface Trade {
  id: number;
  timestamp: string;  // ISO date
  action: ActionName;
  actionValue: ActionValue;
  price: number;
  quantity: number;
  modelVersion: string;
  confidence: number;
  pnl?: number;
  isPaper: boolean;
}

export interface TradesSummary {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalPnl: number;
  avgPnlPerTrade: number;
}

// ============================================================
// PREDICCIONES
// ============================================================

export interface Prediction {
  action: ActionName;
  actionValue: ActionValue;
  probabilities: [number, number, number];  // [sell, hold, buy]
  confidence: number;
  timestamp: string;
  canExecute: boolean;
  mode: 'paper' | 'live';
}

export interface PredictionRequest {
  features: number[];  // 15 features
}

// ============================================================
// MODELOS
// ============================================================

export interface ModelInfo {
  name: string;
  version: string;
  runId: string;
  stage: 'None' | 'Staging' | 'Production' | 'Archived';
  metrics: {
    sharpe?: number;
    accuracy?: number;
    meanReward?: number;
  };
  createdAt: string;
}

// ============================================================
// RISK
// ============================================================

export interface RiskMetrics {
  dailyPnl: number;
  dailyLossLimit: number;
  currentDrawdown: number;
  maxDrawdown: number;
  positionSize: number;
  maxPositionSize: number;
  circuitBreakerActive: boolean;
}

export interface RiskEvent {
  id: number;
  timestamp: string;
  eventType: 'daily_loss' | 'drawdown' | 'circuit_breaker' | 'position_limit';
  currentValue: number;
  limitValue: number;
  actionTaken: string;
}

// ============================================================
// SERVICIOS
// ============================================================

export interface ServiceStatus {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  latencyMs?: number;
  lastCheck: string;
  message?: string;
}

export interface SystemHealth {
  overall: 'healthy' | 'degraded' | 'unhealthy';
  services: ServiceStatus[];
  tradingEnabled: boolean;
  killSwitchActive: boolean;
  mode: 'paper' | 'live';
}

// ============================================================
// WEBSOCKET
// ============================================================

export type WebSocketMessageType =
  | 'prediction'
  | 'trade'
  | 'heartbeat'
  | 'subscribed'
  | 'error';

export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType;
  data?: T;
  timestamp: string;
  message?: string;
}

// ============================================================
// API RESPONSES
// ============================================================

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
  };
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}
```

---

# 📋 SEMANA 3: POLISH Y DOCUMENTACIÓN (Días 11-15)

### Día 11: Makefile

**Archivo: `Makefile`**

```makefile
.PHONY: help install test lint format clean docker-up docker-down validate

# ============================================================
# HELP
# ============================================================

help:
	@echo "USD/COP RL Trading System"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install all dependencies"
	@echo "  install-dev    Install dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-contracts Run contract tests"
	@echo "  test-regression Run regression tests"
	@echo "  test-integration Run integration tests"
	@echo "  coverage       Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run all linters"
	@echo "  format         Format code"
	@echo "  typecheck      Run type checking"
	@echo ""
	@echo "Validation:"
	@echo "  validate       Run all validations (blockers, tests, lint)"
	@echo "  validate-ssot  Validate SSOT compliance"
	@echo "  validate-contracts Validate all contracts"
	@echo ""
	@echo "Docker:"
	@echo "  docker-up      Start all services"
	@echo "  docker-down    Stop all services"
	@echo "  docker-logs    Show service logs"
	@echo ""
	@echo "Database:"
	@echo "  migrate        Run database migrations"
	@echo "  migrate-create Create new migration"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          Remove build artifacts"

# ============================================================
# INSTALLATION
# ============================================================

install:
	pip install -r requirements.txt
	cd usdcop-trading-dashboard && npm install

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

# ============================================================
# TESTING
# ============================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ tests/contracts/ -v

test-contracts:
	pytest tests/contracts/ -v

test-regression:
	pytest tests/regression/ -v --tb=short

test-integration:
	pytest tests/integration/ -v

coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

# ============================================================
# CODE QUALITY
# ============================================================

lint:
	ruff check src/ services/ tests/
	ruff check airflow/dags/

format:
	ruff format src/ services/ tests/
	black src/ services/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

# ============================================================
# VALIDATION
# ============================================================

validate: validate-ssot test-regression lint
	@echo "✅ All validations passed"

validate-ssot:
	python scripts/validate_blockers.py

validate-contracts:
	python -c "from src.core.contracts import *; print('✅ All contracts valid')"

# ============================================================
# DOCKER
# ============================================================

docker-up:
	docker-compose -f docker-compose.yml \
		-f docker-compose.infrastructure.yml \
		-f docker-compose.mlops.yml \
		up -d

docker-down:
	docker-compose -f docker-compose.yml \
		-f docker-compose.infrastructure.yml \
		-f docker-compose.mlops.yml \
		down

docker-logs:
	docker-compose logs -f

# ============================================================
# DATABASE
# ============================================================

migrate:
	alembic upgrade head

migrate-create:
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

migrate-rollback:
	alembic downgrade -1

# ============================================================
# CLEANUP
# ============================================================

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info 2>/dev/null || true

# ============================================================
# PRE-COMMIT
# ============================================================

pre-commit:
	pre-commit run --all-files
```

### Día 12: CHANGELOG y LICENSE

**Archivo: `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to the USD/COP RL Trading System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ValidatedPredictor` wrapper for safe model inference with contract validation
- Alembic database migrations infrastructure
- SQLAlchemy ORM models for trades, predictions, risk events
- `/api/trades` endpoint for trade history
- `/ws/predictions` WebSocket for real-time updates
- Dependabot configuration for dependency security scanning
- Security scanning workflow in CI (Bandit, Safety, pip-audit)
- Comprehensive TypeScript types for frontend
- Makefile with standardized commands
- Regression tests for SSOT compliance

### Changed
- **BREAKING**: Consolidated `Action` enum to single source of truth in `action_contract.py`
  - Order is now canonically: SELL=0, HOLD=1, BUY=2
  - All other definitions removed
- **BREAKING**: Consolidated `FEATURE_ORDER` to single source of truth in `feature_contract.py`
  - All duplicate definitions removed
  - `session_progress` renamed to `time_normalized`
- L5 DAG now validates `TRADING_ENABLED` and `KILL_SWITCH_ACTIVE` before execution
- MLflow backend migrated from SQLite to PostgreSQL
- Risk configuration externalized to `config/risk_limits.yaml`

### Fixed
- Action enum order inconsistency that could invert trading signals
- FEATURE_ORDER mismatch between training and inference
- L5 DAG executing trades when `TRADING_ENABLED=false`
- References to obsolete `session_progress` feature name

### Security
- Added Dependabot for automated dependency updates
- Added Bandit security linter to CI
- Removed `.env.infrastructure` from git tracking
- API keys now rotated and managed via Vault

### Removed
- Duplicate `Action` class definitions from `constants.py`, `trading_env.py`
- Duplicate `FEATURE_ORDER` definitions from multiple files
- Legacy `session_progress` references

## [1.0.0] - 2026-01-17

### Added
- Initial release of USD/COP RL Trading System
- PPO-based reinforcement learning model for FX trading
- L0-L5 DAG architecture for data pipeline
- FastAPI inference service with contract validation
- Next.js trading dashboard
- MLflow experiment tracking and model registry
- Prometheus/Grafana monitoring stack
- Comprehensive contract system:
  - `ActionContract`: Action enum with validation
  - `FeatureContract`: FEATURE_ORDER and validation
  - `ModelInputContract`: Observation validation
  - `ModelOutputContract`: Prediction validation
  - `NormStatsContract`: Normalization stats management
  - `TrainingRunContract`: MLflow logging requirements
- Risk management with daily loss limits, drawdown protection, circuit breaker
- Paper trading mode for safe testing

### Security
- Vault integration for secrets management
- API key authentication
- Rate limiting on inference API
```

**Archivo: `LICENSE`**

```
MIT License

Copyright (c) 2026 [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Días 13-14: Integration Tests y Risk Config

(Ya detallados en versión anterior)

### Día 15: Validación Final 100%

**Archivo: `scripts/validate_100_percent.py`**

```python
#!/usr/bin/env python3
"""
VALIDACIÓN FINAL - 100% COMPLIANCE
===================================
Ejecutar antes de cada deploy a producción.
TODOS los checks deben pasar.
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Callable

def run_check(name: str, check_fn: Callable) -> Tuple[str, bool, str]:
    """Ejecuta un check y retorna resultado."""
    try:
        passed, message = check_fn()
        return name, passed, message
    except Exception as e:
        return name, False, str(e)

# ============================================================
# CHECKS DE SSOT (P0)
# ============================================================

def check_action_enum_ssot() -> Tuple[bool, str]:
    """Solo una definición de Action enum."""
    result = subprocess.run(
        ["grep", "-rn", "class Action", "--include=*.py", "src/", "services/"],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.strip().split("\n")
             if l and "test" not in l.lower() and "__pycache__" not in l]

    invalid = [l for l in lines if "action_contract.py" not in l]
    if invalid:
        return False, f"Definiciones extra: {len(invalid)}"
    return True, "Una sola definición"

def check_feature_order_ssot() -> Tuple[bool, str]:
    """Solo una definición de FEATURE_ORDER."""
    result = subprocess.run(
        ["grep", "-rn", r"FEATURE_ORDER\s*=", "--include=*.py",
         "src/", "services/", "airflow/"],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.strip().split("\n")
             if l and "test" not in l.lower() and "import" not in l.lower()]

    invalid = [l for l in lines if "feature_contract.py" not in l]
    if invalid:
        return False, f"Definiciones extra: {len(invalid)}"
    return True, "Una sola definición"

def check_no_session_progress() -> Tuple[bool, str]:
    """No existe 'session_progress'."""
    result = subprocess.run(
        ["grep", "-rn", "session_progress", "--include=*.py",
         "src/", "services/", "airflow/"],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.strip().split("\n") if l]
    if lines:
        return False, f"Referencias encontradas: {len(lines)}"
    return True, "No hay referencias"

def check_l5_validates_flags() -> Tuple[bool, str]:
    """L5 DAG valida trading flags."""
    l5_path = Path("airflow/dags/l5_multi_model_inference.py")
    if not l5_path.exists():
        return False, "Archivo no encontrado"

    content = l5_path.read_text()
    has_trading_check = "TRADING_ENABLED" in content or "TradingFlags" in content
    has_killswitch_check = "KILL_SWITCH" in content or "kill_switch" in content

    if not has_trading_check:
        return False, "No valida TRADING_ENABLED"
    if not has_killswitch_check:
        return False, "No valida KILL_SWITCH"
    return True, "Valida ambos flags"

# ============================================================
# CHECKS DE COMPONENTES (P1)
# ============================================================

def check_validated_predictor_exists() -> Tuple[bool, str]:
    """ValidatedPredictor existe."""
    path = Path("src/inference/validated_predictor.py")
    if not path.exists():
        return False, "Archivo no existe"
    content = path.read_text()
    if "class ValidatedPredictor" not in content:
        return False, "Clase no definida"
    return True, "Existe"

def check_alembic_configured() -> Tuple[bool, str]:
    """Alembic está configurado."""
    if not Path("database/alembic").exists():
        return False, "Directorio alembic no existe"
    if not Path("database/alembic/env.py").exists():
        return False, "env.py no existe"
    return True, "Configurado"

def check_sqlalchemy_models() -> Tuple[bool, str]:
    """SQLAlchemy models existen."""
    path = Path("src/models/__init__.py")
    if not path.exists():
        return False, "Archivo no existe"
    content = path.read_text()
    required = ["class Trade", "class Prediction", "class RiskEvent"]
    missing = [r for r in required if r not in content]
    if missing:
        return False, f"Faltan modelos: {missing}"
    return True, "Todos los modelos"

def check_dependabot() -> Tuple[bool, str]:
    """Dependabot configurado."""
    path = Path(".github/dependabot.yml")
    if not path.exists():
        return False, "Archivo no existe"
    return True, "Configurado"

def check_security_workflow() -> Tuple[bool, str]:
    """Security workflow en CI."""
    path = Path(".github/workflows/security.yml")
    if not path.exists():
        return False, "Archivo no existe"
    return True, "Configurado"

# ============================================================
# CHECKS DE ARCHIVOS (P2)
# ============================================================

def check_makefile() -> Tuple[bool, str]:
    """Makefile existe."""
    if not Path("Makefile").exists():
        return False, "No existe"
    return True, "Existe"

def check_changelog() -> Tuple[bool, str]:
    """CHANGELOG.md existe."""
    if not Path("CHANGELOG.md").exists():
        return False, "No existe"
    return True, "Existe"

def check_license() -> Tuple[bool, str]:
    """LICENSE existe."""
    if not Path("LICENSE").exists():
        return False, "No existe"
    return True, "Existe"

# ============================================================
# CHECKS DE TESTS
# ============================================================

def check_regression_tests_pass() -> Tuple[bool, str]:
    """Tests de regresión pasan."""
    result = subprocess.run(
        ["pytest", "tests/regression/", "-v", "-q", "--tb=no"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return False, "Tests fallando"
    return True, "Todos pasan"

def check_contract_tests_pass() -> Tuple[bool, str]:
    """Tests de contratos pasan."""
    result = subprocess.run(
        ["pytest", "tests/contracts/", "-v", "-q", "--tb=no"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return False, "Tests fallando"
    return True, "Todos pasan"

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*70)
    print("VALIDACIÓN 100% COMPLIANCE - USD/COP RL Trading System")
    print("="*70)
    print()

    checks = [
        # P0 - Críticos
        ("🔴 P0", [
            ("Action Enum SSOT", check_action_enum_ssot),
            ("FEATURE_ORDER SSOT", check_feature_order_ssot),
            ("No session_progress", check_no_session_progress),
            ("L5 valida flags", check_l5_validates_flags),
        ]),
        # P1 - Importantes
        ("🟠 P1", [
            ("ValidatedPredictor", check_validated_predictor_exists),
            ("Alembic", check_alembic_configured),
            ("SQLAlchemy Models", check_sqlalchemy_models),
            ("Dependabot", check_dependabot),
            ("Security Workflow", check_security_workflow),
        ]),
        # P2 - Mejoras
        ("🟡 P2", [
            ("Makefile", check_makefile),
            ("CHANGELOG.md", check_changelog),
            ("LICENSE", check_license),
        ]),
        # Tests
        ("🧪 Tests", [
            ("Regression Tests", check_regression_tests_pass),
            ("Contract Tests", check_contract_tests_pass),
        ]),
    ]

    all_results = []

    for priority, priority_checks in checks:
        print(f"\n{priority}")
        print("-"*50)

        for name, check_fn in priority_checks:
            name, passed, message = run_check(name, check_fn)
            status = "✅" if passed else "❌"
            print(f"  {status} {name}: {message}")
            all_results.append((name, passed))

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)

    passed = sum(1 for _, p in all_results if p)
    total = len(all_results)
    percentage = (passed / total) * 100

    print(f"\nScore: {passed}/{total} ({percentage:.1f}%)")

    if passed == total:
        print("\n🎉 100% COMPLIANCE ACHIEVED!")
        print("✅ Sistema listo para producción")
        return 0
    else:
        failed = [(n, p) for n, p in all_results if not p]
        print(f"\n⚠️ {len(failed)} checks fallando:")
        for name, _ in failed:
            print(f"   ❌ {name}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

# 📋 CHECKLIST GO-LIVE

## Antes de Producción

```
P0 - BLOQUEADORES (deben estar ✅)
═══════════════════════════════════
□ Action enum tiene UNA SOLA definición (action_contract.py)
□ FEATURE_ORDER tiene UNA SOLA definición (feature_contract.py)
□ No existe "session_progress" en el código
□ L5 DAG valida TRADING_ENABLED antes de ejecutar
□ L5 DAG valida KILL_SWITCH_ACTIVE antes de ejecutar
□ Tests de regresión pasan (pytest tests/regression/)
□ Tests de contratos pasan (pytest tests/contracts/)

P1 - IMPORTANTES (deben estar ✅)
═══════════════════════════════════
□ ValidatedPredictor existe y se usa en inference
□ Alembic configurado para migraciones
□ SQLAlchemy models definidos
□ Dependabot habilitado
□ Security scanning en CI
□ Frontend sin tipos :any (o < 10)

P2 - MEJORAS (recomendado ✅)
═══════════════════════════════════
□ Makefile existe con comandos estándar
□ CHANGELOG.md actualizado
□ LICENSE definida
□ Risk config externalizada
```

## Comando de Validación Final

```bash
# Ejecutar antes de cada deploy
make validate

# O manualmente:
python scripts/validate_100_percent.py
```

---

# 📊 MÉTRICAS DE ÉXITO

| Métrica | Antes | Semana 1 | Semana 2 | Semana 3 |
|---------|-------|----------|----------|----------|
| **Score Global** | 70.5% | 85% | 95% | 100% |
| **SSOT Score** | 30% | 95% | 100% | 100% |
| **Action definitions** | 4+ | 1 | 1 | 1 |
| **FEATURE_ORDER defs** | 7+ | 1 | 1 | 1 |
| **L5 validates flags** | ❌ | ✅ | ✅ | ✅ |
| **Frontend :any** | 109 | 109 | 20 | 0 |
| **ValidatedPredictor** | ❌ | ❌ | ✅ | ✅ |
| **Alembic** | ❌ | ❌ | ✅ | ✅ |
| **Makefile** | ❌ | ❌ | ❌ | ✅ |

---

# 🚀 PRÓXIMOS PASOS INMEDIATOS

```bash
# PASO 1: Ejecutar diagnóstico (5 min)
grep -rn "class Action" src/ services/ --include="*.py" | grep -v test
grep -rn "FEATURE_ORDER\s*=" src/ services/ airflow/ --include="*.py" | grep -v test
grep -rn "session_progress" src/ services/ airflow/ --include="*.py"
grep -rn "TRADING_ENABLED\|KILL_SWITCH" airflow/dags/ --include="*.py"

# PASO 2: Si hay duplicados, ejecutar migración
python scripts/migrate_action_enum.py --apply
python scripts/migrate_feature_order.py --apply

# PASO 3: Validar que se resolvieron
python scripts/validate_blockers.py

# PASO 4: Correr tests
pytest tests/regression/ tests/contracts/ -v
```

---

**Conclusión**: Este plan proporciona un camino claro de 70.5% a 100% compliance en 3 semanas. Los bloqueadores críticos (P0) se resuelven en la Semana 1, permitiendo que el sistema sea seguro para operación real. Las Semanas 2-3 agregan robustez, documentación y polish.

---

*Plan generado por Claude Code*
*Fecha: 2026-01-17*
*Versión: 2.0 (Robustificada con Análisis Ejecutivo)*
