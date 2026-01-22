# PLAN DE REMEDIACION DE CONTRATOS - VERSION ROBUSTECIDA
## USD/COP RL Trading System - De 63% a 100%

**Fecha**: 2026-01-17
**Auditoría Base**: 250 preguntas, 18 categorías
**Score Actual**: 63% (146 Pass, 71 Parcial, 27 Fail)
**Score Objetivo**: 100%
**Duración**: 3 semanas (15 días hábiles)

---

# LOS 5 GAPS CRITICOS A RESOLVER

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           5 GAPS CRITICOS - PRIORIDAD MAXIMA                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   GAP #1: ACTION ENUM FRAGMENTADO                                                   │
│   ═══════════════════════════════                                                   │
│   ❌ 4 definiciones diferentes en el código:                                        │
│      • src/core/constants.py:         ACTION_HOLD=0, BUY=1, SELL=2  (WRONG ORDER)   │
│      • src/training/environments/:    TradingAction SHORT=-1, HOLD=0, LONG=1        │
│      • src/shared/schemas/core.py:    BackendAction strings (LONG, SHORT, HOLD)     │
│      • services/inference_api/:       SignalType BUY, SELL, HOLD strings            │
│   🎯 SOLUCIÓN: SSOT en src/core/contracts/action_contract.py                        │
│                SELL=0, HOLD=1, BUY=2 (orden correcto)                               │
│                                                                                      │
│   GAP #2: FEATURE_ORDER DUPLICADO                                                   │
│   ═══════════════════════════════                                                   │
│   ❌ 3 definiciones inconsistentes:                                                 │
│      • src/feature_store/adapters.py:71     "session_progress" (NOMBRE INCORRECTO!)│
│      • src/features/feature_reader.py:29    Solo 13 features (FALTAN 2!)           │
│      • src/ml_workflow/training_pipeline    Definición local (DUPLICADO!)          │
│   🎯 SOLUCIÓN: SSOT en src/core/contracts/feature_contract.py                       │
│                15 features exactas, "time_normalized" (no session_progress)         │
│                                                                                      │
│   GAP #3: MLFLOW SIGNATURE NO ATTACHED                                              │
│   ═════════════════════════════════                                                 │
│   ❌ Signature se guarda como JSON artifact, NO attached al modelo:                 │
│      mlflow.log_dict(signature_dict, "model_signature.json")  # INCORRECTO!        │
│   🎯 SOLUCIÓN: Usar mlflow.pyfunc.log_model(..., signature=signature)               │
│                                                                                      │
│   GAP #4: NO EXISTE TrainingRunContract                                             │
│   ══════════════════════════════════                                                │
│   ❌ No hay validación de qué params/metrics son obligatorios:                      │
│      • dataset_hash puede faltar sin error                                          │
│      • norm_stats_hash puede faltar sin error                                       │
│   🎯 SOLUCIÓN: TrainingRunContract que valida ANTES de mlflow.end_run()             │
│                                                                                      │
│   GAP #5: MODEL OUTPUT NO VALIDADO (Score: 7%)                                      │
│   ════════════════════════════════════════════                                      │
│   ❌ Output del modelo no se valida:                                                │
│      • Action puede ser 5, -1, o cualquier valor                                   │
│      • Confidence puede ser -0.5 o 1.5                                              │
│      • No hay validación de action_probs                                            │
│   🎯 SOLUCIÓN: ModelOutputContract con validate_model_output()                      │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

# ROADMAP DE 3 SEMANAS CON METRICAS

```
SEMANA 1: CONTRATOS DE MODELO (P0)
═══════════════════════════════════════════════════════════════════════════════════
Score: 63% ──► 78%

├── Día 1: ACTION ENUM SSOT + MODEL OUTPUT CONTRACT
│   • Crear src/core/contracts/action_contract.py
│   • Migrar todos los usos existentes
│   • Tests de validación
│   └── Score esperado: 67%

├── Día 2: FEATURE_ORDER SSOT + FEATURE CONTRACT
│   • Crear src/core/contracts/feature_contract.py
│   • Fix session_progress → time_normalized
│   • Tests de SSOT único
│   └── Score esperado: 71%

├── Día 3: MODEL INPUT CONTRACT + VALIDATION WRAPPER
│   • Crear src/core/contracts/model_input_contract.py
│   • ObservationValidator para TODOS los paths
│   • Tests de validación
│   └── Score esperado: 74%

├── Día 4: MLFLOW SIGNATURE + TRAINING RUN CONTRACT
│   • Crear src/training/mlflow_signature.py
│   • Crear src/core/contracts/training_run_contract.py
│   • Tests de logging obligatorio
│   └── Score esperado: 76%

└── Día 5: MODEL METADATA CONTRACT + TESTS COMPLETOS
    • Crear src/core/contracts/model_metadata_contract.py
    • Suite completa tests/unit/test_model_contracts.py
    └── Score esperado: 78%


SEMANA 2: CONTRATOS DE DATOS Y API (P1)
═══════════════════════════════════════════════════════════════════════════════════
Score: 78% ──► 88%

├── Día 6: NORM STATS CONTRACT + JSON SCHEMA
├── Día 7: L0→L1 CONTRACT + PRICE VALIDATION
├── Día 8: L1→L5 CONTRACT + HASH VALIDATION
├── Día 9: API REQUEST/RESPONSE CONTRACTS
└── Día 10: DATABASE CONTRACTS + ALEMBIC


SEMANA 3: SYNC + POLISH (P2)
═══════════════════════════════════════════════════════════════════════════════════
Score: 88% ──► 100%

├── Día 11: FRONTEND-BACKEND SYNC + CI
├── Día 12: WEBSOCKET CONTRACTS
├── Día 13: JSONB SCHEMA + ALEMBIC MIGRATIONS
├── Día 14: E2E CONTRACT TESTS SUITE
└── Día 15: RE-AUDIT + DOCUMENTACION + GO-LIVE
```

---

# ════════════════════════════════════════════════════════════════════════════════
# SEMANA 1: CONTRATOS DE MODELO (P0) - IMPLEMENTACION DETALLADA
# Score: 63% → 78%
# ════════════════════════════════════════════════════════════════════════════════

---

## DIA 1: ACTION ENUM SSOT + MODEL OUTPUT CONTRACT

### Problema Actual - Análisis Detallado

```python
# ══════════════════════════════════════════════════════════════════════════════
# PROBLEMA: 4 DEFINICIONES DIFERENTES DE ACTION
# ══════════════════════════════════════════════════════════════════════════════

# Definición 1: src/core/constants.py (ORDEN INCORRECTO!)
ACTION_HOLD = 0   # ❌ HOLD no debería ser 0
ACTION_BUY = 1    # ❌ BUY no debería ser 1
ACTION_SELL = 2   # ❌ SELL no debería ser 2

# Definición 2: src/training/environments/trading_env.py
class TradingAction(Enum):
    SHORT = -1    # ❌ Valores negativos no compatibles con modelo
    HOLD = 0
    LONG = 1

# Definición 3: src/shared/schemas/core.py
class BackendAction(str, Enum):
    LONG = "LONG"     # ❌ Strings no compatibles con argmax
    SHORT = "SHORT"
    HOLD = "HOLD"

# Definición 4: services/inference_api/schemas.py
class SignalType(str, Enum):
    BUY = "BUY"       # ❌ Otro conjunto de strings
    SELL = "SELL"
    HOLD = "HOLD"

# PROBLEMA CRITICO:
# El modelo PPO usa softmax → argmax que produce 0, 1, o 2
# Cada definición interpreta estos valores diferentemente!
# SELL=0, HOLD=1, BUY=2 es el mapping correcto del modelo entrenado
```

### Solución 1A: Action Contract SSOT

**Archivo**: `src/core/contracts/action_contract.py`

```python
"""
ACTION CONTRACT - Single Source of Truth
========================================
Este es el ÚNICO lugar donde se definen las acciones del modelo.
TODOS los demás módulos DEBEN importar de aquí.

Contract ID: CTR-ACTION-001
Version: 1.0.0

IMPORTANTE:
- El modelo PPO produce output con shape (3,) = [P(SELL), P(HOLD), P(BUY)]
- argmax produce: 0=SELL, 1=HOLD, 2=BUY
- Este mapping es INMUTABLE y corresponde al modelo entrenado
"""
from enum import IntEnum
from typing import Dict, Tuple, Final, List
from dataclasses import dataclass


class Action(IntEnum):
    """
    Acción del modelo RL - SSOT.

    MAPPING CORRECTO (corresponde al output del modelo):
        SELL = 0  -> Vender/Short  -> argmax cuando P(SELL) es mayor
        HOLD = 1  -> Mantener      -> argmax cuando P(HOLD) es mayor
        BUY = 2   -> Comprar/Long  -> argmax cuando P(BUY) es mayor

    USUARIOS DE ESTE ENUM:
    - Training (PPO output interpretation)
    - Inference (model.predict())
    - Backtest (signal generation)
    - API (response format)
    - Frontend (display)
    """
    SELL = 0
    HOLD = 1
    BUY = 2

    @classmethod
    def from_int(cls, value: int) -> "Action":
        """
        Convierte int a Action con validación estricta.

        Args:
            value: Valor entero (debe ser 0, 1, o 2)

        Returns:
            Action correspondiente

        Raises:
            InvalidActionError: Si value no está en {0, 1, 2}

        Example:
            >>> Action.from_int(0)
            Action.SELL
            >>> Action.from_int(5)
            InvalidActionError: Invalid action value: 5
        """
        if value not in (0, 1, 2):
            raise InvalidActionError(
                f"Invalid action value: {value}. "
                f"Must be 0 (SELL), 1 (HOLD), or 2 (BUY)."
            )
        return cls(value)

    @classmethod
    def from_string(cls, value: str) -> "Action":
        """
        Convierte string a Action (case-insensitive).

        Acepta múltiples formatos:
        - SELL: "sell", "short", "s", "-1"
        - HOLD: "hold", "flat", "h", "0"
        - BUY: "buy", "long", "b", "1"

        Args:
            value: String representando la acción

        Returns:
            Action correspondiente

        Raises:
            InvalidActionError: Si string no reconocido
        """
        mapping = {
            # SELL aliases
            "sell": cls.SELL, "short": cls.SELL, "s": cls.SELL, "-1": cls.SELL,
            # HOLD aliases
            "hold": cls.HOLD, "flat": cls.HOLD, "h": cls.HOLD, "0": cls.HOLD,
            # BUY aliases
            "buy": cls.BUY, "long": cls.BUY, "b": cls.BUY, "1": cls.BUY,
        }
        normalized = value.lower().strip()
        if normalized not in mapping:
            valid = list(set(mapping.keys()))
            raise InvalidActionError(
                f"Invalid action string: '{value}'. "
                f"Valid values: {valid}"
            )
        return mapping[normalized]

    @classmethod
    def from_model_output(cls, logits_or_probs: list) -> "Action":
        """
        Convierte output del modelo (logits o probabilidades) a Action.

        Args:
            logits_or_probs: Array de 3 valores [SELL, HOLD, BUY]

        Returns:
            Action con mayor probabilidad

        Example:
            >>> Action.from_model_output([0.1, 0.2, 0.7])
            Action.BUY
        """
        if len(logits_or_probs) != 3:
            raise InvalidActionError(
                f"Model output must have 3 values, got {len(logits_or_probs)}"
            )
        return cls(int(max(range(3), key=lambda i: logits_or_probs[i])))

    def to_string(self) -> str:
        """Convierte Action a string legible."""
        return self.name

    def to_signal(self) -> str:
        """Convierte a señal de trading."""
        return {
            self.SELL: "SHORT",
            self.HOLD: "FLAT",
            self.BUY: "LONG",
        }[self]

    def to_position(self) -> int:
        """Convierte a posición numérica: -1 (short), 0 (flat), 1 (long)."""
        return {
            self.SELL: -1,
            self.HOLD: 0,
            self.BUY: 1,
        }[self]

    @property
    def is_entry(self) -> bool:
        """True si es señal de entrada (BUY o SELL)."""
        return self in (Action.BUY, Action.SELL)

    @property
    def is_exit(self) -> bool:
        """True si es señal de salida (HOLD)."""
        return self == Action.HOLD

    @property
    def direction(self) -> int:
        """Dirección: -1 (short), 0 (flat), 1 (long)."""
        return self.to_position()


class InvalidActionError(ValueError):
    """Error cuando una acción es inválida."""
    pass


# ════════════════════════════════════════════════════════════════════════════════
# ACTION CONTRACT CONSTANTS - INMUTABLES
# ════════════════════════════════════════════════════════════════════════════════

ACTION_CONTRACT_VERSION: Final[str] = "1.0.0"
VALID_ACTIONS: Final[Tuple[int, ...]] = (0, 1, 2)
ACTION_NAMES: Final[Dict[int, str]] = {0: "SELL", 1: "HOLD", 2: "BUY"}
ACTION_COUNT: Final[int] = 3
ACTION_PROBS_DIM: Final[int] = 3


# ════════════════════════════════════════════════════════════════════════════════
# MODEL OUTPUT CONTRACT
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelOutputContract:
    """
    Contrato formal del output del modelo.

    Contract ID: CTR-MODEL-OUTPUT-001

    El modelo produce:
    - action: int en {0, 1, 2}
    - confidence: float en [0.0, 1.0]
    - action_probs: array de 3 floats que suman 1.0
    """
    # Output constraints
    valid_actions: Tuple[int, ...] = VALID_ACTIONS
    confidence_min: float = 0.0
    confidence_max: float = 1.0
    action_probs_dim: int = ACTION_PROBS_DIM
    action_probs_sum_tolerance: float = 0.01

    def validate_action(self, action: int) -> List[str]:
        """Valida que action es válido."""
        errors = []
        if action not in self.valid_actions:
            errors.append(
                f"Invalid action {action}. Must be in {self.valid_actions}"
            )
        return errors

    def validate_confidence(self, confidence: float) -> List[str]:
        """Valida que confidence está en rango."""
        errors = []
        if not (self.confidence_min <= confidence <= self.confidence_max):
            errors.append(
                f"Confidence {confidence} out of range "
                f"[{self.confidence_min}, {self.confidence_max}]"
            )
        return errors

    def validate_action_probs(self, action_probs: list) -> List[str]:
        """Valida action_probs."""
        errors = []

        if action_probs is None:
            return errors  # Optional

        # Check dimension
        if len(action_probs) != self.action_probs_dim:
            errors.append(
                f"action_probs length {len(action_probs)} != {self.action_probs_dim}"
            )
            return errors

        # Check sum ≈ 1.0
        prob_sum = sum(action_probs)
        if abs(prob_sum - 1.0) > self.action_probs_sum_tolerance:
            errors.append(
                f"action_probs sum {prob_sum:.4f} != 1.0 "
                f"(tolerance: {self.action_probs_sum_tolerance})"
            )

        # Check all probs >= 0
        for i, p in enumerate(action_probs):
            if p < 0:
                errors.append(f"action_probs[{i}] = {p} is negative")

        return errors

    def validate_output(
        self,
        action: int,
        confidence: float,
        action_probs: list = None
    ) -> Tuple[bool, List[str]]:
        """
        Valida el output completo del modelo.

        Args:
            action: Acción predicha (0, 1, o 2)
            confidence: Confianza [0, 1]
            action_probs: Probabilidades por acción (opcional)

        Returns:
            (is_valid, errors)
        """
        errors = []
        errors.extend(self.validate_action(action))
        errors.extend(self.validate_confidence(confidence))
        errors.extend(self.validate_action_probs(action_probs))

        return len(errors) == 0, errors


# Singleton instance
MODEL_OUTPUT_CONTRACT: Final[ModelOutputContract] = ModelOutputContract()


def validate_model_output(
    action: int,
    confidence: float,
    action_probs: list = None,
    raise_on_error: bool = True
) -> bool:
    """
    Función de conveniencia para validar output del modelo.

    Args:
        action: Acción predicha (0, 1, o 2)
        confidence: Confianza de la predicción [0, 1]
        action_probs: Probabilidades por acción (opcional)
        raise_on_error: Si True, lanza excepción en error

    Returns:
        True si válido

    Raises:
        InvalidActionError si inválido y raise_on_error=True

    Example:
        >>> validate_model_output(action=0, confidence=0.85)
        True
        >>> validate_model_output(action=5, confidence=0.5)
        InvalidActionError: Model output validation failed
    """
    is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
        action, confidence, action_probs
    )

    if not is_valid and raise_on_error:
        raise InvalidActionError(
            f"Model output validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    return is_valid


# ════════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY ALIASES (DEPRECATED)
# ════════════════════════════════════════════════════════════════════════════════

# Estos aliases existen SOLO para migración gradual
# TODO: Remover después de migrar todos los usos (Semana 2)

import warnings

def _deprecated_alias(name: str, value: int) -> int:
    warnings.warn(
        f"{name} is deprecated. Use Action.{ACTION_NAMES[value]} instead.",
        DeprecationWarning,
        stacklevel=3
    )
    return value

# Old constants (deprecated)
ACTION_SELL: Final[int] = 0  # DEPRECATED: usar Action.SELL
ACTION_HOLD: Final[int] = 1  # DEPRECATED: usar Action.HOLD
ACTION_BUY: Final[int] = 2   # DEPRECATED: usar Action.BUY
```

### Solución 1B: Script de Migración Completo

**Archivo**: `scripts/migrate_action_enum.py`

```python
#!/usr/bin/env python
"""
Script para migrar TODOS los usos de Action enum al SSOT.

Este script:
1. Busca todas las definiciones locales de Action/TradingAction
2. Busca todos los usos de ACTION_HOLD, ACTION_BUY, ACTION_SELL
3. Reemplaza con imports del SSOT
4. Genera reporte de cambios

Uso:
    python scripts/migrate_action_enum.py --dry-run    # Preview changes
    python scripts/migrate_action_enum.py --apply      # Apply changes
"""
import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass


REPO_ROOT = Path(__file__).parent.parent


@dataclass
class Replacement:
    """Representa un reemplazo a realizar."""
    file: Path
    line_num: int
    old_text: str
    new_text: str
    pattern_name: str


# ════════════════════════════════════════════════════════════════════════════════
# PATRONES DE BÚSQUEDA Y REEMPLAZO
# ════════════════════════════════════════════════════════════════════════════════

PATTERNS: List[Tuple[str, str, str]] = [
    # (pattern, replacement, description)

    # Import replacements
    (
        r"from src\.core\.constants import (.*?ACTION_HOLD.*?|.*?ACTION_BUY.*?|.*?ACTION_SELL.*?)",
        "from src.core.contracts.action_contract import Action",
        "Replace old constants import"
    ),
    (
        r"from src\.training\.environments\.\w+ import TradingAction",
        "from src.core.contracts.action_contract import Action",
        "Replace TradingAction import"
    ),
    (
        r"from src\.shared\.schemas\.core import BackendAction",
        "from src.core.contracts.action_contract import Action",
        "Replace BackendAction import"
    ),

    # Usage replacements
    (r"\bACTION_HOLD\b", "Action.HOLD.value", "Replace ACTION_HOLD"),
    (r"\bACTION_BUY\b", "Action.BUY.value", "Replace ACTION_BUY"),
    (r"\bACTION_SELL\b", "Action.SELL.value", "Replace ACTION_SELL"),

    # TradingAction replacements
    (r"\bTradingAction\.LONG\b", "Action.BUY", "Replace TradingAction.LONG"),
    (r"\bTradingAction\.SHORT\b", "Action.SELL", "Replace TradingAction.SHORT"),
    (r"\bTradingAction\.HOLD\b", "Action.HOLD", "Replace TradingAction.HOLD"),

    # BackendAction replacements
    (r'\bBackendAction\.LONG\b', 'Action.BUY.to_signal()', "Replace BackendAction.LONG"),
    (r'\bBackendAction\.SHORT\b', 'Action.SELL.to_signal()', "Replace BackendAction.SHORT"),
    (r'\bBackendAction\.HOLD\b', 'Action.HOLD.to_signal()', "Replace BackendAction.HOLD"),

    # String comparisons
    (r"== ['\"]LONG['\"]", "== Action.BUY.to_signal()", "Replace LONG string comparison"),
    (r"== ['\"]SHORT['\"]", "== Action.SELL.to_signal()", "Replace SHORT string comparison"),
]

FILES_TO_SKIP = {
    "action_contract.py",       # Skip the SSOT itself
    "migrate_action_enum.py",   # Skip this script
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
}


def should_process_file(filepath: Path) -> bool:
    """Check if file should be processed."""
    if any(skip in str(filepath) for skip in FILES_TO_SKIP):
        return False
    return filepath.suffix == ".py"


def find_replacements(filepath: Path) -> List[Replacement]:
    """Find all replacements needed in a file."""
    replacements = []

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return []

    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        for pattern, replacement, desc in PATTERNS:
            if re.search(pattern, line):
                new_line = re.sub(pattern, replacement, line)
                if new_line != line:
                    replacements.append(Replacement(
                        file=filepath,
                        line_num=line_num,
                        old_text=line.strip(),
                        new_text=new_line.strip(),
                        pattern_name=desc
                    ))

    return replacements


def apply_replacements(filepath: Path, replacements: List[Replacement]) -> int:
    """Apply replacements to a file. Returns count of changes."""
    content = filepath.read_text(encoding="utf-8")
    original = content

    for pattern, replacement, _ in PATTERNS:
        content = re.sub(pattern, replacement, content)

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return len([r for r in replacements if r.file == filepath])

    return 0


def find_local_definitions() -> List[Tuple[Path, int, str]]:
    """Find local Action/TradingAction class definitions."""
    definitions = []

    patterns = [
        r"class TradingAction\(",
        r"class BackendAction\(",
        r"class SignalType\(",
        r"^ACTION_HOLD\s*=",
        r"^ACTION_BUY\s*=",
        r"^ACTION_SELL\s*=",
    ]

    for filepath in REPO_ROOT.rglob("*.py"):
        if not should_process_file(filepath):
            continue

        try:
            lines = filepath.read_text(encoding="utf-8").split("\n")
        except:
            continue

        for line_num, line in enumerate(lines, 1):
            for pattern in patterns:
                if re.search(pattern, line):
                    relative = filepath.relative_to(REPO_ROOT)
                    definitions.append((relative, line_num, line.strip()))

    return definitions


def main():
    parser = argparse.ArgumentParser(description="Migrate Action enum to SSOT")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        print("Usage: python migrate_action_enum.py [--dry-run | --apply]")
        sys.exit(1)

    print("=" * 80)
    print("ACTION ENUM MIGRATION TO SSOT")
    print("=" * 80)

    # Find local definitions
    print("\n📍 LOCAL DEFINITIONS TO REMOVE:")
    print("-" * 40)
    definitions = find_local_definitions()
    for path, line_num, line in definitions:
        print(f"  {path}:{line_num}")
        print(f"    {line[:70]}...")
    print(f"\nTotal: {len(definitions)} local definitions found")

    # Find replacements
    print("\n📝 REPLACEMENTS TO MAKE:")
    print("-" * 40)
    all_replacements: List[Replacement] = []

    for filepath in REPO_ROOT.rglob("*.py"):
        if should_process_file(filepath):
            replacements = find_replacements(filepath)
            all_replacements.extend(replacements)

    # Group by file
    by_file: Dict[Path, List[Replacement]] = {}
    for r in all_replacements:
        by_file.setdefault(r.file, []).append(r)

    for filepath, reps in sorted(by_file.items()):
        relative = filepath.relative_to(REPO_ROOT)
        print(f"\n  {relative}: {len(reps)} replacements")
        for r in reps[:3]:  # Show first 3
            print(f"    L{r.line_num}: {r.pattern_name}")
        if len(reps) > 3:
            print(f"    ... and {len(reps) - 3} more")

    print(f"\nTotal: {len(all_replacements)} replacements in {len(by_file)} files")

    # Apply if requested
    if args.apply:
        print("\n🔄 APPLYING CHANGES...")
        print("-" * 40)

        total_changes = 0
        for filepath in by_file:
            changes = apply_replacements(filepath, by_file[filepath])
            if changes > 0:
                relative = filepath.relative_to(REPO_ROOT)
                print(f"  ✅ {relative}: {changes} changes")
                total_changes += changes

        print(f"\n✅ Migration complete: {total_changes} changes applied")
        print("\n⚠️  NEXT STEPS:")
        print("  1. Run: pytest tests/ -v")
        print("  2. Review and remove local Action class definitions")
        print("  3. Update any missed usages manually")
    else:
        print("\n⚠️  DRY RUN: No changes made")
        print("  Run with --apply to apply changes")


if __name__ == "__main__":
    main()
```

### Solución 1C: Tests para Action Contract

**Archivo**: `tests/contracts/test_action_contract.py`

```python
"""
Tests para Action Contract SSOT.

Verifica:
1. Action enum tiene valores correctos
2. Conversiones funcionan correctamente
3. Validación detecta errores
4. NO existen otras definiciones de Action
"""
import pytest
import subprocess
from pathlib import Path

from src.core.contracts.action_contract import (
    Action,
    InvalidActionError,
    validate_model_output,
    MODEL_OUTPUT_CONTRACT,
    ACTION_COUNT,
    VALID_ACTIONS,
)


REPO_ROOT = Path(__file__).parent.parent.parent


class TestActionEnum:
    """Tests para Action enum."""

    def test_action_values_are_correct(self):
        """Verifica mapping SELL=0, HOLD=1, BUY=2."""
        assert Action.SELL.value == 0, "SELL debe ser 0 (argmax de P(SELL))"
        assert Action.HOLD.value == 1, "HOLD debe ser 1 (argmax de P(HOLD))"
        assert Action.BUY.value == 2, "BUY debe ser 2 (argmax de P(BUY))"

    def test_action_count_is_three(self):
        """Verifica que hay exactamente 3 acciones."""
        assert len(Action) == 3
        assert ACTION_COUNT == 3

    def test_from_int_valid_values(self):
        """from_int con 0, 1, 2 retorna Action correcta."""
        assert Action.from_int(0) == Action.SELL
        assert Action.from_int(1) == Action.HOLD
        assert Action.from_int(2) == Action.BUY

    def test_from_int_invalid_negative(self):
        """from_int con valor negativo lanza error."""
        with pytest.raises(InvalidActionError, match="Invalid action value: -1"):
            Action.from_int(-1)

    def test_from_int_invalid_too_large(self):
        """from_int con valor > 2 lanza error."""
        with pytest.raises(InvalidActionError, match="Invalid action value: 3"):
            Action.from_int(3)
        with pytest.raises(InvalidActionError):
            Action.from_int(5)
        with pytest.raises(InvalidActionError):
            Action.from_int(100)

    def test_from_string_sell_aliases(self):
        """from_string reconoce aliases de SELL."""
        for alias in ["sell", "SELL", "short", "SHORT", "s", "S", "-1"]:
            assert Action.from_string(alias) == Action.SELL, f"'{alias}' should map to SELL"

    def test_from_string_hold_aliases(self):
        """from_string reconoce aliases de HOLD."""
        for alias in ["hold", "HOLD", "flat", "FLAT", "h", "H", "0"]:
            assert Action.from_string(alias) == Action.HOLD, f"'{alias}' should map to HOLD"

    def test_from_string_buy_aliases(self):
        """from_string reconoce aliases de BUY."""
        for alias in ["buy", "BUY", "long", "LONG", "b", "B", "1"]:
            assert Action.from_string(alias) == Action.BUY, f"'{alias}' should map to BUY"

    def test_from_string_invalid(self):
        """from_string con string inválido lanza error."""
        with pytest.raises(InvalidActionError):
            Action.from_string("invalid")
        with pytest.raises(InvalidActionError):
            Action.from_string("CALL")  # Options terminology
        with pytest.raises(InvalidActionError):
            Action.from_string("")

    def test_from_model_output(self):
        """from_model_output selecciona action con mayor prob."""
        assert Action.from_model_output([0.8, 0.1, 0.1]) == Action.SELL
        assert Action.from_model_output([0.1, 0.8, 0.1]) == Action.HOLD
        assert Action.from_model_output([0.1, 0.1, 0.8]) == Action.BUY
        # Edge case: equal probs, returns first max
        assert Action.from_model_output([0.33, 0.34, 0.33]) == Action.HOLD

    def test_from_model_output_invalid_length(self):
        """from_model_output con array de length != 3 falla."""
        with pytest.raises(InvalidActionError, match="must have 3 values"):
            Action.from_model_output([0.5, 0.5])
        with pytest.raises(InvalidActionError):
            Action.from_model_output([0.25, 0.25, 0.25, 0.25])

    def test_to_signal(self):
        """to_signal retorna señal de trading correcta."""
        assert Action.SELL.to_signal() == "SHORT"
        assert Action.HOLD.to_signal() == "FLAT"
        assert Action.BUY.to_signal() == "LONG"

    def test_to_position(self):
        """to_position retorna -1, 0, 1."""
        assert Action.SELL.to_position() == -1
        assert Action.HOLD.to_position() == 0
        assert Action.BUY.to_position() == 1

    def test_is_entry(self):
        """is_entry True para BUY y SELL."""
        assert Action.BUY.is_entry is True
        assert Action.SELL.is_entry is True
        assert Action.HOLD.is_entry is False

    def test_direction(self):
        """direction property funciona."""
        assert Action.SELL.direction == -1
        assert Action.HOLD.direction == 0
        assert Action.BUY.direction == 1


class TestModelOutputContract:
    """Tests para ModelOutputContract."""

    def test_validate_valid_output(self):
        """Output válido pasa validación."""
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=0, confidence=0.85
        )
        assert is_valid
        assert len(errors) == 0

    def test_validate_invalid_action(self):
        """Action inválido falla validación."""
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=5, confidence=0.5
        )
        assert not is_valid
        assert any("Invalid action" in e for e in errors)

    def test_validate_negative_action(self):
        """Action negativo falla validación."""
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=-1, confidence=0.5
        )
        assert not is_valid

    def test_validate_confidence_negative(self):
        """Confidence negativo falla."""
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=1, confidence=-0.5
        )
        assert not is_valid
        assert any("out of range" in e for e in errors)

    def test_validate_confidence_above_one(self):
        """Confidence > 1 falla."""
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=1, confidence=1.5
        )
        assert not is_valid

    def test_validate_with_valid_probs(self):
        """action_probs válidas pasan."""
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=1, confidence=0.7, action_probs=[0.1, 0.7, 0.2]
        )
        assert is_valid
        assert len(errors) == 0

    def test_validate_probs_wrong_length(self):
        """action_probs con length != 3 falla."""
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=1, confidence=0.5, action_probs=[0.5, 0.5]
        )
        assert not is_valid
        assert any("length" in e for e in errors)

    def test_validate_probs_dont_sum_to_one(self):
        """action_probs que no suman 1.0 falla."""
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=1, confidence=0.5, action_probs=[0.5, 0.5, 0.5]
        )
        assert not is_valid
        assert any("sum" in e for e in errors)

    def test_validate_probs_negative(self):
        """action_probs con valor negativo falla."""
        is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(
            action=1, confidence=0.5, action_probs=[-0.1, 0.6, 0.5]
        )
        assert not is_valid
        assert any("negative" in e for e in errors)


class TestValidateModelOutput:
    """Tests para función validate_model_output."""

    def test_valid_returns_true(self):
        """Output válido retorna True."""
        assert validate_model_output(action=0, confidence=0.9) is True
        assert validate_model_output(action=1, confidence=0.5) is True
        assert validate_model_output(action=2, confidence=0.1) is True

    def test_invalid_raises_by_default(self):
        """Output inválido lanza excepción por defecto."""
        with pytest.raises(InvalidActionError, match="validation failed"):
            validate_model_output(action=5, confidence=0.5)

    def test_invalid_returns_false_when_raise_disabled(self):
        """Output inválido retorna False si raise_on_error=False."""
        result = validate_model_output(
            action=5, confidence=0.5, raise_on_error=False
        )
        assert result is False

    def test_boundary_confidence_values(self):
        """Confidence en boundaries (0.0, 1.0) es válido."""
        assert validate_model_output(action=1, confidence=0.0) is True
        assert validate_model_output(action=1, confidence=1.0) is True


class TestNoLocalActionDefinitions:
    """Tests que verifican que Action solo se define en SSOT."""

    def test_no_trading_action_class_definitions(self):
        """No debe haber class TradingAction fuera del SSOT."""
        result = subprocess.run(
            ["git", "grep", "-n", "class TradingAction("],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True
        )

        violations = []
        for line in result.stdout.strip().split("\n"):
            if line and "action_contract.py" not in line:
                violations.append(line.split(":")[0])

        assert not violations, (
            f"TradingAction class defined outside SSOT: {violations}\n"
            "Remove these and import from src.core.contracts.action_contract"
        )

    def test_no_action_constant_definitions(self):
        """No debe haber ACTION_HOLD/BUY/SELL definitions fuera de SSOT."""
        patterns = [
            "^ACTION_HOLD\\s*=",
            "^ACTION_BUY\\s*=",
            "^ACTION_SELL\\s*="
        ]

        violations = []
        for pattern in patterns:
            result = subprocess.run(
                ["git", "grep", "-En", pattern],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True
            )

            for line in result.stdout.strip().split("\n"):
                if line and "action_contract.py" not in line:
                    violations.append(line.split(":")[0])

        assert not violations, (
            f"ACTION_* constants defined outside SSOT: {set(violations)}\n"
            "Remove these and import from src.core.contracts.action_contract"
        )
```

### Checklist Día 1

```
□ 1.1 Crear directorio src/core/contracts/
□ 1.2 Crear src/core/contracts/__init__.py
□ 1.3 Crear src/core/contracts/action_contract.py (código arriba)
□ 1.4 Crear scripts/migrate_action_enum.py (código arriba)
□ 1.5 Ejecutar: python scripts/migrate_action_enum.py --dry-run
□ 1.6 Review output del dry-run
□ 1.7 Ejecutar: python scripts/migrate_action_enum.py --apply
□ 1.8 Crear tests/contracts/test_action_contract.py (código arriba)
□ 1.9 Ejecutar: pytest tests/contracts/test_action_contract.py -v
□ 1.10 Verificar que TODOS los tests pasan
□ 1.11 Commit: "feat: Add Action enum SSOT with ModelOutputContract"

VERIFICACIONES MANUALES:
□ grep -rn "class TradingAction" src/ muestra 0 resultados fuera de SSOT
□ grep -rn "^ACTION_HOLD\s*=" src/ muestra solo SSOT
□ Action.SELL.value == 0 (no 2!)
□ Action.BUY.value == 2 (no 1!)
```

---

## DIA 2: FEATURE_ORDER SSOT + FEATURE CONTRACT

### Problema Actual - Análisis Detallado

```python
# ══════════════════════════════════════════════════════════════════════════════
# PROBLEMA: 3 DEFINICIONES INCONSISTENTES DE FEATURE_ORDER
# ══════════════════════════════════════════════════════════════════════════════

# Definición 1: src/feature_store/adapters.py:71
FEATURE_ORDER = [
    "log_ret_5m", "log_ret_1h", "log_ret_4h",
    "rsi_9", "atr_pct", "adx_14",
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
    "brent_change_1d", "rate_spread", "usdmxn_change_1d",
    "position",
    "session_progress",  # ❌ NOMBRE INCORRECTO! Debe ser "time_normalized"
]

# Definición 2: src/features/feature_reader.py:29
EXPECTED_FEATURE_ORDER = [
    "log_ret_5m", "log_ret_1h", "log_ret_4h",
    "rsi_9", "atr_pct", "adx_14",
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
    "brent_change_1d", "rate_spread", "usdmxn_change_1d",
    # ❌ FALTAN: "position", "time_normalized"
]  # Solo 13 features!

# Definición 3: src/ml_workflow/training_pipeline.py
FEATURE_ORDER = [...]  # ❌ Otra definición local

# PROBLEMA CRITICO:
# - El modelo fue entrenado con 15 features en orden específico
# - Inference puede usar orden diferente → PREDICCIONES ERRONEAS
# - "session_progress" no existe, es "time_normalized"
```

### Solución 2A: Feature Contract SSOT

**Archivo**: `src/core/contracts/feature_contract.py`

```python
"""
FEATURE CONTRACT - Single Source of Truth
==========================================
Este es el ÚNICO lugar donde se define el orden y spec de features.
TODOS los componentes DEBEN importar de aquí.

Contract ID: CTR-FEATURE-001
Version: 2.0.0

ADVERTENCIA:
Si cambias FEATURE_ORDER:
1. Re-entrena TODOS los modelos
2. Actualiza norm_stats.json
3. Incrementa FEATURE_CONTRACT_VERSION
4. Actualiza tests
"""
from typing import Tuple, Dict, Final, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import hashlib
import json


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE ORDER - INMUTABLE
# ════════════════════════════════════════════════════════════════════════════════

FEATURE_ORDER: Final[Tuple[str, ...]] = (
    # Technical indicators (indices 0-5)
    "log_ret_5m",       # 0: 5-minute log return (z-normalized)
    "log_ret_1h",       # 1: 1-hour log return (z-normalized)
    "log_ret_4h",       # 2: 4-hour log return (z-normalized)
    "rsi_9",            # 3: RSI with 9 periods (z-normalized)
    "atr_pct",          # 4: ATR as percentage of price (z-normalized)
    "adx_14",           # 5: ADX with 14 periods (z-normalized)

    # Macro indicators (indices 6-12)
    "dxy_z",            # 6: DXY index z-score
    "dxy_change_1d",    # 7: DXY 1-day change (z-normalized)
    "vix_z",            # 8: VIX z-score
    "embi_z",           # 9: EMBI Colombia z-score
    "brent_change_1d",  # 10: Brent oil 1-day change (z-normalized)
    "rate_spread",      # 11: Interest rate spread (z-normalized)
    "usdmxn_change_1d", # 12: USD/MXN 1-day change (z-normalized)

    # State features (indices 13-14)
    "position",         # 13: Current position (-1=short, 0=flat, 1=long)
    "time_normalized",  # 14: Normalized time of day [0.0, 1.0]
)

# Constants
OBSERVATION_DIM: Final[int] = 15
FEATURE_CONTRACT_VERSION: Final[str] = "2.0.0"

# Compute feature order hash for contract verification
FEATURE_ORDER_HASH: Final[str] = hashlib.sha256(
    ",".join(FEATURE_ORDER).encode("utf-8")
).hexdigest()[:16]


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE METADATA
# ════════════════════════════════════════════════════════════════════════════════

class FeatureType(Enum):
    """Categoría de la feature."""
    TECHNICAL = "technical"  # Price-derived indicators
    MACRO = "macro"          # Macroeconomic indicators
    STATE = "state"          # Trading state
    TIME = "time"            # Time-based


class FeatureUnit(Enum):
    """Unidad/normalización de la feature."""
    ZSCORE = "z-score"           # (x - mean) / std
    PERCENTAGE = "percentage"     # 0-100
    NORMALIZED = "normalized"     # 0-1
    RAW = "raw"                  # No normalization


@dataclass(frozen=True)
class FeatureSpec:
    """Especificación completa de una feature."""
    name: str
    index: int
    type: FeatureType
    unit: FeatureUnit
    description: str
    clip_min: float = -5.0
    clip_max: float = 5.0
    requires_normalization: bool = True
    source: str = ""  # e.g., "L1_features", "trading_state"


# Complete feature specifications
FEATURE_SPECS: Final[Dict[str, FeatureSpec]] = {
    "log_ret_5m": FeatureSpec(
        name="log_ret_5m", index=0, type=FeatureType.TECHNICAL,
        unit=FeatureUnit.ZSCORE,
        description="5-minute log return, z-normalized",
        source="L1_features"
    ),
    "log_ret_1h": FeatureSpec(
        name="log_ret_1h", index=1, type=FeatureType.TECHNICAL,
        unit=FeatureUnit.ZSCORE,
        description="1-hour log return, z-normalized",
        source="L1_features"
    ),
    "log_ret_4h": FeatureSpec(
        name="log_ret_4h", index=2, type=FeatureType.TECHNICAL,
        unit=FeatureUnit.ZSCORE,
        description="4-hour log return, z-normalized",
        source="L1_features"
    ),
    "rsi_9": FeatureSpec(
        name="rsi_9", index=3, type=FeatureType.TECHNICAL,
        unit=FeatureUnit.ZSCORE,
        description="RSI(9), z-normalized",
        clip_min=-3.0, clip_max=3.0,
        source="L1_features"
    ),
    "atr_pct": FeatureSpec(
        name="atr_pct", index=4, type=FeatureType.TECHNICAL,
        unit=FeatureUnit.ZSCORE,
        description="ATR as % of price, z-normalized",
        source="L1_features"
    ),
    "adx_14": FeatureSpec(
        name="adx_14", index=5, type=FeatureType.TECHNICAL,
        unit=FeatureUnit.ZSCORE,
        description="ADX(14), z-normalized",
        source="L1_features"
    ),
    "dxy_z": FeatureSpec(
        name="dxy_z", index=6, type=FeatureType.MACRO,
        unit=FeatureUnit.ZSCORE,
        description="DXY index z-score",
        source="L0_macro"
    ),
    "dxy_change_1d": FeatureSpec(
        name="dxy_change_1d", index=7, type=FeatureType.MACRO,
        unit=FeatureUnit.ZSCORE,
        description="DXY 1-day change, z-normalized",
        source="L0_macro"
    ),
    "vix_z": FeatureSpec(
        name="vix_z", index=8, type=FeatureType.MACRO,
        unit=FeatureUnit.ZSCORE,
        description="VIX z-score",
        source="L0_macro"
    ),
    "embi_z": FeatureSpec(
        name="embi_z", index=9, type=FeatureType.MACRO,
        unit=FeatureUnit.ZSCORE,
        description="EMBI Colombia z-score",
        source="L0_macro"
    ),
    "brent_change_1d": FeatureSpec(
        name="brent_change_1d", index=10, type=FeatureType.MACRO,
        unit=FeatureUnit.ZSCORE,
        description="Brent oil 1-day change, z-normalized",
        source="L0_macro"
    ),
    "rate_spread": FeatureSpec(
        name="rate_spread", index=11, type=FeatureType.MACRO,
        unit=FeatureUnit.ZSCORE,
        description="Interest rate spread (COL - US), z-normalized",
        source="L0_macro"
    ),
    "usdmxn_change_1d": FeatureSpec(
        name="usdmxn_change_1d", index=12, type=FeatureType.MACRO,
        unit=FeatureUnit.ZSCORE,
        description="USD/MXN 1-day change, z-normalized",
        source="L0_macro"
    ),
    "position": FeatureSpec(
        name="position", index=13, type=FeatureType.STATE,
        unit=FeatureUnit.RAW,
        description="Current position: -1 (short), 0 (flat), 1 (long)",
        clip_min=-1.0, clip_max=1.0, requires_normalization=False,
        source="trading_state"
    ),
    "time_normalized": FeatureSpec(
        name="time_normalized", index=14, type=FeatureType.TIME,
        unit=FeatureUnit.NORMALIZED,
        description="Time of day normalized [0=market open, 1=market close]",
        clip_min=0.0, clip_max=1.0, requires_normalization=False,
        source="trading_state"
    ),
}


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE CONTRACT CLASS
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FeatureContract:
    """
    Contrato formal de features - SSOT.

    Define:
    - Orden exacto de features (immutable tuple)
    - Dimensión de observación (15)
    - Rango de clipping post-normalización
    - Tipo de datos (float32)
    - Método de normalización (z-score)
    """
    version: str = FEATURE_CONTRACT_VERSION
    observation_dim: int = OBSERVATION_DIM
    feature_order: Tuple[str, ...] = FEATURE_ORDER
    feature_order_hash: str = FEATURE_ORDER_HASH
    clip_range: Tuple[float, float] = (-5.0, 5.0)
    dtype: str = "float32"
    normalization_method: str = "zscore"

    def get_feature_index(self, name: str) -> int:
        """
        Obtiene el índice de una feature por nombre.

        Args:
            name: Nombre de la feature

        Returns:
            Índice en el array de observación

        Raises:
            FeatureContractError si feature no existe
        """
        if name not in self.feature_order:
            raise FeatureContractError(
                f"Unknown feature: '{name}'. "
                f"Valid features: {list(self.feature_order)}"
            )
        return self.feature_order.index(name)

    def get_feature_spec(self, name: str) -> FeatureSpec:
        """Obtiene la especificación de una feature."""
        if name not in FEATURE_SPECS:
            raise FeatureContractError(f"No spec for feature: {name}")
        return FEATURE_SPECS[name]

    def get_features_by_type(self, ftype: FeatureType) -> List[str]:
        """Obtiene lista de features de un tipo específico."""
        return [
            name for name, spec in FEATURE_SPECS.items()
            if spec.type == ftype
        ]

    def get_normalizable_features(self) -> List[str]:
        """Obtiene features que requieren normalización."""
        return [
            name for name, spec in FEATURE_SPECS.items()
            if spec.requires_normalization
        ]

    def validate_observation(
        self,
        obs: np.ndarray,
        strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Valida que una observación cumple el contrato.

        Args:
            obs: Array numpy con la observación
            strict: Si True, verifica clip range también

        Returns:
            (is_valid, errors)
        """
        errors = []

        # Check shape
        expected_shape = (self.observation_dim,)
        if obs.shape != expected_shape:
            errors.append(
                f"Invalid shape: expected {expected_shape}, got {obs.shape}"
            )
            return False, errors  # Can't continue validation

        # Check dtype
        if obs.dtype != np.float32:
            errors.append(
                f"Invalid dtype: expected float32, got {obs.dtype}"
            )

        # Check for NaN
        nan_mask = np.isnan(obs)
        if np.any(nan_mask):
            nan_indices = np.where(nan_mask)[0].tolist()
            nan_features = [self.feature_order[i] for i in nan_indices]
            errors.append(
                f"NaN values at indices {nan_indices}: {nan_features}"
            )

        # Check for Inf
        inf_mask = np.isinf(obs)
        if np.any(inf_mask):
            inf_indices = np.where(inf_mask)[0].tolist()
            inf_features = [self.feature_order[i] for i in inf_indices]
            errors.append(
                f"Inf values at indices {inf_indices}: {inf_features}"
            )

        # Check clip range per feature (if strict)
        if strict and not np.any(nan_mask) and not np.any(inf_mask):
            for i, (name, value) in enumerate(zip(self.feature_order, obs)):
                spec = FEATURE_SPECS[name]
                if value < spec.clip_min or value > spec.clip_max:
                    errors.append(
                        f"Feature '{name}' (idx {i}): value {value:.4f} "
                        f"outside range [{spec.clip_min}, {spec.clip_max}]"
                    )

        return len(errors) == 0, errors

    def validate_feature_dict(
        self,
        features: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Valida un diccionario de features.

        Args:
            features: Dict {feature_name: value}

        Returns:
            (is_valid, errors)
        """
        errors = []

        # Check all required features present
        missing = set(self.feature_order) - set(features.keys())
        if missing:
            errors.append(f"Missing features: {sorted(missing)}")

        # Check no extra features
        extra = set(features.keys()) - set(self.feature_order)
        if extra:
            errors.append(f"Extra features not in contract: {sorted(extra)}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el contrato a diccionario."""
        return {
            "version": self.version,
            "observation_dim": self.observation_dim,
            "feature_order": list(self.feature_order),
            "feature_order_hash": self.feature_order_hash,
            "clip_range": list(self.clip_range),
            "dtype": self.dtype,
            "normalization_method": self.normalization_method,
        }

    def dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convierte dict de features a array ordenado.

        Args:
            features: Dict {feature_name: value}

        Returns:
            Array numpy float32 en orden correcto
        """
        # Validate first
        is_valid, errors = self.validate_feature_dict(features)
        if not is_valid:
            raise FeatureContractError(
                f"Cannot convert invalid feature dict:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        return np.array(
            [features[name] for name in self.feature_order],
            dtype=np.float32
        )


class FeatureContractError(ValueError):
    """Error de violación del contrato de features."""
    pass


# ════════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ════════════════════════════════════════════════════════════════════════════════

FEATURE_CONTRACT: Final[FeatureContract] = FeatureContract()


# ════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def validate_feature_vector(
    obs: np.ndarray,
    raise_on_error: bool = True,
    strict: bool = True
) -> bool:
    """
    Valida que un vector de features cumple el contrato.

    Args:
        obs: Array numpy con la observación
        raise_on_error: Si True, lanza excepción en error
        strict: Si True, verifica clip range

    Returns:
        True si válido

    Raises:
        FeatureContractError si inválido y raise_on_error=True
    """
    is_valid, errors = FEATURE_CONTRACT.validate_observation(obs, strict=strict)

    if not is_valid and raise_on_error:
        raise FeatureContractError(
            f"Feature vector validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    return is_valid


def get_feature_index(name: str) -> int:
    """Obtiene el índice de una feature por nombre."""
    return FEATURE_CONTRACT.get_feature_index(name)


def get_feature_names() -> List[str]:
    """Retorna lista de nombres de features en orden."""
    return list(FEATURE_ORDER)


def features_dict_to_array(features: Dict[str, float]) -> np.ndarray:
    """Convierte dict de features a array ordenado."""
    return FEATURE_CONTRACT.dict_to_array(features)
```

### Solución 2B: Script de Migración de FEATURE_ORDER

**Archivo**: `scripts/migrate_feature_order.py`

```python
#!/usr/bin/env python
"""
Script para migrar FEATURE_ORDER al SSOT.

Busca y reporta:
1. Definiciones locales de FEATURE_ORDER
2. Uso incorrecto de "session_progress"
3. Arrays de features con length != 15
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def main():
    print("=" * 80)
    print("FEATURE_ORDER MIGRATION ANALYSIS")
    print("=" * 80)

    # 1. Find local FEATURE_ORDER definitions
    print("\n📍 LOCAL FEATURE_ORDER DEFINITIONS:")
    print("-" * 40)

    result = subprocess.run(
        ["git", "grep", "-n", "FEATURE_ORDER\\s*=\\s*\\["],
        cwd=REPO_ROOT, capture_output=True, text=True
    )

    for line in result.stdout.strip().split("\n"):
        if line and "feature_contract.py" not in line:
            print(f"  ❌ {line.split(':')[0]}:{line.split(':')[1]}")

    # 2. Find "session_progress" usage (wrong name)
    print("\n🔍 'session_progress' USAGE (should be 'time_normalized'):")
    print("-" * 40)

    result = subprocess.run(
        ["git", "grep", "-n", "session_progress"],
        cwd=REPO_ROOT, capture_output=True, text=True
    )

    for line in result.stdout.strip().split("\n"):
        if line:
            print(f"  ⚠️  {line}")

    # 3. Find potential feature array issues
    print("\n📐 OBSERVATION_DIM DEFINITIONS:")
    print("-" * 40)

    for pattern in ["OBSERVATION_DIM", "obs_dim", "observation_dim"]:
        result = subprocess.run(
            ["git", "grep", "-n", f"{pattern}\\s*="],
            cwd=REPO_ROOT, capture_output=True, text=True
        )
        for line in result.stdout.strip().split("\n"):
            if line and "feature_contract.py" not in line:
                print(f"  {line.split(':')[0]}:{line.split(':')[1]}")

    print("\n" + "=" * 80)
    print("ACTIONS REQUIRED:")
    print("=" * 80)
    print("""
1. Remove all local FEATURE_ORDER definitions
2. Replace imports with:
   from src.core.contracts.feature_contract import FEATURE_ORDER, OBSERVATION_DIM

3. Replace 'session_progress' with 'time_normalized':
   - src/feature_store/adapters.py:71

4. Fix feature arrays with != 15 features:
   - src/features/feature_reader.py:29 (add position, time_normalized)
""")


if __name__ == "__main__":
    main()
```

### Solución 2C: Tests para Feature Contract

**Archivo**: `tests/contracts/test_feature_contract.py`

```python
"""
Tests para Feature Contract SSOT.
"""
import pytest
import numpy as np
import subprocess
from pathlib import Path

from src.core.contracts.feature_contract import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FEATURE_CONTRACT,
    FEATURE_SPECS,
    FEATURE_ORDER_HASH,
    FeatureType,
    validate_feature_vector,
    get_feature_index,
    features_dict_to_array,
    FeatureContractError,
)


REPO_ROOT = Path(__file__).parent.parent.parent


class TestFeatureOrderSSoT:
    """Tests para FEATURE_ORDER como SSOT."""

    def test_feature_order_has_15_elements(self):
        """FEATURE_ORDER tiene exactamente 15 elementos."""
        assert len(FEATURE_ORDER) == 15
        assert len(FEATURE_ORDER) == OBSERVATION_DIM

    def test_feature_order_is_immutable(self):
        """FEATURE_ORDER es tuple (immutable)."""
        assert isinstance(FEATURE_ORDER, tuple)

    def test_all_expected_features_present(self):
        """Todas las features esperadas están presentes."""
        expected = {
            # Technical
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            # Macro
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d",
            # State
            "position", "time_normalized"
        }
        assert set(FEATURE_ORDER) == expected

    def test_no_session_progress_in_features(self):
        """'session_progress' NO debe estar en features."""
        assert "session_progress" not in FEATURE_ORDER
        assert "time_normalized" in FEATURE_ORDER

    def test_time_normalized_is_last(self):
        """'time_normalized' debe ser la última feature (index 14)."""
        assert FEATURE_ORDER[14] == "time_normalized"
        assert get_feature_index("time_normalized") == 14

    def test_position_is_second_to_last(self):
        """'position' debe ser index 13."""
        assert FEATURE_ORDER[13] == "position"
        assert get_feature_index("position") == 13

    def test_feature_order_hash_is_deterministic(self):
        """El hash de FEATURE_ORDER es determinístico."""
        import hashlib
        expected_hash = hashlib.sha256(
            ",".join(FEATURE_ORDER).encode("utf-8")
        ).hexdigest()[:16]
        assert FEATURE_ORDER_HASH == expected_hash


class TestFeatureSpecs:
    """Tests para especificaciones de features."""

    def test_all_features_have_specs(self):
        """Todas las features tienen especificación."""
        for name in FEATURE_ORDER:
            assert name in FEATURE_SPECS, f"Missing spec for {name}"

    def test_spec_indices_match_order(self):
        """Índices en specs coinciden con FEATURE_ORDER."""
        for i, name in enumerate(FEATURE_ORDER):
            assert FEATURE_SPECS[name].index == i, (
                f"{name}: spec.index={FEATURE_SPECS[name].index} != order={i}"
            )

    def test_technical_features_count(self):
        """Hay 6 features técnicas."""
        technical = FEATURE_CONTRACT.get_features_by_type(FeatureType.TECHNICAL)
        assert len(technical) == 6

    def test_macro_features_count(self):
        """Hay 7 features macro."""
        macro = FEATURE_CONTRACT.get_features_by_type(FeatureType.MACRO)
        assert len(macro) == 7

    def test_normalizable_features(self):
        """Features normalizables no incluyen position ni time_normalized."""
        normalizable = FEATURE_CONTRACT.get_normalizable_features()
        assert "position" not in normalizable
        assert "time_normalized" not in normalizable
        assert len(normalizable) == 13


class TestFeatureValidation:
    """Tests para validación de features."""

    @pytest.fixture
    def valid_observation(self):
        """Observación válida de 15 features."""
        return np.zeros(15, dtype=np.float32)

    def test_validate_valid_observation(self, valid_observation):
        """Observación válida pasa."""
        assert validate_feature_vector(valid_observation)

    def test_validate_wrong_shape_14(self):
        """Shape (14,) falla."""
        obs = np.zeros(14, dtype=np.float32)
        with pytest.raises(FeatureContractError, match="Invalid shape"):
            validate_feature_vector(obs)

    def test_validate_wrong_shape_16(self):
        """Shape (16,) falla."""
        obs = np.zeros(16, dtype=np.float32)
        with pytest.raises(FeatureContractError, match="Invalid shape"):
            validate_feature_vector(obs)

    def test_validate_detects_nan(self):
        """NaN es detectado."""
        obs = np.zeros(15, dtype=np.float32)
        obs[5] = np.nan
        with pytest.raises(FeatureContractError, match="NaN"):
            validate_feature_vector(obs)

    def test_validate_detects_inf(self):
        """Inf es detectado."""
        obs = np.zeros(15, dtype=np.float32)
        obs[3] = np.inf
        with pytest.raises(FeatureContractError, match="Inf"):
            validate_feature_vector(obs)

    def test_validate_detects_out_of_range(self):
        """Valor fuera de clip range es detectado."""
        obs = np.zeros(15, dtype=np.float32)
        obs[0] = 10.0  # log_ret_5m clip range is [-5, 5]
        with pytest.raises(FeatureContractError, match="outside range"):
            validate_feature_vector(obs, strict=True)

    def test_validate_position_range(self):
        """position debe estar en [-1, 1]."""
        obs = np.zeros(15, dtype=np.float32)
        obs[13] = 2.0  # position = 2 is invalid
        with pytest.raises(FeatureContractError, match="position"):
            validate_feature_vector(obs, strict=True)

    def test_validate_time_normalized_range(self):
        """time_normalized debe estar en [0, 1]."""
        obs = np.zeros(15, dtype=np.float32)
        obs[14] = 1.5  # time_normalized > 1 is invalid
        with pytest.raises(FeatureContractError, match="time_normalized"):
            validate_feature_vector(obs, strict=True)


class TestFeatureDict:
    """Tests para conversión dict → array."""

    def test_dict_to_array_valid(self):
        """Dict válido se convierte correctamente."""
        features = {name: 0.0 for name in FEATURE_ORDER}
        result = features_dict_to_array(features)
        assert result.shape == (15,)
        assert result.dtype == np.float32

    def test_dict_to_array_preserves_order(self):
        """Orden de features se preserva en array."""
        features = {name: float(i) for i, name in enumerate(FEATURE_ORDER)}
        result = features_dict_to_array(features)
        for i, name in enumerate(FEATURE_ORDER):
            assert result[i] == float(i), f"{name} at wrong index"

    def test_dict_missing_feature_fails(self):
        """Dict con feature faltante falla."""
        features = {name: 0.0 for name in FEATURE_ORDER}
        del features["position"]
        with pytest.raises(FeatureContractError, match="Missing"):
            features_dict_to_array(features)

    def test_dict_extra_feature_fails(self):
        """Dict con feature extra falla."""
        features = {name: 0.0 for name in FEATURE_ORDER}
        features["extra_feature"] = 0.0
        with pytest.raises(FeatureContractError, match="Extra"):
            features_dict_to_array(features)


class TestNoLocalDefinitions:
    """Tests que verifican que FEATURE_ORDER solo está en SSOT."""

    def test_no_local_feature_order_definitions(self):
        """FEATURE_ORDER no se define en otros archivos."""
        result = subprocess.run(
            ["git", "grep", "-l", "^FEATURE_ORDER\\s*="],
            cwd=REPO_ROOT, capture_output=True, text=True
        )

        violations = []
        for line in result.stdout.strip().split("\n"):
            if line and "feature_contract.py" not in line:
                violations.append(line)

        assert not violations, (
            f"FEATURE_ORDER defined outside SSOT: {violations}\n"
            "Import from src.core.contracts.feature_contract instead"
        )

    def test_no_session_progress_anywhere(self):
        """'session_progress' no debe existir en el código."""
        result = subprocess.run(
            ["git", "grep", "-l", "session_progress"],
            cwd=REPO_ROOT, capture_output=True, text=True
        )

        violations = [
            line for line in result.stdout.strip().split("\n")
            if line and "test_" not in line and ".md" not in line
        ]

        assert not violations, (
            f"'session_progress' found in: {violations}\n"
            "Replace with 'time_normalized'"
        )
```

### Checklist Día 2

```
□ 2.1 Crear src/core/contracts/feature_contract.py (código arriba)
□ 2.2 Crear scripts/migrate_feature_order.py (código arriba)
□ 2.3 Ejecutar: python scripts/migrate_feature_order.py
□ 2.4 Fix src/feature_store/adapters.py: session_progress → time_normalized
□ 2.5 Fix src/features/feature_reader.py: agregar position, time_normalized
□ 2.6 Buscar y eliminar otras definiciones locales de FEATURE_ORDER
□ 2.7 Crear tests/contracts/test_feature_contract.py (código arriba)
□ 2.8 Ejecutar: pytest tests/contracts/test_feature_contract.py -v
□ 2.9 Verificar: grep -rn "session_progress" src/ muestra 0 resultados
□ 2.10 Commit: "feat: Add FEATURE_ORDER SSOT with FeatureContract"

VERIFICACIONES:
□ FEATURE_ORDER tiene exactamente 15 elementos
□ "time_normalized" en index 14 (no "session_progress")
□ Todos los archivos importan FEATURE_ORDER del SSOT
```

---

## DIAS 3-5: MODEL INPUT, MLFLOW SIGNATURE, METADATA

### Día 3: Model Input Contract

**Archivo**: `src/core/contracts/model_input_contract.py`

(Ver implementación completa en documento original - este día se enfoca en ObservationValidator que valida TODOS los inputs antes de pasar al modelo)

### Día 4: MLflow Signature + TrainingRunContract

**Archivos clave**:
- `src/training/mlflow_signature.py` - Crea signature attached al modelo
- `src/core/contracts/training_run_contract.py` - Valida params/metrics obligatorios

### Día 5: Model Metadata Contract + Tests

**Archivos clave**:
- `src/core/contracts/model_metadata_contract.py` - Metadata Pydantic con validación
- `tests/unit/test_model_contracts.py` - Suite completa

---

# ════════════════════════════════════════════════════════════════════════════════
# TESTS E2E DE CONTRATOS
# ════════════════════════════════════════════════════════════════════════════════

**Archivo**: `tests/integration/test_contracts_e2e.py`

```python
"""
End-to-End Contract Tests.

Verifica que todos los contratos funcionan juntos correctamente.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from src.core.contracts.action_contract import (
    Action, validate_model_output, InvalidActionError
)
from src.core.contracts.feature_contract import (
    FEATURE_ORDER, OBSERVATION_DIM, FEATURE_CONTRACT,
    validate_feature_vector, features_dict_to_array
)
from src.core.contracts.model_input_contract import (
    MODEL_INPUT_CONTRACT, ObservationValidator
)


class TestTrainingInferenceParit:
    """Tests que verifican paridad entre training e inference."""

    @pytest.fixture
    def realistic_observation(self):
        """Observación realista basada en datos históricos."""
        return np.array([
            0.001,   # log_ret_5m
            0.003,   # log_ret_1h
            0.01,    # log_ret_4h
            -0.5,    # rsi_9 (z-score)
            0.2,     # atr_pct
            0.1,     # adx_14
            -0.3,    # dxy_z
            0.001,   # dxy_change_1d
            0.5,     # vix_z
            0.2,     # embi_z
            0.002,   # brent_change_1d
            0.1,     # rate_spread
            -0.001,  # usdmxn_change_1d
            0.0,     # position (flat)
            0.5,     # time_normalized (midday)
        ], dtype=np.float32)

    def test_observation_passes_all_contracts(self, realistic_observation):
        """Observación realista pasa todos los contratos."""
        # Feature contract
        assert validate_feature_vector(realistic_observation)

        # Input contract
        MODEL_INPUT_CONTRACT.validate(realistic_observation)

        # Observation validator
        validator = ObservationValidator(strict_mode=True)
        validated = validator.validate_and_prepare(realistic_observation)
        assert validated is not None

    def test_feature_order_matches_training(self, realistic_observation):
        """Orden de features es consistente con training."""
        # Verificar que podemos convertir dict → array → mismo orden
        feature_dict = dict(zip(FEATURE_ORDER, realistic_observation))
        reconstructed = features_dict_to_array(feature_dict)

        np.testing.assert_array_equal(realistic_observation, reconstructed)

    def test_model_output_valid_for_all_actions(self):
        """Model output válido para todas las acciones."""
        for action in [0, 1, 2]:
            assert validate_model_output(
                action=action,
                confidence=0.8,
                action_probs=[0.1, 0.1, 0.8] if action == 2 else [0.8, 0.1, 0.1]
            )

    def test_action_roundtrip(self):
        """Conversión Action int → enum → int es consistente."""
        for i in [0, 1, 2]:
            action = Action.from_int(i)
            assert action.value == i

            # String roundtrip
            signal = action.to_signal()
            recovered = Action.from_string(signal)
            assert recovered == action


class TestContractInteraction:
    """Tests de interacción entre contratos."""

    def test_invalid_observation_fails_early(self):
        """Observación inválida falla en feature contract, no llega a model."""
        # Wrong shape
        obs_wrong_shape = np.zeros(14, dtype=np.float32)

        with pytest.raises(Exception):
            validate_feature_vector(obs_wrong_shape)

    def test_validator_catches_nan_before_model(self):
        """ObservationValidator detecta NaN antes de enviar al modelo."""
        obs = np.zeros(15, dtype=np.float32)
        obs[5] = np.nan

        validator = ObservationValidator(strict_mode=True)

        with pytest.raises(Exception, match="NaN"):
            validator.validate_and_prepare(obs)

    def test_action_contract_validates_model_output(self):
        """Action contract valida output del modelo."""
        # Simular output inválido del modelo
        with pytest.raises(InvalidActionError):
            validate_model_output(action=5, confidence=0.8)


class TestContractVersions:
    """Tests de versionado de contratos."""

    def test_feature_contract_version(self):
        """Feature contract tiene versión."""
        assert FEATURE_CONTRACT.version == "2.0.0"

    def test_feature_order_hash_exists(self):
        """Feature order tiene hash para verificación."""
        assert len(FEATURE_CONTRACT.feature_order_hash) == 16

    def test_contracts_can_be_serialized(self):
        """Contratos pueden serializarse para logging."""
        contract_dict = FEATURE_CONTRACT.to_dict()

        # Verificar que es serializable a JSON
        json_str = json.dumps(contract_dict)
        assert len(json_str) > 0

        # Verificar campos clave
        loaded = json.loads(json_str)
        assert loaded["version"] == "2.0.0"
        assert loaded["observation_dim"] == 15
        assert len(loaded["feature_order"]) == 15
```

---

# METRICAS Y PROGRESO ESPERADO

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           METRICAS DE PROGRESO                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   INICIO:          63% (146 Pass, 71 Parcial, 27 Fail)                             │
│                                                                                      │
│   SEMANA 1:                                                                          │
│   ├── Día 1: 63% → 67%  (+4%)  Action SSOT + Output Contract                        │
│   ├── Día 2: 67% → 71%  (+4%)  Feature SSOT + Feature Contract                      │
│   ├── Día 3: 71% → 74%  (+3%)  Model Input Contract                                 │
│   ├── Día 4: 74% → 76%  (+2%)  MLflow Signature + Training Contract                 │
│   └── Día 5: 76% → 78%  (+2%)  Model Metadata + Tests                               │
│                                                                                      │
│   SEMANA 2:                                                                          │
│   ├── Día 6: 78% → 80%  (+2%)  Norm Stats Contract                                  │
│   ├── Día 7: 80% → 82%  (+2%)  L0→L1 Price Validation                               │
│   ├── Día 8: 82% → 84%  (+2%)  L1→L5 Hash Validation                                │
│   ├── Día 9: 84% → 86%  (+2%)  API Contracts                                        │
│   └── Día 10: 86% → 88% (+2%)  Database Contracts                                   │
│                                                                                      │
│   SEMANA 3:                                                                          │
│   ├── Día 11: 88% → 91% (+3%)  Frontend-Backend Sync                                │
│   ├── Día 12: 91% → 93% (+2%)  WebSocket Contracts                                  │
│   ├── Día 13: 93% → 95% (+2%)  JSONB + Alembic                                      │
│   ├── Día 14: 95% → 98% (+3%)  E2E Tests                                            │
│   └── Día 15: 98% → 100%(+2%)  Re-Audit + Documentation                             │
│                                                                                      │
│   FIN:             100% (250 Pass, 0 Parcial, 0 Fail)                               │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

# CHECKLIST FINAL DE VERIFICACION

```
SEMANA 1 - CONTRATOS DE MODELO:
═══════════════════════════════════════════════════════════════════════════════════
□ src/core/contracts/action_contract.py creado
□ src/core/contracts/feature_contract.py creado
□ src/core/contracts/model_input_contract.py creado
□ src/core/contracts/model_output_contract.py creado
□ src/core/contracts/training_run_contract.py creado
□ src/core/contracts/model_metadata_contract.py creado
□ tests/contracts/test_action_contract.py - 100% pass
□ tests/contracts/test_feature_contract.py - 100% pass
□ tests/unit/test_model_contracts.py - 100% pass
□ Action.SELL == 0, Action.HOLD == 1, Action.BUY == 2
□ FEATURE_ORDER tiene 15 elementos, "time_normalized" (no session_progress)
□ NO existen definiciones locales de Action o FEATURE_ORDER
□ MLflow signature attached al modelo (no como artifact separado)

SEMANA 2 - CONTRATOS DE DATOS Y API:
═══════════════════════════════════════════════════════════════════════════════════
□ config/schemas/norm_stats.schema.json creado
□ src/core/contracts/norm_stats_contract.py creado
□ L0→L1 price validation (3000 <= USD/COP <= 6000)
□ L1→L5 norm_stats_hash validation
□ API kill-switch requiere confirmation: "CONFIRM_KILL_SWITCH"
□ Database models en SQLAlchemy
□ Alembic configurado para migrations

SEMANA 3 - SYNC Y POLISH:
═══════════════════════════════════════════════════════════════════════════════════
□ CI job compara Zod schemas vs Pydantic models
□ WebSocket contracts completos con tests
□ JSONB CHECK constraint para features_snapshot
□ tests/integration/test_contracts_e2e.py - 100% pass
□ Re-audit: 250/250 preguntas PASS
□ Documentación de contratos completa
□ Ready for production
```

---

# ARCHIVOS CREADOS POR EL PLAN

| Archivo | Propósito |
|---------|-----------|
| `src/core/contracts/__init__.py` | Package init |
| `src/core/contracts/action_contract.py` | Action enum SSOT |
| `src/core/contracts/feature_contract.py` | Feature contract SSOT |
| `src/core/contracts/model_input_contract.py` | Input validation |
| `src/core/contracts/model_output_contract.py` | Output validation |
| `src/core/contracts/training_run_contract.py` | MLflow logging contract |
| `src/core/contracts/model_metadata_contract.py` | Model metadata |
| `src/core/contracts/norm_stats_contract.py` | Norm stats validation |
| `src/training/mlflow_signature.py` | MLflow signature helper |
| `config/schemas/norm_stats.schema.json` | JSON Schema |
| `scripts/migrate_action_enum.py` | Migration script |
| `scripts/migrate_feature_order.py` | Migration script |
| `tests/contracts/test_action_contract.py` | Action tests |
| `tests/contracts/test_feature_contract.py` | Feature tests |
| `tests/unit/test_model_contracts.py` | Model contract tests |
| `tests/integration/test_contracts_e2e.py` | E2E tests |

---

*Plan de Remediación de Contratos v2.0 - ROBUSTECIDO*
*250 preguntas → 100% compliance*
*3 semanas de implementación*
*Última actualización: 2026-01-17*
