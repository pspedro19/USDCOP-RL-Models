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
    """
    SELL = 0
    HOLD = 1
    BUY = 2

    @classmethod
    def from_int(cls, value: int) -> "Action":
        """Convierte int a Action con validación estricta."""
        if value not in (0, 1, 2):
            raise InvalidActionError(
                f"Invalid action value: {value}. "
                f"Must be 0 (SELL), 1 (HOLD), or 2 (BUY)."
            )
        return cls(value)

    @classmethod
    def from_string(cls, value: str) -> "Action":
        """Convierte string a Action (case-insensitive)."""
        mapping = {
            "sell": cls.SELL, "short": cls.SELL, "s": cls.SELL, "-1": cls.SELL,
            "hold": cls.HOLD, "flat": cls.HOLD, "h": cls.HOLD, "0": cls.HOLD,
            "buy": cls.BUY, "long": cls.BUY, "b": cls.BUY, "1": cls.BUY,
        }
        normalized = value.lower().strip()
        if normalized not in mapping:
            raise InvalidActionError(f"Invalid action string: '{value}'")
        return mapping[normalized]

    @classmethod
    def from_model_output(cls, logits_or_probs: list) -> "Action":
        """Convierte output del modelo (logits o probabilidades) a Action."""
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
        return {self.SELL: "SHORT", self.HOLD: "FLAT", self.BUY: "LONG"}[self]

    def to_position(self) -> int:
        """Convierte a posición numérica: -1 (short), 0 (flat), 1 (long)."""
        return {self.SELL: -1, self.HOLD: 0, self.BUY: 1}[self]

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


# Constants
ACTION_CONTRACT_VERSION: Final[str] = "1.0.0"
VALID_ACTIONS: Final[Tuple[int, ...]] = (0, 1, 2)
ACTION_NAMES: Final[Dict[int, str]] = {0: "SELL", 1: "HOLD", 2: "BUY"}
ACTION_COUNT: Final[int] = 3
ACTION_PROBS_DIM: Final[int] = 3


@dataclass(frozen=True)
class ModelOutputContract:
    """Contrato formal del output del modelo."""
    valid_actions: Tuple[int, ...] = VALID_ACTIONS
    confidence_min: float = 0.0
    confidence_max: float = 1.0
    action_probs_dim: int = ACTION_PROBS_DIM
    action_probs_sum_tolerance: float = 0.01

    def validate_output(
        self, action: int, confidence: float, action_probs: list = None
    ) -> Tuple[bool, List[str]]:
        """Valida el output completo del modelo."""
        errors = []

        if action not in self.valid_actions:
            errors.append(f"Invalid action {action}. Must be in {self.valid_actions}")

        if not (self.confidence_min <= confidence <= self.confidence_max):
            errors.append(f"Confidence {confidence} out of range [{self.confidence_min}, {self.confidence_max}]")

        if action_probs is not None:
            if len(action_probs) != self.action_probs_dim:
                errors.append(f"action_probs length {len(action_probs)} != {self.action_probs_dim}")
            elif abs(sum(action_probs) - 1.0) > self.action_probs_sum_tolerance:
                errors.append(f"action_probs sum {sum(action_probs):.4f} != 1.0")
            else:
                for i, p in enumerate(action_probs):
                    if p < 0:
                        errors.append(f"action_probs[{i}] = {p} is negative")

        return len(errors) == 0, errors


MODEL_OUTPUT_CONTRACT: Final[ModelOutputContract] = ModelOutputContract()


def validate_model_output(
    action: int,
    confidence: float,
    action_probs: list = None,
    raise_on_error: bool = True
) -> bool:
    """Función de conveniencia para validar output del modelo."""
    is_valid, errors = MODEL_OUTPUT_CONTRACT.validate_output(action, confidence, action_probs)

    if not is_valid and raise_on_error:
        raise InvalidActionError(
            f"Model output validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return is_valid


# Backward compatibility aliases (deprecated)
ACTION_SELL: Final[int] = 0
ACTION_HOLD: Final[int] = 1
ACTION_BUY: Final[int] = 2
