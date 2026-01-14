"""
USD/COP RL Trading System - Custom Exception Hierarchy
=======================================================

Jerarquia completa de excepciones para el sistema de trading RL.

PRINCIPIOS DE DISENO:
1. Fail Fast: Validacion temprana en constructores y metodos publicos
2. Granularidad: Excepciones especificas para cada tipo de error
3. Contexto: Mensajes descriptivos con valores actuales vs esperados
4. Recuperabilidad: Distincion entre errores recuperables y fatales
5. Trazabilidad: Encadenamiento de excepciones (from e)

JERARQUIA:
    TradingSystemError (base)
    |
    +-- ConfigurationError
    |   +-- InvalidConfigValueError
    |   +-- MissingConfigKeyError
    |   +-- ConfigSchemaError
    |
    +-- DataError
    |   +-- DataNotFoundError
    |   +-- DataValidationError
    |   +-- MissingColumnsError
    |   +-- InsufficientDataError
    |   +-- DataQualityError
    |   +-- TimeSeriesGapError
    |
    +-- EnvironmentError
    |   +-- InvalidActionError
    |   +-- InvalidObservationError
    |   +-- EpisodeTerminationError
    |   +-- StateInconsistencyError
    |
    +-- ModelError
    |   +-- ModelNotFoundError
    |   +-- ModelLoadError
    |   +-- ModelInferenceError
    |   +-- EnsembleError
    |
    +-- ValidationError
    |   +-- CVSplitError
    |   +-- MetricsCalculationError
    |   +-- AcceptanceCriteriaError
    |
    +-- RiskManagementError
    |   +-- DrawdownLimitError
    |   +-- PositionLimitError
    |   +-- KillSwitchTriggeredError
    |
    +-- RewardError
    |   +-- RewardCalculationError
    |   +-- CurriculumPhaseError

Author: Claude Code
Version: 1.0.0
Date: 2025-12-26
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import traceback


# =============================================================================
# ERROR SEVERITY LEVELS
# =============================================================================

class ErrorSeverity(Enum):
    """Niveles de severidad para categorizar errores."""
    WARNING = "WARNING"       # Puede continuar con degradacion
    ERROR = "ERROR"           # Debe detenerse pero es recuperable
    CRITICAL = "CRITICAL"     # Debe detenerse inmediatamente
    FATAL = "FATAL"           # Corrupcion de estado, irrecuperable


class ErrorCategory(Enum):
    """Categorias de errores para agrupacion y metricas."""
    CONFIGURATION = "CONFIGURATION"
    DATA = "DATA"
    ENVIRONMENT = "ENVIRONMENT"
    MODEL = "MODEL"
    VALIDATION = "VALIDATION"
    RISK = "RISK"
    REWARD = "REWARD"
    SYSTEM = "SYSTEM"


# =============================================================================
# BASE EXCEPTION
# =============================================================================

class TradingSystemError(Exception):
    """
    Excepcion base para todo el sistema de trading RL USD/COP.

    Todas las excepciones custom heredan de esta clase, permitiendo:
    - Captura generica: except TradingSystemError
    - Contexto enriquecido: detalles, sugerencias, valores
    - Logging estructurado: severity, category, timestamp

    Attributes:
        message: Descripcion del error
        severity: Nivel de severidad
        category: Categoria del error
        details: Diccionario con contexto adicional
        suggestion: Sugerencia para resolver el error
        original_exception: Excepcion original (si aplica)

    Example:
        >>> raise TradingSystemError(
        ...     "Fallo en inicializacion",
        ...     severity=ErrorSeverity.ERROR,
        ...     details={'component': 'Environment'},
        ...     suggestion="Verificar configuracion del dataset"
        ... )
    """

    default_severity: ErrorSeverity = ErrorSeverity.ERROR
    default_category: ErrorCategory = ErrorCategory.SYSTEM

    def __init__(
        self,
        message: str,
        *,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.message = message
        self.severity = severity or self.default_severity
        self.category = category or self.default_category
        self.details = details or {}
        self.suggestion = suggestion
        self.original_exception = original_exception

        # Construir mensaje completo
        full_message = self._build_message()
        super().__init__(full_message)

    def _build_message(self) -> str:
        """Construir mensaje formateado."""
        parts = [f"[{self.severity.value}] {self.message}"]

        if self.details:
            details_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            parts.append(f"  Details: {details_str}")

        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")

        if self.original_exception:
            parts.append(f"  Caused by: {type(self.original_exception).__name__}: {self.original_exception}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para logging/serialization."""
        return {
            'error_type': type(self).__name__,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'details': self.details,
            'suggestion': self.suggestion,
            'original_exception': str(self.original_exception) if self.original_exception else None,
        }

    @property
    def is_recoverable(self) -> bool:
        """Indica si el error es potencialmente recuperable."""
        return self.severity in (ErrorSeverity.WARNING, ErrorSeverity.ERROR)

    @property
    def requires_immediate_stop(self) -> bool:
        """Indica si requiere detencion inmediata."""
        return self.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.FATAL)


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(TradingSystemError):
    """Error base para problemas de configuracion."""

    default_severity = ErrorSeverity.ERROR
    default_category = ErrorCategory.CONFIGURATION


class InvalidConfigValueError(ConfigurationError):
    """
    Valor de configuracion invalido.

    Se lanza cuando un parametro tiene un valor fuera del rango permitido
    o un tipo incorrecto.

    Example:
        >>> raise InvalidConfigValueError(
        ...     param_name='learning_rate',
        ...     value=-0.01,
        ...     expected='float > 0',
        ...     suggestion="Usar valor entre 1e-5 y 1e-2"
        ... )
    """

    def __init__(
        self,
        param_name: str,
        value: Any,
        expected: str,
        *,
        config_section: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        details = {
            'param_name': param_name,
            'actual_value': value,
            'actual_type': type(value).__name__,
            'expected': expected,
        }
        if config_section:
            details['config_section'] = config_section

        message = f"Valor invalido para '{param_name}': {value!r} (esperado: {expected})"

        super().__init__(
            message,
            details=details,
            suggestion=suggestion or f"Verificar valor de '{param_name}' en configuracion",
        )

        self.param_name = param_name
        self.value = value
        self.expected = expected


class MissingConfigKeyError(ConfigurationError):
    """
    Clave de configuracion faltante.

    Se lanza cuando una configuracion requerida no esta presente.
    """

    def __init__(
        self,
        key: str,
        *,
        config_section: Optional[str] = None,
        available_keys: Optional[List[str]] = None,
        default_value: Optional[Any] = None,
    ):
        details = {'missing_key': key}
        if config_section:
            details['config_section'] = config_section
        if available_keys:
            details['available_keys'] = available_keys[:10]  # Limitar para no sobrecargar

        message = f"Clave de configuracion faltante: '{key}'"

        suggestion = f"Agregar '{key}' a la configuracion"
        if default_value is not None:
            suggestion += f" (default sugerido: {default_value!r})"

        super().__init__(message, details=details, suggestion=suggestion)

        self.key = key
        self.default_value = default_value


class ConfigSchemaError(ConfigurationError):
    """
    Error de esquema de configuracion.

    Se lanza cuando la estructura de la configuracion no es valida.
    """

    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        message: str,
        *,
        schema_path: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
    ):
        details = {}
        if schema_path:
            details['schema_path'] = schema_path
        if validation_errors:
            details['validation_errors'] = validation_errors

        super().__init__(
            message,
            details=details,
            suggestion="Revisar esquema de configuracion en documentation",
        )


# =============================================================================
# DATA ERRORS
# =============================================================================

class DataError(TradingSystemError):
    """Error base para problemas de datos."""

    default_severity = ErrorSeverity.ERROR
    default_category = ErrorCategory.DATA


class DataNotFoundError(DataError):
    """
    Datos no encontrados.

    Se lanza cuando un archivo o dataset no existe.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        data_type: str = "dataset",
        searched_paths: Optional[List[str]] = None,
    ):
        path = Path(path)
        details = {
            'path': str(path),
            'data_type': data_type,
            'path_exists': path.parent.exists(),
        }
        if searched_paths:
            details['searched_paths'] = searched_paths

        message = f"{data_type.capitalize()} no encontrado: {path}"

        super().__init__(
            message,
            details=details,
            suggestion=f"Verificar ruta del {data_type} y permisos de lectura",
        )

        self.path = path


class MissingColumnsError(DataError):
    """
    Columnas faltantes en DataFrame.

    Se lanza cuando columnas requeridas no estan presentes.

    Example:
        >>> raise MissingColumnsError(
        ...     missing=['close', 'volume'],
        ...     available=['open', 'high', 'low'],
        ...     context="Carga de datos OHLCV"
        ... )
    """

    def __init__(
        self,
        missing: List[str],
        *,
        available: Optional[List[str]] = None,
        context: Optional[str] = None,
    ):
        details = {
            'missing_columns': missing,
            'n_missing': len(missing),
        }
        if available:
            details['available_columns'] = available[:20]
            details['n_available'] = len(available)
        if context:
            details['context'] = context

        message = f"Columnas faltantes: {missing}"

        super().__init__(
            message,
            details=details,
            suggestion="Verificar que el dataset contiene todas las columnas requeridas",
        )

        self.missing = missing
        self.available = available


class InsufficientDataError(DataError):
    """
    Datos insuficientes para operacion.

    Se lanza cuando no hay suficientes filas/barras para una operacion.
    """

    def __init__(
        self,
        actual: int,
        required: int,
        *,
        context: str = "procesamiento",
        data_source: Optional[str] = None,
    ):
        details = {
            'actual_rows': actual,
            'required_rows': required,
            'deficit': required - actual,
            'context': context,
        }
        if data_source:
            details['data_source'] = data_source

        message = f"Datos insuficientes para {context}: {actual} filas (minimo: {required})"

        super().__init__(
            message,
            details=details,
            suggestion="Aumentar datos de entrada o reducir requerimientos minimos",
        )

        self.actual = actual
        self.required = required


class DataValidationError(DataError):
    """
    Error de validacion de datos.

    Se lanza cuando los datos no pasan validaciones de calidad.
    """

    def __init__(
        self,
        message: str,
        *,
        validation_type: str = "general",
        failed_checks: Optional[List[str]] = None,
        statistics: Optional[Dict[str, float]] = None,
    ):
        details = {'validation_type': validation_type}
        if failed_checks:
            details['failed_checks'] = failed_checks
        if statistics:
            details['statistics'] = statistics

        super().__init__(
            message,
            details=details,
            suggestion="Revisar preprocesamiento de datos y calidad del dataset",
        )


class DataQualityError(DataError):
    """
    Error de calidad de datos.

    Se lanza cuando hay problemas de calidad como NaN, outliers extremos, etc.
    """

    def __init__(
        self,
        issue_type: str,
        *,
        column: Optional[str] = None,
        affected_rows: Optional[int] = None,
        affected_pct: Optional[float] = None,
        threshold: Optional[float] = None,
    ):
        details = {'issue_type': issue_type}
        if column:
            details['column'] = column
        if affected_rows is not None:
            details['affected_rows'] = affected_rows
        if affected_pct is not None:
            details['affected_pct'] = f"{affected_pct:.2%}"
        if threshold is not None:
            details['threshold'] = threshold

        message = f"Problema de calidad de datos: {issue_type}"
        if column:
            message += f" en columna '{column}'"

        super().__init__(
            message,
            details=details,
            suggestion="Aplicar limpieza de datos o ajustar thresholds de calidad",
        )


class TimeSeriesGapError(DataError):
    """
    Gaps en serie temporal.

    Se lanza cuando hay discontinuidades en los datos temporales.
    """

    def __init__(
        self,
        n_gaps: int,
        *,
        largest_gap_bars: Optional[int] = None,
        gap_locations: Optional[List[int]] = None,
        expected_frequency: Optional[str] = None,
    ):
        details = {
            'n_gaps': n_gaps,
        }
        if largest_gap_bars is not None:
            details['largest_gap_bars'] = largest_gap_bars
        if gap_locations:
            details['gap_locations'] = gap_locations[:10]
        if expected_frequency:
            details['expected_frequency'] = expected_frequency

        message = f"Detectados {n_gaps} gaps en serie temporal"

        super().__init__(
            message,
            details=details,
            suggestion="Aplicar forward fill o eliminar periodos con gaps",
        )


# =============================================================================
# ENVIRONMENT ERRORS
# =============================================================================

class EnvironmentError(TradingSystemError):
    """Error base para problemas del environment de trading."""

    default_severity = ErrorSeverity.ERROR
    default_category = ErrorCategory.ENVIRONMENT


class InvalidActionError(EnvironmentError):
    """
    Accion invalida en environment.

    Se lanza cuando una accion esta fuera del espacio de acciones.
    """

    def __init__(
        self,
        action: Any,
        *,
        action_space_low: float = -1.0,
        action_space_high: float = 1.0,
        step: Optional[int] = None,
    ):
        details = {
            'action': action,
            'action_space': f"[{action_space_low}, {action_space_high}]",
        }
        if step is not None:
            details['step'] = step

        message = f"Accion invalida: {action} (espacio: [{action_space_low}, {action_space_high}])"

        super().__init__(
            message,
            details=details,
            suggestion="Aplicar clipping de acciones antes de paso a environment",
        )


class InvalidObservationError(EnvironmentError):
    """
    Observacion invalida generada por environment.

    Se lanza cuando una observacion contiene NaN, Inf, o valores fuera de rango.
    """

    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        issue: str,
        *,
        observation_shape: Optional[Tuple[int, ...]] = None,
        problematic_indices: Optional[List[int]] = None,
        step: Optional[int] = None,
    ):
        details = {'issue': issue}
        if observation_shape:
            details['observation_shape'] = observation_shape
        if problematic_indices:
            details['problematic_indices'] = problematic_indices
        if step is not None:
            details['step'] = step

        message = f"Observacion invalida: {issue}"

        super().__init__(
            message,
            details=details,
            suggestion="Revisar calculo de features y normalizacion de observaciones",
        )


class EpisodeTerminationError(EnvironmentError):
    """
    Error de terminacion de episodio.

    Se lanza cuando un episodio termina por condicion anomala.
    """

    def __init__(
        self,
        reason: str,
        *,
        step: int,
        portfolio_value: Optional[float] = None,
        drawdown: Optional[float] = None,
    ):
        details = {
            'termination_reason': reason,
            'step': step,
        }
        if portfolio_value is not None:
            details['portfolio_value'] = portfolio_value
        if drawdown is not None:
            details['drawdown'] = f"{drawdown:.2%}"

        message = f"Episodio terminado en step {step}: {reason}"

        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            details=details,
        )


class StateInconsistencyError(EnvironmentError):
    """
    Inconsistencia de estado en environment.

    Se lanza cuando el estado interno del environment es inconsistente.
    """

    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        message: str,
        *,
        expected_state: Optional[Dict[str, Any]] = None,
        actual_state: Optional[Dict[str, Any]] = None,
    ):
        details = {}
        if expected_state:
            details['expected'] = expected_state
        if actual_state:
            details['actual'] = actual_state

        super().__init__(
            message,
            details=details,
            suggestion="Reiniciar environment y verificar logica de step()",
        )


# =============================================================================
# MODEL ERRORS
# =============================================================================

class ModelError(TradingSystemError):
    """Error base para problemas de modelos."""

    default_severity = ErrorSeverity.ERROR
    default_category = ErrorCategory.MODEL


class ModelNotFoundError(ModelError):
    """
    Modelo no encontrado.

    Se lanza cuando un archivo de modelo no existe.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        model_type: str = "PPO",
    ):
        path = Path(model_path)
        details = {
            'model_path': str(path),
            'model_type': model_type,
            'path_exists': path.exists(),
            'parent_exists': path.parent.exists(),
        }

        message = f"Modelo {model_type} no encontrado: {path}"

        super().__init__(
            message,
            details=details,
            suggestion="Verificar ruta del modelo o entrenar uno nuevo",
        )


class ModelLoadError(ModelError):
    """
    Error al cargar modelo.

    Se lanza cuando un modelo existe pero no puede cargarse.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        reason: str,
        *,
        original_exception: Optional[Exception] = None,
    ):
        details = {
            'model_path': str(model_path),
            'reason': reason,
        }

        message = f"Error cargando modelo: {reason}"

        super().__init__(
            message,
            details=details,
            original_exception=original_exception,
            suggestion="Verificar compatibilidad de version de modelo y dependencias",
        )


class ModelInferenceError(ModelError):
    """
    Error durante inferencia de modelo.

    Se lanza cuando model.predict() falla.
    """

    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        message: str,
        *,
        observation_shape: Optional[Tuple[int, ...]] = None,
        expected_shape: Optional[Tuple[int, ...]] = None,
        original_exception: Optional[Exception] = None,
    ):
        details = {}
        if observation_shape:
            details['observation_shape'] = observation_shape
        if expected_shape:
            details['expected_shape'] = expected_shape

        super().__init__(
            message,
            details=details,
            original_exception=original_exception,
            suggestion="Verificar que observation space coincide con el usado en training",
        )


class EnsembleError(ModelError):
    """
    Error en ensemble de modelos.

    Se lanza cuando hay problemas con la combinacion de modelos.
    """

    def __init__(
        self,
        message: str,
        *,
        n_models: Optional[int] = None,
        weights: Optional[List[float]] = None,
        failed_model_idx: Optional[int] = None,
    ):
        details = {}
        if n_models is not None:
            details['n_models'] = n_models
        if weights:
            details['weights'] = weights
        if failed_model_idx is not None:
            details['failed_model_idx'] = failed_model_idx

        super().__init__(
            message,
            details=details,
            suggestion="Verificar que todos los modelos del ensemble son compatibles",
        )


# =============================================================================
# VALIDATION ERRORS
# =============================================================================

class ValidationError(TradingSystemError):
    """Error base para problemas de validacion."""

    default_severity = ErrorSeverity.ERROR
    default_category = ErrorCategory.VALIDATION


class CVSplitError(ValidationError):
    """
    Error en split de cross-validation.

    Se lanza cuando los splits de CV no son validos.
    """

    def __init__(
        self,
        message: str,
        *,
        n_samples: Optional[int] = None,
        n_splits: Optional[int] = None,
        embargo_bars: Optional[int] = None,
        min_train_size: Optional[int] = None,
        min_test_size: Optional[int] = None,
    ):
        details = {}
        if n_samples is not None:
            details['n_samples'] = n_samples
        if n_splits is not None:
            details['n_splits'] = n_splits
        if embargo_bars is not None:
            details['embargo_bars'] = embargo_bars
        if min_train_size is not None:
            details['min_train_size'] = min_train_size
        if min_test_size is not None:
            details['min_test_size'] = min_test_size

        super().__init__(
            message,
            details=details,
            suggestion="Reducir n_splits o embargo_bars, o aumentar datos",
        )


class MetricsCalculationError(ValidationError):
    """
    Error calculando metricas.

    Se lanza cuando el calculo de metricas falla.
    """

    def __init__(
        self,
        metric_name: str,
        reason: str,
        *,
        input_length: Optional[int] = None,
        original_exception: Optional[Exception] = None,
    ):
        details = {
            'metric': metric_name,
            'reason': reason,
        }
        if input_length is not None:
            details['input_length'] = input_length

        message = f"Error calculando metrica '{metric_name}': {reason}"

        super().__init__(
            message,
            details=details,
            original_exception=original_exception,
        )


class AcceptanceCriteriaError(ValidationError):
    """
    Error de criterios de aceptacion.

    Se lanza cuando un modelo no cumple criterios minimos.
    """

    default_severity = ErrorSeverity.WARNING

    def __init__(
        self,
        failures: List[str],
        *,
        metrics: Optional[Dict[str, float]] = None,
        criteria: Optional[Dict[str, float]] = None,
    ):
        details = {
            'n_failures': len(failures),
            'failures': failures,
        }
        if metrics:
            details['actual_metrics'] = metrics
        if criteria:
            details['criteria'] = criteria

        message = f"Modelo no cumple {len(failures)} criterio(s) de aceptacion"

        super().__init__(
            message,
            details=details,
            suggestion="Revisar hiperparametros o aumentar tiempo de training",
        )

        self.failures = failures


# =============================================================================
# RISK MANAGEMENT ERRORS
# =============================================================================

class RiskManagementError(TradingSystemError):
    """Error base para problemas de gestion de riesgo."""

    default_severity = ErrorSeverity.CRITICAL
    default_category = ErrorCategory.RISK


class DrawdownLimitError(RiskManagementError):
    """
    Limite de drawdown excedido.

    Se lanza cuando el drawdown supera limites configurados.
    """

    def __init__(
        self,
        current_drawdown: float,
        limit: float,
        *,
        portfolio_value: Optional[float] = None,
        peak_value: Optional[float] = None,
        action_taken: str = "trading paused",
    ):
        details = {
            'current_drawdown': f"{current_drawdown:.2%}",
            'limit': f"{limit:.2%}",
            'action_taken': action_taken,
        }
        if portfolio_value is not None:
            details['portfolio_value'] = portfolio_value
        if peak_value is not None:
            details['peak_value'] = peak_value

        message = f"Drawdown {current_drawdown:.2%} excede limite {limit:.2%}"

        super().__init__(
            message,
            details=details,
            suggestion="Revisar risk limits o evaluar condiciones de mercado",
        )


class PositionLimitError(RiskManagementError):
    """
    Limite de posicion excedido.

    Se lanza cuando una posicion excede limites permitidos.
    """

    def __init__(
        self,
        requested_position: float,
        max_position: float,
        *,
        current_regime: Optional[str] = None,
        position_multiplier: Optional[float] = None,
    ):
        details = {
            'requested_position': requested_position,
            'max_position': max_position,
        }
        if current_regime:
            details['current_regime'] = current_regime
        if position_multiplier is not None:
            details['position_multiplier'] = position_multiplier

        message = f"Posicion {requested_position} excede limite {max_position}"

        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            details=details,
        )


class KillSwitchTriggeredError(RiskManagementError):
    """
    Kill switch activado.

    Se lanza cuando un kill switch automatico detiene el trading.
    """

    default_severity = ErrorSeverity.CRITICAL

    def __init__(
        self,
        trigger_reason: str,
        *,
        trigger_value: Optional[float] = None,
        threshold: Optional[float] = None,
        action: str = "trading halted",
    ):
        details = {
            'trigger_reason': trigger_reason,
            'action': action,
        }
        if trigger_value is not None:
            details['trigger_value'] = trigger_value
        if threshold is not None:
            details['threshold'] = threshold

        message = f"Kill switch activado: {trigger_reason}"

        super().__init__(
            message,
            details=details,
            suggestion="Revision manual requerida antes de reanudar trading",
        )


# =============================================================================
# REWARD ERRORS
# =============================================================================

class RewardError(TradingSystemError):
    """Error base para problemas de reward function."""

    default_severity = ErrorSeverity.ERROR
    default_category = ErrorCategory.REWARD


class RewardCalculationError(RewardError):
    """
    Error en calculo de reward.

    Se lanza cuando el reward no puede calcularse.
    """

    def __init__(
        self,
        message: str,
        *,
        component: Optional[str] = None,
        input_values: Optional[Dict[str, float]] = None,
        original_exception: Optional[Exception] = None,
    ):
        details = {}
        if component:
            details['component'] = component
        if input_values:
            details['input_values'] = input_values

        super().__init__(
            message,
            details=details,
            original_exception=original_exception,
        )


class CurriculumPhaseError(RewardError):
    """
    Error de fase de curriculum.

    Se lanza cuando hay problemas con la transicion de fases.
    """

    def __init__(
        self,
        current_phase: str,
        expected_phase: str,
        *,
        progress: Optional[float] = None,
        phase_boundaries: Optional[Tuple[float, float]] = None,
    ):
        details = {
            'current_phase': current_phase,
            'expected_phase': expected_phase,
        }
        if progress is not None:
            details['progress'] = f"{progress:.2%}"
        if phase_boundaries:
            details['phase_boundaries'] = phase_boundaries

        message = f"Fase de curriculum inconsistente: {current_phase} (esperado: {expected_phase})"

        super().__init__(
            message,
            details=details,
        )


# =============================================================================
# VALIDATION UTILITIES (FAIL FAST)
# =============================================================================

class FailFastValidator:
    """
    Utilidad para validacion temprana con Fail Fast pattern.

    Agrupa validaciones y lanza excepciones apropiadas.

    Example:
        >>> validator = FailFastValidator()
        >>> validator.require_positive('learning_rate', lr, config_section='ppo')
        >>> validator.require_columns(df, ['close', 'volume'], context='OHLCV')
        >>> validator.require_min_rows(df, 10000, context='training')
        >>> validator.validate()  # Lanza si hubo errores
    """

    def __init__(self, raise_immediately: bool = True):
        """
        Args:
            raise_immediately: Si True, lanza excepcion inmediatamente.
                               Si False, acumula errores para validar al final.
        """
        self.raise_immediately = raise_immediately
        self.errors: List[TradingSystemError] = []

    def _handle_error(self, error: TradingSystemError):
        """Manejar error segun configuracion."""
        if self.raise_immediately:
            raise error
        self.errors.append(error)

    def validate(self):
        """Lanzar excepcion si hay errores acumulados."""
        if self.errors:
            if len(self.errors) == 1:
                raise self.errors[0]
            else:
                messages = [str(e) for e in self.errors]
                raise TradingSystemError(
                    f"Multiples errores de validacion ({len(self.errors)})",
                    severity=ErrorSeverity.ERROR,
                    details={'errors': messages},
                )

    # === CONFIGURATION VALIDATIONS ===

    def require_positive(
        self,
        name: str,
        value: float,
        *,
        config_section: Optional[str] = None,
    ):
        """Validar que valor es positivo."""
        if value <= 0:
            self._handle_error(InvalidConfigValueError(
                param_name=name,
                value=value,
                expected='float > 0',
                config_section=config_section,
            ))

    def require_range(
        self,
        name: str,
        value: float,
        min_val: float,
        max_val: float,
        *,
        config_section: Optional[str] = None,
    ):
        """Validar que valor esta en rango."""
        if not (min_val <= value <= max_val):
            self._handle_error(InvalidConfigValueError(
                param_name=name,
                value=value,
                expected=f'{min_val} <= x <= {max_val}',
                config_section=config_section,
            ))

    def require_type(
        self,
        name: str,
        value: Any,
        expected_type: Type,
        *,
        config_section: Optional[str] = None,
    ):
        """Validar tipo de valor."""
        if not isinstance(value, expected_type):
            self._handle_error(InvalidConfigValueError(
                param_name=name,
                value=value,
                expected=expected_type.__name__,
                config_section=config_section,
            ))

    def require_in(
        self,
        name: str,
        value: Any,
        allowed: List[Any],
        *,
        config_section: Optional[str] = None,
    ):
        """Validar que valor esta en lista permitida."""
        if value not in allowed:
            self._handle_error(InvalidConfigValueError(
                param_name=name,
                value=value,
                expected=f'one of {allowed}',
                config_section=config_section,
            ))

    # === DATA VALIDATIONS ===

    def require_columns(
        self,
        df,
        required: List[str],
        *,
        context: Optional[str] = None,
    ):
        """Validar que DataFrame tiene columnas requeridas."""
        missing = [c for c in required if c not in df.columns]
        if missing:
            self._handle_error(MissingColumnsError(
                missing=missing,
                available=list(df.columns),
                context=context,
            ))

    def require_min_rows(
        self,
        df,
        min_rows: int,
        *,
        context: str = "procesamiento",
    ):
        """Validar minimo de filas."""
        if len(df) < min_rows:
            self._handle_error(InsufficientDataError(
                actual=len(df),
                required=min_rows,
                context=context,
            ))

    def require_no_nulls(
        self,
        df,
        columns: Optional[List[str]] = None,
        *,
        max_null_pct: float = 0.0,
    ):
        """Validar que no hay nulls (o estan bajo threshold)."""
        check_cols = columns or df.columns
        for col in check_cols:
            if col in df.columns:
                null_pct = df[col].isnull().mean()
                if null_pct > max_null_pct:
                    self._handle_error(DataQualityError(
                        issue_type='excessive_nulls',
                        column=col,
                        affected_pct=null_pct,
                        threshold=max_null_pct,
                    ))

    def require_file_exists(
        self,
        path: Union[str, Path],
        *,
        data_type: str = "file",
    ):
        """Validar que archivo existe."""
        path = Path(path)
        if not path.exists():
            self._handle_error(DataNotFoundError(
                path=path,
                data_type=data_type,
            ))


# =============================================================================
# EXCEPTION HANDLER DECORATOR
# =============================================================================

def handle_exceptions(
    *catch_types: Type[Exception],
    reraise_as: Optional[Type[TradingSystemError]] = None,
    default_message: str = "Error inesperado",
    log_traceback: bool = True,
):
    """
    Decorator para manejar excepciones de forma consistente.

    Example:
        >>> @handle_exceptions(ValueError, KeyError, reraise_as=ConfigurationError)
        ... def load_config(path):
        ...     ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except TradingSystemError:
                # Re-raise nuestras excepciones directamente
                raise
            except catch_types as e:
                if reraise_as:
                    raise reraise_as(
                        f"{default_message}: {e}",
                        original_exception=e,
                    ) from e
                raise
            except Exception as e:
                if log_traceback:
                    tb = traceback.format_exc()
                    raise TradingSystemError(
                        f"{default_message}: {e}",
                        severity=ErrorSeverity.CRITICAL,
                        details={'traceback': tb},
                        original_exception=e,
                    ) from e
                raise
        return wrapper
    return decorator


# =============================================================================
# TESTING AND EXAMPLES
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("EXCEPTION HIERARCHY TEST - USD/COP RL Trading System")
    print("=" * 70)

    # Test 1: Configuration Error
    print("\n1. InvalidConfigValueError:")
    print("-" * 50)
    try:
        raise InvalidConfigValueError(
            param_name='learning_rate',
            value=-0.001,
            expected='float > 0',
            config_section='ppo',
            suggestion="Usar valor entre 1e-5 y 1e-2",
        )
    except TradingSystemError as e:
        print(e)
        print(f"\nRecoverable: {e.is_recoverable}")

    # Test 2: Data Error
    print("\n2. MissingColumnsError:")
    print("-" * 50)
    try:
        raise MissingColumnsError(
            missing=['close', 'volume'],
            available=['open', 'high', 'low', 'timestamp'],
            context="Carga de datos OHLCV",
        )
    except DataError as e:
        print(e)

    # Test 3: Risk Error
    print("\n3. DrawdownLimitError:")
    print("-" * 50)
    try:
        raise DrawdownLimitError(
            current_drawdown=0.12,
            limit=0.10,
            portfolio_value=8800,
            peak_value=10000,
        )
    except RiskManagementError as e:
        print(e)
        print(f"\nRequires immediate stop: {e.requires_immediate_stop}")

    # Test 4: FailFastValidator
    print("\n4. FailFastValidator (non-immediate mode):")
    print("-" * 50)
    validator = FailFastValidator(raise_immediately=False)
    validator.require_positive('gamma', -0.5, config_section='ppo')
    validator.require_range('clip_range', 1.5, 0.1, 0.5, config_section='ppo')

    try:
        validator.validate()
    except TradingSystemError as e:
        print(e)

    print("\n" + "=" * 70)
    print("Exception hierarchy ready for use")
    print("=" * 70)
