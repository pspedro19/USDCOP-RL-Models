"""
ML Workflow Tracker - Disciplined ML experiment tracking.
Contrato ID: CTR-008
CLAUDE-T12 | Plan Item: P0-7

Previene:
- Multiples looks al validation set sin documentar
- Contaminacion del test set
- P-hacking por busqueda exhaustiva
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, Dict, List, Any


@dataclass
class ExperimentLog:
    """Registro de experimento para auditoria."""
    experiment_id: str
    timestamp: datetime
    phase: str  # 'exploration', 'validation', 'test'
    hyperparameters: dict
    validation_looked: bool
    metrics: Optional[dict] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a dict JSON-compatible."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class MLWorkflowTracker:
    """
    Tracker para workflow ML disciplinado.
    Contrato: CTR-008

    Previene:
    - Multiples looks al validation set sin documentar
    - Contaminacion del test set
    - P-hacking por busqueda exhaustiva

    Ejemplo:
        tracker = MLWorkflowTracker()
        tracker.log_validation_look("Probando learning_rate=1e-4")
        summary = tracker.get_exploration_summary()
    """

    # Umbrales de warning
    MAX_VALIDATION_LOOKS_WARNING = 20
    MAX_VALIDATION_LOOKS_ERROR = 50

    def __init__(self, config_path: Optional[Path] = None):
        """
        Inicializa el tracker.

        Args:
            config_path: Ruta al archivo de configuracion.
                        Default: config/hyperparameter_decisions.json
        """
        if config_path is None:
            # Buscar relativo a la raiz del proyecto
            possible_paths = [
                Path("config/hyperparameter_decisions.json"),
                Path(__file__).parent.parent.parent / "config" / "hyperparameter_decisions.json"
            ]
            for p in possible_paths:
                if p.exists():
                    config_path = p
                    break
            if config_path is None:
                config_path = possible_paths[0]

        self.config_path = Path(config_path)
        self.config = self._load_or_create_config()

    def _load_or_create_config(self) -> dict:
        """Carga configuracion existente o crea una nueva."""
        if self.config_path.exists():
            with open(self.config_path, encoding='utf-8') as f:
                return json.load(f)

        # Config por defecto
        return {
            "model_version": "",
            "created_at": datetime.now().isoformat()[:10],
            "exploration_phase": {
                "start_date": None,
                "end_date": None,
                "experiments_run": 0,
                "validation_looks": 0,
                "notes": ""
            },
            "final_hyperparameters": {},
            "validation_phase": {
                "single_pass_date": None,
                "validation_metrics": None,
                "passed": None
            },
            "test_phase": {
                "execution_date": None,
                "test_set_touched": False,
                "final_metrics": None,
                "notes": "Test set reservado para evaluacion final"
            },
            "audit_trail": []
        }

    def _save_config(self) -> None:
        """Guarda configuracion a disco."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def log_validation_look(self, action: str, notes: str = "") -> int:
        """
        Registra cada vez que se mira el validation set.

        Args:
            action: Descripcion de la accion realizada
            notes: Notas adicionales

        Returns:
            Numero total de validation looks

        Raises:
            Warning si supera MAX_VALIDATION_LOOKS_WARNING
        """
        self.config["exploration_phase"]["validation_looks"] += 1
        current_looks = self.config["exploration_phase"]["validation_looks"]

        self.config["audit_trail"].append({
            "date": datetime.now().isoformat(),
            "action": action,
            "validation_looks": current_looks,
            "notes": notes
        })
        self._save_config()

        # Warnings
        if current_looks > self.MAX_VALIDATION_LOOKS_ERROR:
            raise ValueError(
                f"ALTO RIESGO: {current_looks} validation looks - "
                f"resultados no son confiables. Considere usar nuevo validation set."
            )
        elif current_looks > self.MAX_VALIDATION_LOOKS_WARNING:
            print(f"WARNING: {current_looks} validation looks - alto riesgo de overfitting")

        return current_looks

    def log_experiment(self, experiment: ExperimentLog) -> None:
        """
        Registra un experimento completo.

        Args:
            experiment: Objeto ExperimentLog con detalles del experimento
        """
        self.config["exploration_phase"]["experiments_run"] += 1

        if experiment.validation_looked:
            self.log_validation_look(
                action=f"Experiment {experiment.experiment_id}",
                notes=experiment.notes
            )

        self._save_config()

    def validate_test_set_untouched(self) -> bool:
        """
        Verifica que test set no ha sido tocado.

        Returns:
            True si test set no ha sido tocado

        Raises:
            ValueError si test set ya fue tocado
        """
        if self.config["test_phase"]["test_set_touched"]:
            raise ValueError(
                "TEST SET YA FUE TOCADO - Resultados no son confiables\n"
                f"Fecha: {self.config['test_phase'].get('execution_date')}\n"
                "Para resultados validos, necesita un nuevo test set."
            )
        return True

    def mark_test_execution(self, metrics: dict) -> None:
        """
        Marca ejecucion unica del test set.

        Args:
            metrics: Metricas finales del test set

        Raises:
            ValueError si test set ya fue ejecutado
        """
        if self.config["test_phase"]["test_set_touched"]:
            raise ValueError(
                "Test set ya fue ejecutado - no se permite re-ejecucion.\n"
                "Para evaluar nuevamente, necesita un test set completamente nuevo."
            )

        self.config["test_phase"]["test_set_touched"] = True
        self.config["test_phase"]["execution_date"] = datetime.now().isoformat()
        self.config["test_phase"]["final_metrics"] = metrics
        self._save_config()

        print("Test set ejecutado - metricas finales registradas")

    def mark_validation_passed(self, metrics: dict) -> None:
        """
        Marca que la fase de validacion fue completada.

        Args:
            metrics: Metricas de validacion (sharpe, max_drawdown, win_rate, etc.)
        """
        self.config["validation_phase"]["single_pass_date"] = datetime.now().isoformat()[:10]
        self.config["validation_phase"]["validation_metrics"] = metrics
        self.config["validation_phase"]["passed"] = True
        self._save_config()

    def set_final_hyperparameters(self, hyperparameters: dict, selection_criteria: str) -> None:
        """
        Registra los hiperparametros finales seleccionados.

        Args:
            hyperparameters: Dict con hiperparametros (learning_rate, n_steps, etc.)
            selection_criteria: Descripcion de como se seleccionaron
        """
        self.config["final_hyperparameters"] = hyperparameters
        self.config["final_hyperparameters"]["selection_criteria"] = selection_criteria
        self._save_config()

    def get_exploration_summary(self) -> dict:
        """
        Resumen de fase de exploracion.

        Returns:
            Dict con total_validation_looks, audit_entries, test_set_touched
        """
        return {
            "total_validation_looks": self.config["exploration_phase"]["validation_looks"],
            "experiments_run": self.config["exploration_phase"].get("experiments_run", 0),
            "audit_entries": len(self.config["audit_trail"]),
            "test_set_touched": self.config["test_phase"]["test_set_touched"],
            "validation_passed": self.config["validation_phase"].get("passed", False),
            "risk_level": self._calculate_risk_level()
        }

    def _calculate_risk_level(self) -> str:
        """Calcula nivel de riesgo de overfitting."""
        looks = self.config["exploration_phase"]["validation_looks"]

        if looks <= 10:
            return "LOW"
        elif looks <= self.MAX_VALIDATION_LOOKS_WARNING:
            return "MEDIUM"
        elif looks <= self.MAX_VALIDATION_LOOKS_ERROR:
            return "HIGH"
        else:
            return "CRITICAL"

    def get_audit_trail(self) -> List[dict]:
        """
        Retorna el audit trail completo.

        Returns:
            Lista de entries del audit trail
        """
        return self.config["audit_trail"]

    def can_proceed_to_test(self) -> bool:
        """
        Verifica si se puede proceder a la fase de test.

        Returns:
            True si validation paso y test set no ha sido tocado
        """
        validation_passed = self.config["validation_phase"].get("passed") is True
        test_untouched = not self.config["test_phase"]["test_set_touched"]
        return validation_passed and test_untouched
