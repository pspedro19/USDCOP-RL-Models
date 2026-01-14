"""
Tests para ML Workflow Disciplinado (P0-7).
CLAUDE-T12 | Contrato: CTR-008

Valida:
- Tracking de validation looks
- Proteccion del test set
- Audit trail
"""

import pytest
import json
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Importar directamente sin pasar por src.__init__.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from ml_workflow.experiment_tracker import MLWorkflowTracker, ExperimentLog


class TestMLWorkflowDiscipline:
    """Tests para workflow ML disciplinado."""

    @pytest.fixture
    def temp_config_path(self):
        """Crea archivo temporal para tests."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump({
                "model_version": "test",
                "exploration_phase": {"validation_looks": 0, "experiments_run": 0},
                "validation_phase": {"single_pass_date": None, "passed": None},
                "test_phase": {"test_set_touched": False},
                "audit_trail": []
            }, f)
            return Path(f.name)

    @pytest.fixture
    def tracker(self, temp_config_path):
        """MLWorkflowTracker con config temporal."""
        return MLWorkflowTracker(temp_config_path)

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Validation looks son trackeados
    # ══════════════════════════════════════════════════════════════
    def test_validation_looks_are_tracked(self, tracker):
        """Cada look al validation set DEBE ser registrado."""
        assert tracker.config["exploration_phase"]["validation_looks"] == 0

        tracker.log_validation_look("Test look 1")
        tracker.log_validation_look("Test look 2")

        assert tracker.config["exploration_phase"]["validation_looks"] == 2
        assert len(tracker.config["audit_trail"]) == 2

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Test set no puede tocarse dos veces
    # ══════════════════════════════════════════════════════════════
    def test_test_set_protection(self, tracker):
        """Test set NO debe poder tocarse dos veces."""
        # Primera verificacion OK
        assert tracker.validate_test_set_untouched() is True

        # Marcar ejecucion
        tracker.mark_test_execution({"sharpe": 1.5})

        # Segunda ejecucion debe fallar
        with pytest.raises(ValueError, match="ya fue ejecutado"):
            tracker.mark_test_execution({"sharpe": 1.6})

    # ══════════════════════════════════════════════════════════════
    # TEST 3: Validate test set ya tocado lanza error
    # ══════════════════════════════════════════════════════════════
    def test_validate_fails_after_test_touched(self, tracker):
        """validate_test_set_untouched DEBE fallar despues de tocar test set."""
        tracker.mark_test_execution({"sharpe": 1.5})

        with pytest.raises(ValueError, match="YA FUE TOCADO"):
            tracker.validate_test_set_untouched()

    # ══════════════════════════════════════════════════════════════
    # TEST 4: Audit trail registra fechas
    # ══════════════════════════════════════════════════════════════
    def test_audit_trail_has_timestamps(self, tracker):
        """Audit trail DEBE incluir timestamps ISO."""
        tracker.log_validation_look("Test action", notes="Test notes")

        trail = tracker.get_audit_trail()
        assert len(trail) == 1
        assert "date" in trail[0]
        # Verificar formato ISO
        datetime.fromisoformat(trail[0]["date"])  # No debe lanzar excepcion

    # ══════════════════════════════════════════════════════════════
    # TEST 5: Exploration summary retorna datos correctos
    # ══════════════════════════════════════════════════════════════
    def test_exploration_summary(self, tracker):
        """get_exploration_summary DEBE retornar datos correctos."""
        tracker.log_validation_look("Look 1")
        tracker.log_validation_look("Look 2")
        tracker.log_validation_look("Look 3")

        summary = tracker.get_exploration_summary()

        assert summary["total_validation_looks"] == 3
        assert summary["audit_entries"] == 3
        assert summary["test_set_touched"] is False
        assert summary["risk_level"] == "LOW"

    # ══════════════════════════════════════════════════════════════
    # TEST 6: Risk level aumenta con mas looks
    # ══════════════════════════════════════════════════════════════
    def test_risk_level_increases(self, tracker):
        """Risk level DEBE aumentar con mas validation looks."""
        # LOW: <= 10
        for i in range(10):
            tracker.log_validation_look(f"Look {i+1}")
        assert tracker._calculate_risk_level() == "LOW"

        # MEDIUM: 11-20
        for i in range(10):
            tracker.log_validation_look(f"Look {i+11}")
        assert tracker._calculate_risk_level() == "MEDIUM"

    # ══════════════════════════════════════════════════════════════
    # TEST 7: Config se persiste a disco
    # ══════════════════════════════════════════════════════════════
    def test_config_persists_to_disk(self, temp_config_path):
        """Config DEBE persistirse a disco."""
        tracker1 = MLWorkflowTracker(temp_config_path)
        tracker1.log_validation_look("Persistent look")

        # Crear nuevo tracker con mismo path
        tracker2 = MLWorkflowTracker(temp_config_path)

        assert tracker2.config["exploration_phase"]["validation_looks"] == 1

    # ══════════════════════════════════════════════════════════════
    # TEST 8: Validation passed se registra
    # ══════════════════════════════════════════════════════════════
    def test_mark_validation_passed(self, tracker):
        """mark_validation_passed DEBE registrar metricas."""
        metrics = {"sharpe": 1.42, "max_drawdown": -0.12}
        tracker.mark_validation_passed(metrics)

        assert tracker.config["validation_phase"]["passed"] is True
        assert tracker.config["validation_phase"]["validation_metrics"] == metrics
        assert tracker.config["validation_phase"]["single_pass_date"] is not None

    # ══════════════════════════════════════════════════════════════
    # TEST 9: can_proceed_to_test funciona
    # ══════════════════════════════════════════════════════════════
    def test_can_proceed_to_test(self, tracker):
        """can_proceed_to_test DEBE retornar True solo si validation paso."""
        # Inicialmente no puede proceder
        assert tracker.can_proceed_to_test() is False

        # Despues de validation
        tracker.mark_validation_passed({"sharpe": 1.5})
        assert tracker.can_proceed_to_test() is True

        # Despues de test execution
        tracker.mark_test_execution({"sharpe": 1.5})
        assert tracker.can_proceed_to_test() is False

    # ══════════════════════════════════════════════════════════════
    # TEST 10: Final hyperparameters se registran
    # ══════════════════════════════════════════════════════════════
    def test_set_final_hyperparameters(self, tracker):
        """set_final_hyperparameters DEBE registrar hiperparametros."""
        hp = {"learning_rate": 3e-4, "n_steps": 2048}
        tracker.set_final_hyperparameters(hp, "Best Sharpe in validation")

        assert tracker.config["final_hyperparameters"]["learning_rate"] == 3e-4
        assert "selection_criteria" in tracker.config["final_hyperparameters"]


class TestMLWorkflowConfigFile:
    """Tests que verifican existencia de archivo de configuracion."""

    def test_hyperparameter_decisions_file_exists(self):
        """Archivo de decisiones de hiperparametros DEBE existir."""
        config_path = Path("config/hyperparameter_decisions.json")
        assert config_path.exists(), \
            "config/hyperparameter_decisions.json no existe - crear con historial"

    def test_hyperparameter_decisions_has_required_fields(self):
        """Archivo DEBE tener campos requeridos."""
        config_path = Path("config/hyperparameter_decisions.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            required_fields = [
                "model_version",
                "exploration_phase",
                "validation_phase",
                "test_phase",
                "audit_trail"
            ]
            for field in required_fields:
                assert field in config, f"Campo requerido faltante: {field}"

    def test_audit_trail_has_entries(self):
        """Audit trail DEBE tener registros de exploracion."""
        config_path = Path("config/hyperparameter_decisions.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            assert len(config.get("audit_trail", [])) > 0, \
                "No hay registros de exploracion - documentar proceso"


class TestExperimentLog:
    """Tests para ExperimentLog dataclass."""

    def test_experiment_log_to_dict(self):
        """ExperimentLog.to_dict DEBE ser JSON serializable."""
        log = ExperimentLog(
            experiment_id="exp-001",
            timestamp=datetime.now(),
            phase="exploration",
            hyperparameters={"lr": 1e-4},
            validation_looked=True,
            metrics={"sharpe": 1.5},
            notes="Test experiment"
        )

        d = log.to_dict()

        # Debe ser serializable
        json_str = json.dumps(d)
        assert "exp-001" in json_str
        assert "exploration" in json_str
