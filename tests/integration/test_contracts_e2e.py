"""
End-to-End Contract Tests.

Verifica que todos los contratos funcionan juntos correctamente.
"""
import pytest
import numpy as np
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


class TestTrainingInferenceParity:
    """Tests que verifican paridad entre training e inference."""

    @pytest.fixture
    def realistic_observation(self):
        """Observación realista basada en datos históricos."""
        return np.array([
            0.001,   # log_ret_5m
            0.003,   # log_ret_1h
            0.01,    # log_ret_4h
            -0.5,    # rsi_9
            0.2,     # atr_pct
            0.1,     # adx_14
            -0.3,    # dxy_z
            0.001,   # dxy_change_1d
            0.5,     # vix_z
            0.2,     # embi_z
            0.002,   # brent_change_1d
            0.1,     # rate_spread
            -0.001,  # usdmxn_change_1d
            0.0,     # position
            0.5,     # time_normalized
        ], dtype=np.float32)

    def test_observation_passes_all_contracts(self, realistic_observation):
        """Observación realista pasa todos los contratos."""
        assert validate_feature_vector(realistic_observation)
        MODEL_INPUT_CONTRACT.validate(realistic_observation)
        validator = ObservationValidator(strict_mode=True)
        validated = validator.validate_and_prepare(realistic_observation)
        assert validated is not None

    def test_feature_order_matches_training(self, realistic_observation):
        """Orden de features es consistente con training."""
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
            signal = action.to_signal()
            recovered = Action.from_string(signal)
            assert recovered == action


class TestContractInteraction:
    """Tests de interacción entre contratos."""

    def test_invalid_observation_fails_early(self):
        """Observación inválida falla en feature contract."""
        obs_wrong_shape = np.zeros(14, dtype=np.float32)
        with pytest.raises(Exception):
            validate_feature_vector(obs_wrong_shape)

    def test_validator_catches_nan_before_model(self):
        """ObservationValidator detecta NaN."""
        obs = np.zeros(15, dtype=np.float32)
        obs[5] = np.nan
        validator = ObservationValidator(strict_mode=True)
        with pytest.raises(Exception):
            validator.validate_and_prepare(obs)

    def test_action_contract_validates_model_output(self):
        """Action contract valida output del modelo."""
        with pytest.raises(InvalidActionError):
            validate_model_output(action=5, confidence=0.8)


class TestContractVersions:
    """Tests de versionado de contratos."""

    def test_feature_contract_version(self):
        # Version 2.1.0 added feature contracts registry support
        assert FEATURE_CONTRACT.version == "2.1.0"

    def test_feature_order_hash_exists(self):
        assert len(FEATURE_CONTRACT.feature_order_hash) == 16

    def test_contracts_can_be_serialized(self):
        contract_dict = FEATURE_CONTRACT.to_dict()
        json_str = json.dumps(contract_dict)
        assert len(json_str) > 0
        loaded = json.loads(json_str)
        assert loaded["version"] == "2.1.0"
        assert loaded["observation_dim"] == 15
        assert len(loaded["feature_order"]) == 15


class TestContractConstants:
    """Tests para constantes de contratos."""

    def test_observation_dim_is_15(self):
        assert OBSERVATION_DIM == 15

    def test_feature_order_length(self):
        assert len(FEATURE_ORDER) == 15

    def test_action_mapping_correct(self):
        assert Action.SELL.value == 0
        assert Action.HOLD.value == 1
        assert Action.BUY.value == 2

    def test_action_count(self):
        assert len(Action) == 3
