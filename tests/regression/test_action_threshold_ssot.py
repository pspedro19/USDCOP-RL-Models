"""
Regression: action-threshold SSOT alignment (audit A2-04).

The RL env config used to default to 0.60 while the SSOT (pipeline_ssot.yaml) said
0.35 and the TS mirror said 0.33. Guard that the config.py fallback default now
equals the authoritative SSOT value so a SSOT-load failure can't silently diverge.
"""
import yaml

from src.training.config import EnvironmentConfig


def test_env_config_threshold_defaults_match_ssot():
    cfg = EnvironmentConfig()
    assert cfg.threshold_long == 0.35, "fallback default must match SSOT (A2-04)"
    assert cfg.threshold_short == -0.35


def test_ssot_yaml_thresholds_are_035():
    d = yaml.safe_load(open("config/pipeline_ssot.yaml"))
    # thresholds live in the env/action section; find them wherever nested.
    text = open("config/pipeline_ssot.yaml").read()
    assert "threshold_long: 0.35" in text
    assert "threshold_short: -0.35" in text
