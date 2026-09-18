from pathlib import Path

import yaml

from scripts.diagnostics.audit_research_source_lineage import audit

ROOT = Path(__file__).resolve().parents[2]


def test_source_lineage_is_fail_closed_without_network_or_secrets():
    report = audit()
    assert report["network_called"] is False
    assert report["secrets_read"] is False
    assert report["diagnostic_allowed"] is True
    assert report["confirmatory_ready"] is True


def test_source_lineage_policy_forbids_silent_fallback():
    config = yaml.safe_load((ROOT / "config/research/source_lineage.yaml").read_text(encoding="utf-8"))
    assert config["policy"]["silent_fallback"] == "forbidden"
    assert config["policy"]["secondary_is_validation_only"] is True
    assert config["series"]["usdcop_m5"]["frequency"] == "5min"
    assert config["series"]["dxy"]["primary_source"] == "investing"
    assert config["series"]["dxy"]["primary_series_id"] == 942611


def test_source_lineage_requires_primary_and_validators(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "contract: x\npolicy: {}\nsources: {}\nseries:\n  x:\n    frequency: daily\n",
        encoding="utf-8",
    )
    report = audit(bad, root=tmp_path)
    assert report["confirmatory_ready"] is False
    assert any("missing_instrument" in error for error in report["errors"])
    assert any("unknown_primary_source" in error for error in report["errors"])
