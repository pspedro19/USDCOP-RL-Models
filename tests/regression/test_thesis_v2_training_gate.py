from pathlib import Path


def test_v2_training_requires_both_evidence_reports():
    source = Path("scripts/analysis/thesis_train_ppo.py").read_text(encoding="utf-8")
    assert 'choices=("v1", "v2")' in source
    assert 'args.dataset_version == "v2"' in source
    assert "dataset v2 exige informes de sanidad e identidad macro existentes" in source
