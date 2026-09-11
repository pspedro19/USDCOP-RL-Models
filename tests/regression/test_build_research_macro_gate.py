from pathlib import Path


def test_research_macro_builder_has_fail_closed_default():
    source = Path("scripts/data/build_research_macro.py").read_text(encoding="utf-8")
    assert "--allow-unverified" in source
    assert "no se escribe ningún artefacto" in source
    assert "if unverified and not args.allow_unverified" in source
