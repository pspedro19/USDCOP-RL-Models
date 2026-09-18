import pytest


def test_stale_figure_replay_requires_separate_output_dir(monkeypatch):
    from scripts.presentation import generar_resultados_y_figuras as module

    monkeypatch.setattr("sys.argv", [
        "generar_resultados_y_figuras.py",
        "--block", "holdout",
        "--allow-stale-artifact",
    ])
    with pytest.raises(SystemExit, match="output-dir"):
        module.main()
