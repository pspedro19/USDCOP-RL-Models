"""Execute DAG task functions without Airflow; NOT an importability certificate."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def task_namespace():
    path = ROOT / "airflow/dags/research_forward_arms.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {
        "_prepare_env": lambda: None,
        "fail_if_upstream_failed": lambda *a, **kw: None,
        "ARMS_LLM": ("llm_direct_fwd_v1",), "ARMS_RL": ("ppo_regime_fwd_k59", "ppo_regime_fwd_k1"),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    namespace["_prepare_env"] = lambda: None
    return namespace


@pytest.mark.parametrize("rc", [1, None, True, False, 0.0])
def test_dag_propagates_invalid_ledger_verdict(monkeypatch, rc):
    from src.research.llm_forward import verify
    monkeypatch.setattr(verify, "main", lambda: rc)
    with pytest.raises(RuntimeError, match="ledger"):
        task_namespace()["audit"]()


def test_unsupported_single_call_dispatch_blocks_before_paid_llm(monkeypatch):
    from src.research.llm_forward import decide
    called = []
    monkeypatch.setattr(decide, "run", lambda *a, **kw: called.append(kw) or 0)
    monkeypatch.setattr(decide, "load_preregistration", lambda *a: ({"arms": []}, "a" * 64))
    monkeypatch.setattr(decide, "arm_spec", lambda spec, arm: {
        "kind": "rl_frozen", "decisions_per_session": 59 if arm.endswith("k1") else 1,
    })
    ns = task_namespace()
    ns["_session_date"] = lambda context: "2023-06-01"
    with pytest.raises(RuntimeError, match="59|per.bar|por.barra"):
        ns["seal_llm"]()
    assert called == []


def test_valid_ledger_status_still_checks_upstream(monkeypatch):
    from src.research.llm_forward import verify
    monkeypatch.setattr(verify, "main", lambda: 0)
    ns = task_namespace()
    called = []
    ns["fail_if_upstream_failed"] = lambda *a, **kw: called.append(kw["task_name"])
    ns["audit"]()
    assert called == ["audit_forward_ledger"]
