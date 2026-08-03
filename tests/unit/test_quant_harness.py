import numpy as np
from src.validation.quant_harness import evaluate_asset, run_harness

def _frame(n=40):
    return {"timestamp": list(range(n)), "available_at": list(range(n)), "close": list(np.arange(n)+100)}

def test_asset_harness_detects_missing_dsr_evidence():
    ev = evaluate_asset("spx500", _frame(), prediction=np.arange(40), actual=np.arange(40),
                        baseline=np.arange(40)+1, returns=np.full(40, .001))
    assert not ev.passed
    assert any(c.name == "statistics.dsr_pbo_evidence" and not c.passed for c in ev.checks)

def test_run_harness_emits_all_contract_fields():
    out = run_harness({a: _frame() for a in ("usdcop", "xauusd", "btcusdt", "spx500")})
    assert out["schema_version"] == 1
    assert {x["asset"] for x in out["assets"]} == {"usdcop", "xauusd", "btcusdt", "spx500"}
