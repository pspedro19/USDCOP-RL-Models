from src.validation.production_harness import evaluate

def candidate(**m):
    return {"model_version":"v1","dataset_hash":"abc","metrics":m}

def test_passes_all_gates():
    r=evaluate(candidate(psi=.05,latency_p95_ms=100,error_rate=0,uptime=1,sharpe=.5,max_drawdown=.1)); assert r.go

def test_drift_blocks():
    r=evaluate(candidate(psi=.5,latency_p95_ms=100,error_rate=0,uptime=1,sharpe=.5,max_drawdown=.1)); assert not r.go and not r.gates['drift']

def test_challenger_regression_blocks():
    c=candidate(psi=.01,latency_p95_ms=1,error_rate=0,uptime=1,sharpe=.1,max_drawdown=.1)
    r=evaluate(c, candidate(psi=.01,latency_p95_ms=1,error_rate=0,uptime=1,sharpe=.5,max_drawdown=.1)); assert not r.go

def test_kill_switch_blocks():
    r=evaluate(candidate(), kill_switch=True); assert not r.go
