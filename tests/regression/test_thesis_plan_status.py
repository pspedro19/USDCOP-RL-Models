import json

from scripts.diagnostics.thesis_plan_status import build_status


def test_status_is_fail_closed_when_prerequisites_are_missing(tmp_path):
    result = build_status(tmp_path)
    assert result["ready_for_v2_rebuild"] is False
    assert result["ready_for_confirmatory_llm"] is False
    assert result["ready_for_forward"] is False
    assert result["llm"]["secrets_read"] is False
    assert result["llm"]["network_called"] is False
    assert result["forward_stream"]["activation_authorized"] is False
    assert result["forward_stream"]["runner_present"] is False
    assert result["data_contract"]["artifact_present"] is False
    assert result["source_lineage"]["artifact_present"] is False
    assert result["source_lineage"]["confirmatory_ready"] is False


def test_status_reads_passed_sanity_artifact(tmp_path):
    artifact = tmp_path / "outputs" / "thesis-repair"
    artifact.mkdir(parents=True)
    (artifact / "sanity_protocol_v2.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    result = build_status(tmp_path)
    assert result["ppo_sanity"]["protocol_pass"] is True


def test_status_reports_data_contract_without_treating_grid_gaps_as_clean(tmp_path):
    artifact = tmp_path / "outputs" / "thesis-repair"
    artifact.mkdir(parents=True)
    (artifact / "research_data_contract_v2.json").write_text(
        json.dumps({
            "structural_m5_clean": True,
            "market_numeric_clean": True,
            "macro_columns_complete": True,
            "macro_numeric_clean": True,
            "macro_availability_declared": True,
            "macro_fresh_enough_for_forward": False,
            "complete_session_grid": False,
        }),
        encoding="utf-8",
    )
    result = build_status(tmp_path)
    assert result["data_contract"]["macro_availability_declared"] is True
    assert result["data_contract"]["macro_fresh_enough_for_forward"] is False
    assert result["data_contract"]["complete_session_grid"] is False


def test_status_reports_stream_components_without_authorizing_them(tmp_path):
    source = tmp_path / "src" / "research" / "llm_forward"
    source.mkdir(parents=True)
    scripts = tmp_path / "scripts" / "analysis"
    scripts.mkdir(parents=True)
    (source / "stream_runner.py").write_text("# fixture\n", encoding="utf-8")
    (source / "settle_thesis.py").write_text("# fixture\n", encoding="utf-8")
    (scripts / "run_ppo_stream_bar.py").write_text("# fixture\n", encoding="utf-8")
    result = build_status(tmp_path)
    assert result["forward_stream"] == {
        "runner_present": True,
        "cli_present": True,
        "aggregate_settlement_present": True,
        "activation_authorized": False,
    }
