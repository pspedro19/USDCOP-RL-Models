"""Adversarial tests of date-resolution evidence, not invented publication times."""

import copy
from datetime import UTC, datetime

import pandas as pd
import pytest

from scripts.diagnostics.audit_research_macro_vintages import ROOT, allowed
from src.research.alfred_vintages import (
    capture_snapshot,
    compare_snapshot,
    parse_snapshot,
    replay_snapshot,
    snapshot_url,
    validate_daily_labels,
)


def raw(series="DGS2", vintage="20230104", values="2023-01-02,4.30\n2023-01-03,4.40\n"):
    return f"observation_date,{series}_{vintage}\n{values}".encode()


def test_exact_vintage_header_and_missing_values():
    result = parse_snapshot(raw(values="2023-01-02,4.30\n2023-01-03,\n"), "DGS2", "2023-01-04")
    assert result.loc["2023-01-02"] == 4.3
    assert pd.isna(result.loc["2023-01-03"])


@pytest.mark.parametrize("payload", [
    raw(vintage="20260104"),
    raw(series="DGS10"),
    b"observation_date,DGS2\n2023-01-02,4.3\n",
    raw(values="2023-01-02,NaN\n"),
    raw(values="2023-01-02,inf\n"),
    raw(values="2023-01-02,4,30\n"),
    raw(values="2023-01-02,4.3\n2023-01-02,4.3\n"),
    raw(values="2023-01-05,4.3\n"),
    raw(values="2023-01-02T00:00:00,4.3\n"),
    raw(values="2023-01-03,4.3\n2023-01-02,4.2\n"),
    b"<html>not a dataset</html>",
])
def test_reject_wrong_vintage_identity_and_malformed_payload(payload):
    with pytest.raises(ValueError):
        parse_snapshot(payload, "DGS2", "2023-01-04")


@pytest.mark.parametrize("bad", ["2023-01-02T00:00Z", "2023-1-2", "NaT", "2037-01-01"])
def test_no_timestamp_or_future_date_in_daily_labels(bad):
    with pytest.raises(ValueError):
        validate_daily_labels([bad])


def test_url_is_fixed_public_origin_without_credentials():
    assert snapshot_url("DGS2", "2023-01-04", "2022-12-01").startswith(
        "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=DGS2&"
    )
    with pytest.raises(ValueError):
        snapshot_url("DGS2&api_key=secret", "2023-01-04", "2022-12-01")


def test_absence_is_not_proof_of_unavailability_at_open():
    local = pd.Series([4.3, 4.4], index=pd.to_datetime(["2023-01-02", "2023-01-03"]))
    vintage = parse_snapshot(raw(vintage="20230103", values="2023-01-02,4.30\n"), "DGS2", "2023-01-03")
    result = compare_snapshot(local, vintage, series="DGS2", session_date="2023-01-04", vintage_date="2023-01-03")
    assert result["status"] == "NOT_IN_PRIOR_DATE_VINTAGE"
    assert result["opening_availability_verified"] is False
    assert result["snapshot_latest_period"] == "2023-01-02"
    assert result["t1_selected_periods"] == ["2023-01-03"]


def test_same_day_vintage_cannot_certify_preopen():
    local = pd.Series([4.3], index=pd.to_datetime(["2023-01-02"]))
    with pytest.raises(ValueError):
        compare_snapshot(local, local, series="DGS2", session_date="2023-01-03", vintage_date="2023-01-03")


def test_value_match_is_date_resolution_only():
    local = pd.Series([4.3, 4.4], index=pd.to_datetime(["2023-01-02", "2023-01-03"]))
    result = compare_snapshot(local, local, series="DGS2", session_date="2023-01-04", vintage_date="2023-01-03")
    assert result["status"] == "VALUES_MATCH_PRIOR_DATE_VINTAGE"
    assert result["opening_availability_verified"] is False
    assert result["period_age_calendar_days"] == 1


def test_revised_level_is_not_conflated_with_publication_delay():
    local = pd.Series([4.3, 4.4], index=pd.to_datetime(["2023-01-02", "2023-01-03"]))
    vintage = local.copy()
    vintage.iloc[-1] = 4.39
    result = compare_snapshot(local, vintage, series="DGS2", session_date="2023-01-04", vintage_date="2023-01-03")
    assert result["status"] == "VALUE_DIFFERS_FROM_PRIOR_DATE_VINTAGE"
    assert result["max_abs_selected_value_difference"] == pytest.approx(0.01)


def test_brent_requires_both_return_endpoints_and_distinct_periods():
    local = pd.Series([80.0, 81.0], index=pd.to_datetime(["2023-01-02", "2023-01-03"]))
    vintage = local.iloc[-1:]
    result = compare_snapshot(local, vintage, series="DCOILBRENTEU", session_date="2023-01-04", vintage_date="2023-01-03")
    assert result["status"] == "NOT_IN_PRIOR_DATE_VINTAGE"
    assert result["n_required_observations"] == 2
    assert result["snapshot_transform"] is None


def test_current_rows_do_not_enter_t1_comparison():
    local = pd.Series([4.3, 4.4, 99.0], index=pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]))
    result = compare_snapshot(local, local.iloc[:2], series="DGS2", session_date="2023-01-04", vintage_date="2023-01-03")
    assert result["t1_transform"] == 4.4


def test_stale_local_is_not_a_vintage_discrepancy():
    local = pd.Series([4.3], index=pd.to_datetime(["2022-01-02"]))
    result = compare_snapshot(local, local.iloc[:0], series="DGS2", session_date="2023-01-04", vintage_date="2023-01-03")
    assert result["status"] == "LOCAL_STALE_UNDER_T1_CONTRACT"
    assert result["t1_transform"] is None


def test_query_window_censoring_does_not_count_as_absent():
    local = pd.Series([80.0, 81.0], index=pd.to_datetime(["2022-10-01", "2023-01-03"]))
    result = compare_snapshot(local, local.iloc[-1:], series="DCOILBRENTEU", session_date="2023-01-04", vintage_date="2023-01-03", capture_start_date="2022-12-01")
    assert result["status"] == "NOT_AUDITED_OUTSIDE_CAPTURE_WINDOW"


@pytest.fixture
def captured(tmp_path, monkeypatch):
    class Response:
        status = 200

        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self):
            return self.url

        def read(self, count):
            return raw(vintage="20230103")

    class Opener:
        def open(self, request, timeout):
            return Response(request.full_url)

    monkeypatch.setattr("src.research.alfred_vintages.urllib.request.build_opener", lambda *args: Opener())
    record = capture_snapshot(tmp_path, "DGS2", "2023-01-04")
    return tmp_path, record


def test_capture_offline_replay(captured):
    directory, record = captured
    assert replay_snapshot(directory, record).iloc[-1] == 4.4
    assert datetime.fromisoformat(record["retrieved_at_utc"]) <= datetime.now(UTC)


@pytest.mark.parametrize("field,value", [
    ("session_date", "2023-02-04"),
    ("url", "https://example.com/"),
    ("parser_sha256", "0" * 64),
    ("raw_path", "../../.env"),
    ("bytes", 0),
    ("retrieved_at_utc", "2099-01-01T00:00:00+00:00"),
])
def test_offline_capture_mutation_rejected(captured, field, value):
    directory, record = captured
    mutated = copy.deepcopy(record)
    mutated[field] = value
    with pytest.raises(ValueError):
        replay_snapshot(directory, mutated)


@pytest.mark.parametrize("filename", [
    ".env", ".env.example", "secrets/ignored.json", "private.pem", "private.key",
    "credentials-fixture.json", "service-account-fixture.json",
])
def test_all_prohibited_secret_paths_rejected_without_reading(filename):
    with pytest.raises(ValueError):
        allowed(ROOT / "outputs/thesis-repair" / filename)
