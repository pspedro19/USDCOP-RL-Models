import numpy as np
import pandas as pd
import pytest

from src.research.publication_join_v2 import join_observed_releases


def release(period, publication, seen, value, series="dxy"):
    return {
        "series": series,
        "period_end": period,
        "publication_at": publication,
        "first_seen_at": seen,
        "value": value,
        "unit": "index_points",
        "source_sha256": "a" * 64,
    }


def join(rows, cutoff="2026-09-11T13:00:00Z", minimum=1, max_age=5):
    return join_observed_releases(
        pd.DataFrame([{"decision_id": "d", "cutoff_utc": cutoff}]),
        pd.DataFrame(rows),
        {
            "dxy": {
                "unit": "index_points",
                "minimum_observations": minimum,
                "max_period_age_calendar_days": max_age,
            }
        },
    ).iloc[0]


def test_publication_date_alone_does_not_make_late_ingest_available():
    row = release("2026-09-10", "2026-09-10T20:00Z", "2026-09-11T14:00Z", 100)
    assert join([row])["status"] == "MISSING_BEFORE_CUTOFF"
    assert join([row], cutoff="2026-09-11T14:00Z")["status"] == "MISSING_BEFORE_CUTOFF"
    assert join([row], cutoff="2026-09-11T14:01Z")["level"] == 100


def test_future_revision_does_not_rewrite_previous_decision():
    rows = [
        release("2026-09-09", "2026-09-09T20:00Z", "2026-09-09T20:01Z", 100),
        release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 101),
    ]
    original = join(rows, minimum=2)
    revised = join(
        [*rows, release("2026-09-10", "2026-09-11T15:00Z", "2026-09-11T15:01Z", 900)], minimum=2
    )
    assert revised["level"] == original["level"] == 101
    assert revised["log_return_previous"] == pytest.approx(np.log(101 / 100))


def test_old_period_revision_does_not_replace_latest_observation():
    rows = [
        release("2026-09-09", "2026-09-09T20:00Z", "2026-09-09T20:01Z", 100),
        release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 101),
        release("2026-09-09", "2026-09-11T12:00Z", "2026-09-11T12:01Z", 99),
    ]
    result = join(rows, minimum=2)
    assert result["level"] == 101
    assert result["log_return_previous"] == pytest.approx(np.log(101 / 99))


def test_recent_revision_cannot_refresh_stale_observation_period():
    row = release("2026-06-30", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 100)
    assert join([row])["status"] == "STALE_OBSERVATION_PERIOD"
    # A quarterly variable needs its own declared freshness, not daily 5-day rules.
    assert join([row], max_age=120)["status"] == "AVAILABLE"


def test_naive_publication_and_duplicate_vintage_fail_closed():
    row = release("2026-09-10", "2026-09-10T20:00", "2026-09-10T20:01Z", 100)
    with pytest.raises(ValueError, match="timezone"):
        join([row])
    row["publication_at"] += "Z"
    with pytest.raises(ValueError, match="duplicate"):
        join([row, row])


@pytest.mark.parametrize("minimum", [True, False, 1.0, 2.0, "2"])
def test_minimum_observations_requires_an_integer_not_a_coercible_value(minimum):
    row = release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 100)
    with pytest.raises(ValueError, match="policy"):
        join([row], minimum=minimum)


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_boolean_source_levels_are_not_prices(value):
    row = release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", value)
    with pytest.raises(ValueError, match="level"):
        join([row])


def test_unknown_policy_fields_do_not_silently_change_the_declared_contract():
    d = pd.DataFrame([{"decision_id": "d", "cutoff_utc": "2026-09-11T13:00Z"}])
    rows = pd.DataFrame([release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 100)])
    policy = {
        "unit": "index_points",
        "minimum_observations": 1,
        "max_period_age_calendar_days": 5,
        "maximum_gap_days": 1,
    }
    with pytest.raises(ValueError, match="policy"):
        join_observed_releases(d, rows, {"dxy": policy})


@pytest.mark.parametrize("decision_id", [None, "", "   ", True, 123])
def test_decision_identity_is_required(decision_id):
    d = pd.DataFrame([{"decision_id": decision_id, "cutoff_utc": "2026-09-11T13:00Z"}])
    rows = pd.DataFrame([release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 100)])
    policy = {"unit": "index_points", "minimum_observations": 1, "max_period_age_calendar_days": 5}
    with pytest.raises(ValueError, match="decision"):
        join_observed_releases(d, rows, {"dxy": policy})


@pytest.mark.parametrize("period", ["09/10/2026", "2026-09-10T00:00:00", 1788998400000000000])
def test_period_labels_do_not_guess_format_or_numeric_epoch(period):
    row = release(period, "2026-09-10T20:00Z", "2026-09-10T20:01Z", 100)
    with pytest.raises(ValueError, match="period_end"):
        join([row])


def test_future_period_in_cot_is_not_made_available_by_utc_midnight():
    row = release("2026-09-11", "2026-09-11T00:30Z", "2026-09-11T00:31Z", 100)
    result = join([row], cutoff="2026-09-11T01:00Z")
    assert result["status"] == "FUTURE_OBSERVATION_PERIOD"
    assert pd.isna(result["level"])
    assert result["period_age_calendar_days"] == -1


def test_both_return_operands_must_satisfy_declared_period_freshness():
    rows = [
        release("2020-01-01", "2020-01-01T20:00Z", "2020-01-01T20:01Z", 100),
        release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 101),
    ]
    result = join(rows, minimum=2)
    assert result["status"] == "STALE_PREVIOUS_OBSERVATION_PERIOD"
    assert pd.isna(result["level"]) and pd.isna(result["log_return_previous"])
    assert result["previous_period_age_calendar_days"] > 2000
    assert result["previous_source_sha256"] == "a" * 64


@pytest.mark.parametrize("first,last", [(1e-300, 1e300), (1e300, 1e-300)])
def test_finite_positive_levels_produce_finite_log_returns_without_ratio_overflow(first, last):
    rows = [
        release("2026-09-09", "2026-09-09T20:00Z", "2026-09-09T20:01Z", first),
        release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", last),
    ]
    with np.errstate(all="raise"):
        result = join(rows, minimum=2)
    assert result["status"] == "AVAILABLE"
    assert result["log_return_previous"] == pytest.approx(np.log(last) - np.log(first))
    assert np.isfinite(result["log_return_previous"])


def test_output_columns_are_stable_when_previous_context_is_missing_or_no_decisions():
    rows = [
        release("2026-09-09", "2026-09-09T20:00Z", "2026-09-09T20:01Z", 100),
        release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 101),
    ]
    complete = join(rows, minimum=2)
    missing = join(rows[-1:], minimum=2)
    assert set(missing.index) == set(complete.index)
    assert {
        "previous_period_end",
        "previous_available_at",
        "previous_period_age_calendar_days",
    } <= set(missing.index)
    empty = join_observed_releases(
        pd.DataFrame(columns=["decision_id", "cutoff_utc"]),
        pd.DataFrame(rows),
        {
            "dxy": {
                "unit": "index_points",
                "minimum_observations": 2,
                "max_period_age_calendar_days": 5,
            }
        },
    )
    assert empty.empty and set(empty.columns) == set(complete.index)


def test_mixed_frequencies_keep_own_units_periods_and_declared_freshness():
    rows = [
        release("2026-09-09", "2026-09-09T20:00Z", "2026-09-09T20:01Z", 100),
        release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 101),
        release("2026-09-04", "2026-09-08T12:00Z", "2026-09-08T12:01Z", 500, "weekly"),
        release("2026-08-31", "2026-09-10T12:00Z", "2026-09-10T12:01Z", 3.2, "monthly"),
        release("2026-06-30", "2026-08-01T12:00Z", "2026-08-01T12:01Z", -0.4, "quarterly"),
    ]
    policies = {
        "dxy": {
            "unit": "index_points",
            "minimum_observations": 2,
            "max_period_age_calendar_days": 5,
        }
    }
    for series, unit, age in [
        ("weekly", "count", 14),
        ("monthly", "percent", 45),
        ("quarterly", "percent", 120),
    ]:
        policies[series] = {
            "unit": unit,
            "minimum_observations": 1,
            "max_period_age_calendar_days": age,
        }
        for row in rows:
            if row["series"] == series:
                row["unit"] = unit
    decisions = pd.DataFrame([{"decision_id": "d", "cutoff_utc": "2026-09-11T13:00Z"}])
    result = join_observed_releases(decisions, pd.DataFrame(rows), policies).set_index("series")
    assert result["status"].eq("AVAILABLE").all()
    assert result["level"].to_dict() == {
        "dxy": 101,
        "weekly": 500,
        "monthly": 3.2,
        "quarterly": -0.4,
    }
    assert result["period_age_calendar_days"].to_dict() == {
        "dxy": 1,
        "weekly": 7,
        "monthly": 11,
        "quarterly": 73,
    }
    assert result.loc["dxy", "log_return_previous"] == pytest.approx(np.log(101 / 100))
    assert result.loc[["weekly", "monthly", "quarterly"], "log_return_previous"].isna().all()


@pytest.mark.parametrize("shuffle_seed", range(10))
def test_batch_equals_each_available_prefix_even_with_out_of_order_revisions(shuffle_seed):
    rows = [
        release("2026-09-08", "2026-09-08T20:00Z", "2026-09-08T20:01Z", 99),
        release("2026-09-09", "2026-09-09T20:00Z", "2026-09-09T20:01Z", 100),
        release("2026-09-10", "2026-09-10T20:00Z", "2026-09-11T14:00Z", 101),
        release("2026-09-09", "2026-09-11T15:00Z", "2026-09-11T15:01Z", 102),
        # Earlier vintage received last must not replace a newer published revision.
        release("2026-09-09", "2026-09-10T20:00Z", "2026-09-11T15:03Z", 98),
    ]
    source = pd.DataFrame(rows).sample(frac=1, random_state=shuffle_seed)
    cutoffs = [
        "2026-09-11T13:00Z",
        "2026-09-11T14:00Z",
        "2026-09-11T14:01Z",
        "2026-09-11T15:01Z",
        "2026-09-11T15:02Z",
        "2026-09-11T15:04Z",
    ]
    decisions = pd.DataFrame(
        [{"decision_id": str(i), "cutoff_utc": cutoff} for i, cutoff in enumerate(cutoffs)]
    )
    policy = {
        "dxy": {
            "unit": "index_points",
            "minimum_observations": 2,
            "max_period_age_calendar_days": 5,
        }
    }
    batch = join_observed_releases(decisions, source, policy).set_index("decision_id")
    known_at = pd.concat(
        [pd.to_datetime(source.publication_at), pd.to_datetime(source.first_seen_at)], axis=1
    ).max(axis=1)
    for i, cutoff in enumerate(cutoffs):
        prefix = source.loc[known_at < pd.Timestamp(cutoff)]
        streamed = join_observed_releases(decisions.iloc[[i]], prefix, policy).set_index(
            "decision_id"
        )
        pd.testing.assert_series_equal(batch.loc[str(i)], streamed.loc[str(i)])
    assert batch.loc["0", "level"] == batch.loc["1", "level"] == 100
    assert batch.loc["2", "level"] == 101
    assert batch.loc["4", "log_return_previous"] == pytest.approx(np.log(101 / 102))
    assert batch.loc["5", "log_return_previous"] == batch.loc["4", "log_return_previous"]


def test_staleness_boundary_includes_both_operands_and_does_not_fill_a_missing_series():
    rows = [
        release("2026-09-06", "2026-09-06T20:00Z", "2026-09-06T20:01Z", 100),
        release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 101),
    ]
    assert join(rows, minimum=2, max_age=5)["status"] == "AVAILABLE"
    assert join(rows, minimum=2, max_age=4)["status"] == "STALE_PREVIOUS_OBSERVATION_PERIOD"
    result = join_observed_releases(
        pd.DataFrame([{"decision_id": "d", "cutoff_utc": "2026-09-11T13:00Z"}]),
        pd.DataFrame(columns=rows[0].keys()),
        {
            "dxy": {
                "unit": "index_points",
                "minimum_observations": 2,
                "max_period_age_calendar_days": 5,
            }
        },
    ).iloc[0]
    assert result["status"] == "MISSING_BEFORE_CUTOFF"
    assert pd.isna(result["level"]) and pd.isna(result["log_return_previous"])


def test_invalid_future_release_rejects_ledger_instead_of_silent_quarantine():
    rows = [
        release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 100),
        release("2026-09-11", "2026-09-11T20:00Z", "2026-09-11T20:01Z", np.inf),
    ]
    with pytest.raises(ValueError, match="level"):
        join(rows)


@pytest.mark.parametrize("dtype", ["string", object])
def test_nullable_unit_is_rejected_and_not_replaced_by_policy_unit(dtype):
    d = pd.DataFrame([{"decision_id": "d", "cutoff_utc": "2026-09-11T13:00Z"}])
    rows = pd.DataFrame([release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 100)])
    rows["unit"] = pd.Series([pd.NA], dtype=dtype)
    policy = {
        "dxy": {
            "unit": "index_points",
            "minimum_observations": 1,
            "max_period_age_calendar_days": 5,
        }
    }
    with pytest.raises(ValueError, match="unit"):
        join_observed_releases(d, rows, policy)


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
@pytest.mark.parametrize("future", [False, True])
def test_nullable_source_levels_are_rejected_even_when_after_cutoff(dtype, future):
    d = pd.DataFrame([{"decision_id": "d", "cutoff_utc": "2026-09-11T13:00Z"}])
    rows = pd.DataFrame(
        [
            release("2026-09-09", "2026-09-09T20:00Z", "2026-09-09T20:01Z", 100),
            release(
                "2026-09-10",
                "2026-09-11T20:00Z" if future else "2026-09-10T20:00Z",
                "2026-09-11T20:01Z" if future else "2026-09-10T20:01Z",
                pd.NA,
            ),
        ]
    )
    rows["value"] = pd.Series([100, pd.NA], dtype=dtype)
    policy = {
        "dxy": {
            "unit": "index_points",
            "minimum_observations": 1,
            "max_period_age_calendar_days": 5,
        }
    }
    with pytest.raises(ValueError, match="level"):
        join_observed_releases(d, rows, policy)


@pytest.mark.parametrize(
    "value", [pd.Timedelta(days=1), np.timedelta64(1, "D"), np.datetime64("2026-09-10")]
)
def test_temporal_values_are_not_coerced_to_numeric_source_levels(value):
    row = release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", value)
    with pytest.raises(ValueError, match="level"):
        join([row])


def test_source_hash_requires_string_and_never_coerces_a_numeric_identifier():
    row = release("2026-09-10", "2026-09-10T20:00Z", "2026-09-10T20:01Z", 100)
    row["source_sha256"] = int("1" * 64)
    with pytest.raises(ValueError, match="hash"):
        join([row])
