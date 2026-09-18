"""C046: prefix admission must not depend on the unobserved end of the session.

Deterministic fixtures test software, not profitability. Market-feature and HMM
observation formulas run normally; macro availability and production deployment
are deliberately not certified here. No fitting, API calls or frozen writes.
"""
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.research import evaluation_mask as masks
from src.research import live_spec as live
from src.research import regime_hmm
from src.research.regime_portable import PortableRegimeModel


def bars_for(day):
    times = pd.date_range(f"{day} 08:00", periods=60, freq="5min", tz="America/Bogota")
    i = np.arange(60, dtype=float)
    close = 4000.0 + pd.Timestamp(day).day + 3 * np.sin(i / 3) + i / 5
    return pd.DataFrame({"time": times, "symbol": "USD/COP", "open": close - 0.2,
                         "high": close + 1, "low": close - 1, "close": close})


@pytest.fixture
def lane(monkeypatch):
    target = pd.Timestamp("2023-06-01").date()
    day = bars_for(target)
    history = pd.concat([bars_for(d.date()) for d in
                         pd.bdate_range(end="2023-05-31", periods=90)], ignore_index=True)
    state = {"seed": pd.concat([history, day], ignore_index=True), "reads": 0}

    def read(*args, **kwargs):
        state["reads"] += 1
        return state["seed"].copy(deep=True)

    # The two builders see the same inputs, not different mocks of their formulas.
    monkeypatch.setattr(live.pd, "read_parquet", read)

    def macro_daily(daily):
        return daily.assign(dxy_ret=0.001, brent_ret=-0.001)

    monkeypatch.setattr(regime_hmm, "_attach_macro", macro_daily)
    monkeypatch.setattr(live, "attach_macro_features", lambda dates:
                        pd.DataFrame(0.01, index=pd.to_datetime(dates), columns=live.MACRO_FEATURES))
    dimension = len(regime_hmm.FEATURE_NAMES)
    model = PortableRegimeModel(
        k=3, startprob=np.full(3, 1 / 3), transmat=np.full((3, 3), 1 / 3),
        means=np.array([np.full(dimension, m) for m in (-0.2, 0, 0.2)]),
        covars=np.array([np.eye(dimension)] * 3), std_means=np.zeros(dimension),
        std_scales=np.ones(dimension), vol_order=(0, 1, 2), labels=("a", "b", "c"),
        feature_names=regime_hmm.FEATURE_NAMES, fit_range=("fixture", "fixture"),
    )
    scaler = live.FrozenScaler(np.zeros(len(live.MARKET_FEATURES)),
                               np.ones(len(live.MARKET_FEATURES)), tuple(live.MARKET_FEATURES))
    return SimpleNamespace(target=target, day=day, history=history, state=state,
                           kwargs={"scaler": scaler, "regime": model})


@pytest.mark.parametrize("n", [1, 11, 30, 59, 60])
def test_current_prefix_does_not_need_target_session_in_seed(lane, n):
    full = live.build_live_spec(lane.target, m5=lane.state["seed"], **lane.kwargs)
    lane.state.update(seed=lane.history, reads=0)
    prefix = live.build_live_spec_partial(lane.target, lane.day.iloc[:n], **lane.kwargs)
    assert lane.state["reads"] == 1, "no second read may inspect full-session admission"
    np.testing.assert_array_equal(prefix.market, full.market[:n])
    np.testing.assert_array_equal(prefix.context, full.context)
    np.testing.assert_array_equal(prefix.closes, full.close[:n])
    assert prefix.spread_pips == full.spread_pips


def test_bad_unobserved_tail_cannot_rewrite_or_remove_prefix(lane):
    prefix = lane.day.iloc[:11]
    before = live.build_live_spec_partial(lane.target, prefix, **lane.kwargs)
    future = lane.day.copy()
    future.loc[11:, ["open", "high", "low", "close"]] = np.nan
    lane.state["seed"] = pd.concat([lane.history, future, bars_for("2023-06-02")], ignore_index=True)
    after = live.build_live_spec_partial(lane.target, prefix, **lane.kwargs)
    np.testing.assert_array_equal(after.market, before.market)
    np.testing.assert_array_equal(after.context, before.context)
    assert after.spread_pips == before.spread_pips


def test_complete_builder_uses_supplied_history_not_unrelated_global_seed(lane):
    supplied = lane.state["seed"].copy()
    baseline = live.build_live_spec(lane.target, m5=supplied, **lane.kwargs)
    lane.state.update(seed=lane.history.iloc[:0], reads=0)
    actual = live.build_live_spec(lane.target, m5=supplied, **lane.kwargs)
    assert lane.state["reads"] == 0
    np.testing.assert_array_equal(actual.market, baseline.market)
    np.testing.assert_array_equal(actual.context, baseline.context)


@pytest.mark.parametrize("n", [1, 11, 59])
def test_utc_and_cot_represent_the_same_prefix(lane, n):
    frame = lane.day.iloc[:n].copy()
    cot = live.build_live_spec_partial(lane.target, frame, **lane.kwargs)
    frame["time"] = frame["time"].dt.tz_convert("UTC")
    utc = live.build_live_spec_partial(lane.target, frame, **lane.kwargs)
    np.testing.assert_array_equal(utc.market, cot.market)
    np.testing.assert_array_equal(utc.context, cot.context)


def test_mixed_known_timezones_represent_the_same_instants(lane):
    frame = lane.day.iloc[:11].copy()
    expected = live.build_live_spec_partial(lane.target, frame, **lane.kwargs)
    frame["time"] = frame["time"].astype(object)
    frame.loc[3, "time"] = frame.loc[3, "time"].tz_convert("UTC")
    actual = live.build_live_spec_partial(lane.target, frame, **lane.kwargs)
    np.testing.assert_array_equal(actual.market, expected.market)
    np.testing.assert_array_equal(actual.context, expected.context)


def test_equivalent_instants_are_duplicate_bars_even_in_different_zones(lane):
    frame = lane.day.iloc[:11].copy()
    frame["time"] = frame["time"].astype(object)
    frame.loc[3, "time"] = frame.loc[2, "time"].tz_convert("UTC")
    with pytest.raises(ValueError):
        live.build_live_spec_partial(lane.target, frame, **lane.kwargs)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_storage_resolution_does_not_change_instants_or_features(lane, unit):
    frame = lane.day.iloc[:11].copy()
    expected = live.build_live_spec_partial(lane.target, frame, **lane.kwargs)
    frame["time"] = frame["time"].dt.as_unit(unit)
    actual = live.build_live_spec_partial(lane.target, frame, **lane.kwargs)
    np.testing.assert_array_equal(actual.market, expected.market)
    np.testing.assert_array_equal(actual.context, expected.context)


@pytest.mark.parametrize("value", [True, np.bool_(True), 4000 + 1j, np.complex128(4000 + 1j),
                                  pd.Timedelta(4000, unit="ns"), pd.Timestamp("2023-06-01")])
def test_non_price_scalar_types_are_rejected_before_numeric_conversion(lane, value):
    frame = lane.day.iloc[:11].copy()
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.Series([value] * len(frame), index=frame.index, dtype=object)
    with pytest.raises(ValueError, match="OHLC|price"):
        live._validated_prefix(lane.target, frame)


def test_mixed_boolean_quote_cannot_become_price_one(lane):
    frame = lane.day.iloc[:11].copy()
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.Series([True] + [4000.0] * 10, index=frame.index, dtype=object)
    with pytest.raises(ValueError, match="OHLC|price"):
        live._validated_prefix(lane.target, frame)


@pytest.mark.parametrize("value", [pd.Timedelta(4000, unit="ns"), pd.Timestamp("2023-06-01")])
def test_native_datetime_quote_column_is_not_a_price(lane, value):
    frame = lane.day.iloc[:11].copy()
    for column in ("open", "high", "low", "close"):
        frame[column] = value
    with pytest.raises(ValueError, match="OHLC|price"):
        live._validated_prefix(lane.target, frame)


@pytest.mark.parametrize("value", ["4000.25", Decimal("4000.25"), np.float64(4000.25)])
def test_numeric_quote_representations_remain_accepted(lane, value):
    frame = lane.day.iloc[:11].copy()
    for column in ("open", "high", "low", "close"):
        frame[column] = value
    actual = live._validated_prefix(lane.target, frame)
    np.testing.assert_array_equal(actual["close"].to_numpy(dtype=float), np.full(11, 4000.25))


@pytest.mark.parametrize("defect", ["naive", "nan_time", "off_grid", "missing_open_bar",
                                    "gap", "duplicate", "next_day", "wrong_symbol",
                                    "nan_price", "infinite_price", "negative_price",
                                    "bad_high", "missing_ohlc", "unsorted"])
def test_malformed_prefix_rejected_before_feature_computation(lane, monkeypatch, defect):
    frame = lane.day.iloc[:11].copy()
    if defect == "naive":
        frame["time"] = frame["time"].dt.tz_localize(None)
    elif defect == "nan_time":
        frame.loc[3, "time"] = pd.NaT
    elif defect == "off_grid":
        frame.loc[3, "time"] += pd.Timedelta(seconds=1)
    elif defect == "missing_open_bar":
        frame = frame.iloc[1:]
    elif defect == "gap":
        frame = frame.drop(index=3)
    elif defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[-1:]])
    elif defect == "next_day":
        frame.loc[3, "time"] += pd.Timedelta(days=1)
    elif defect == "wrong_symbol":
        frame.loc[3, "symbol"] = "XAU/USD"
    elif defect == "missing_ohlc":
        frame = frame.drop(columns="high")
    elif defect == "unsorted":
        frame = frame.iloc[::-1]
    else:
        column, value = {"nan_price": ("close", np.nan),
                         "infinite_price": ("close", np.inf),
                         "negative_price": ("low", -1),
                         "bad_high": ("high", 3000)}[defect]
        frame.loc[3, column] = value
    calls = []
    original = live.build_market_features

    def features(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(live, "build_market_features", features)
    with pytest.raises(ValueError):
        live.build_live_spec_partial(lane.target, frame, **lane.kwargs)
    assert calls == [], "invalid quotes must not become zero-filled market features"


@pytest.mark.parametrize("day", ["2023-06-03", "2023-07-04", "2023-07-20"])
def test_calendar_admission_is_known_before_session_completes(lane, day):
    with pytest.raises(ValueError, match="calendar|calendario|festivo|weekend"):
        live.build_live_spec_partial(day, bars_for(day).iloc[:1], **lane.kwargs)


def test_no_input_frame_is_mutated(lane):
    original_seed = lane.state["seed"].copy(deep=True)
    frame = lane.day.iloc[:11].copy(deep=True)
    original_frame = frame.copy(deep=True)
    live.build_live_spec_partial(lane.target, frame, **lane.kwargs)
    pd.testing.assert_frame_equal(lane.state["seed"], original_seed)
    pd.testing.assert_frame_equal(frame, original_frame)


def test_symbol_optional_in_scoped_usdcop_prefix_is_not_lost_on_concat(lane):
    frame = lane.day.iloc[:11]
    baseline = live.build_live_spec_partial(lane.target, frame, **lane.kwargs)
    actual = live.build_live_spec_partial(lane.target, frame.drop(columns="symbol"), **lane.kwargs)
    np.testing.assert_array_equal(actual.market, baseline.market)
    np.testing.assert_array_equal(actual.context, baseline.context)


def test_insufficient_history_remains_rejected(lane):
    lane.state["seed"] = lane.history.iloc[:60]
    with pytest.raises(ValueError, match="contexto"):
        live.build_live_spec_partial(lane.target, lane.day.iloc[:1], **lane.kwargs)


def test_session_spread_uses_supplied_past_not_global_mask(lane):
    past = lane.history.copy()
    expected = live.session_spread(lane.target, m5=past, regime=lane.kwargs["regime"])
    lane.state.update(seed=past.iloc[:0], reads=0)
    actual = live.session_spread(lane.target, m5=past, regime=lane.kwargs["regime"])
    assert lane.state["reads"] == 0
    assert actual == expected


def test_dataframe_mask_preserves_existing_exclusion_semantics():
    # Full valid, partial weekday, Colombia holiday and USA holiday: no novel filters.
    frame = pd.concat([bars_for("2023-06-01"), bars_for("2023-06-02").iloc[:11],
                       bars_for("2023-07-20"), bars_for("2023-07-04")], ignore_index=True)
    actual = masks._mask_from_frame(frame, source="fixture")
    assert actual.valid == (pd.Timestamp("2023-06-01").date(),)
    assert actual.excluded == {
        "holiday": (pd.Timestamp("2023-07-20").date(),),
        "us_holiday": (pd.Timestamp("2023-07-04").date(),),
        "incomplete": (pd.Timestamp("2023-06-02").date(),),
    }
    assert actual.source == "fixture"


def test_real_2023_quotes_and_macro_prefix_match_full_without_today_in_seed(lane, monkeypatch):
    # Retrospective engineering check on real inputs, with non-fitted fixture
    # model/scaler. It is not a replay of a traded PPO or evidence of alpha/PIT.
    monkeypatch.undo()
    read = pd.read_parquet
    real = read(live.SEED_M5)
    dates = pd.to_datetime(real["time"]).dt.date
    bounded = real[dates <= lane.target]
    history = real[dates < lane.target]
    day = real[dates == lane.target]
    assert len(day) == 60
    full = live.build_live_spec(lane.target, m5=bounded, **lane.kwargs)
    seed_reads = []

    def without_today(path, *args, **kwargs):
        if path == live.SEED_M5:
            seed_reads.append(True)
            return history.copy(deep=True)
        return read(path, *args, **kwargs)

    monkeypatch.setattr(live.pd, "read_parquet", without_today)
    for n in (1, 11, 59):
        prefix = live.build_live_spec_partial(lane.target, day.iloc[:n], **lane.kwargs)
        np.testing.assert_array_equal(prefix.market, full.market[:n])
        np.testing.assert_array_equal(prefix.context, full.context)
        np.testing.assert_array_equal(prefix.closes, full.close[:n])
        assert prefix.spread_pips == full.spread_pips
    assert len(seed_reads) == 3
