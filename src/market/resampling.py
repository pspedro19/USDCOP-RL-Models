"""Deterministic, session-anchored OHLCV resampling.

The resampler is intentionally pure: it never guesses missing source bars and it
never consults wall-clock time.  A target bar is emitted only when the complete
set of consecutive source observations exists inside the configured trading
session.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Iterable
from zoneinfo import ZoneInfo

from src.contracts.asset_profile import AssetProfile


_INTERVALS: dict[str, timedelta] = {
    "PT1M": timedelta(minutes=1),
    "PT5M": timedelta(minutes=5),
    "PT15M": timedelta(minutes=15),
    "PT30M": timedelta(minutes=30),
    "PT1H": timedelta(hours=1),
    "PT4H": timedelta(hours=4),
    "P1D": timedelta(days=1),
}


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _require_decimal(value: Decimal, field: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{field} must be a finite Decimal")


def _parse_session_time(value: str | None, *, default: time) -> time:
    if value is None:
        return default
    try:
        hour, minute = (int(part) for part in value.split(":", maxsplit=1))
        return time(hour=hour, minute=minute)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid session time {value!r}; expected HH:MM") from exc


@dataclass(frozen=True, slots=True)
class SourceBar:
    """One immutable raw observation used as resampling evidence."""

    raw_bar_id: str
    instrument_id: str
    event_time: datetime
    available_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.raw_bar_id:
            raise ValueError("raw_bar_id must be non-empty")
        if not self.instrument_id:
            raise ValueError("instrument_id must be non-empty")
        _require_aware(self.event_time, "event_time")
        _require_aware(self.available_at, "available_at")
        if self.available_at < self.event_time:
            raise ValueError("available_at cannot precede event_time")
        for field in ("open", "high", "low", "close"):
            _require_decimal(getattr(self, field), field)
        if self.volume is not None:
            _require_decimal(self.volume, "volume")
            if self.volume < 0:
                raise ValueError("volume cannot be negative")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high violates OHLC ordering")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low violates OHLC ordering")


@dataclass(frozen=True, slots=True)
class ResampledBar:
    """Canonical bar plus the ordered raw-bar lineage used to build it."""

    instrument_id: str
    interval_id: str
    event_time: datetime
    available_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None
    source_raw_bar_ids: tuple[str, ...]
    bar_method: str = "resampled"


@dataclass(frozen=True, slots=True)
class ResamplePolicy:
    """Asset-specific resampling contract."""

    asset_id: str
    timezone: str
    session_days: tuple[int, ...]
    session_open: time
    session_close: time
    source_interval: str
    target_interval: str
    source_duration: timedelta
    target_duration: timedelta
    source_bars_per_target: int
    compute_latency: timedelta
    is_24x7: bool

    @classmethod
    def for_asset(
        cls,
        profile: AssetProfile,
        *,
        source_interval: str,
        target_interval: str,
        compute_latency: timedelta = timedelta(0),
    ) -> "ResamplePolicy":
        try:
            source_duration = _INTERVALS[source_interval]
            target_duration = _INTERVALS[target_interval]
        except KeyError as exc:
            raise ValueError(f"unsupported interval {exc.args[0]!r}") from exc
        if target_duration <= source_duration:
            raise ValueError("target interval must be larger than source interval")
        quotient, remainder = divmod(
            int(target_duration.total_seconds()),
            int(source_duration.total_seconds()),
        )
        if remainder:
            raise ValueError("target interval must be an exact source-interval multiple")
        if compute_latency < timedelta(0):
            raise ValueError("compute_latency cannot be negative")
        try:
            ZoneInfo(profile.session.timezone)
        except Exception as exc:  # pragma: no cover - platform tzdata failures
            raise ValueError(
                f"unknown session timezone {profile.session.timezone!r}"
            ) from exc

        return cls(
            asset_id=profile.asset_id,
            timezone=profile.session.timezone,
            session_days=profile.session.days,
            session_open=_parse_session_time(
                profile.session.open, default=time.min
            ),
            session_close=_parse_session_time(
                profile.session.close, default=time.max
            ),
            source_interval=source_interval,
            target_interval=target_interval,
            source_duration=source_duration,
            target_duration=target_duration,
            source_bars_per_target=quotient,
            compute_latency=compute_latency,
            is_24x7=profile.session.is_24x7,
        )

    def bucket_start(self, event_time: datetime) -> datetime | None:
        """Return the session-anchored bucket start, or ``None`` off-session."""

        _require_aware(event_time, "event_time")
        local = event_time.astimezone(ZoneInfo(self.timezone))
        if local.weekday() not in self.session_days:
            return None

        session_start = datetime.combine(
            local.date(), self.session_open, tzinfo=local.tzinfo
        )
        if self.is_24x7:
            session_end = session_start + timedelta(days=1)
        else:
            session_end = datetime.combine(
                local.date(), self.session_close, tzinfo=local.tzinfo
            ) + self.source_duration

        if local < session_start or local >= session_end:
            return None
        offset = local - session_start
        bucket_index = offset // self.target_duration
        start = session_start + (bucket_index * self.target_duration)
        if start + self.target_duration > session_end:
            return None
        return start.astimezone(event_time.tzinfo)


def _complete_bucket(
    bars: list[SourceBar],
    *,
    bucket_start: datetime,
    policy: ResamplePolicy,
) -> bool:
    if len(bars) != policy.source_bars_per_target:
        return False
    expected = tuple(
        bucket_start + index * policy.source_duration
        for index in range(policy.source_bars_per_target)
    )
    observed = tuple(bar.event_time for bar in bars)
    return observed == expected


def resample_complete_bars(
    source_bars: Iterable[SourceBar],
    policy: ResamplePolicy,
) -> tuple[ResampledBar, ...]:
    """Aggregate complete buckets and suppress duplicates/incomplete windows."""

    ordered = sorted(source_bars, key=lambda bar: (bar.event_time, bar.raw_bar_id))
    seen_raw_ids: set[str] = set()
    seen_observations: set[tuple[str, datetime]] = set()
    grouped: dict[tuple[str, datetime], list[SourceBar]] = {}

    for bar in ordered:
        if bar.raw_bar_id in seen_raw_ids:
            raise ValueError(f"duplicate raw_bar_id {bar.raw_bar_id!r}")
        observation = (bar.instrument_id, bar.event_time)
        if observation in seen_observations:
            raise ValueError(
                "multiple source bars for the same instrument/event_time are "
                "ambiguous; canonicalize the provider observation first"
            )
        seen_raw_ids.add(bar.raw_bar_id)
        seen_observations.add(observation)
        bucket_start = policy.bucket_start(bar.event_time)
        if bucket_start is not None:
            grouped.setdefault((bar.instrument_id, bucket_start), []).append(bar)

    output: list[ResampledBar] = []
    for (instrument_id, bucket_start), bars in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        bars.sort(key=lambda bar: bar.event_time)
        if not _complete_bucket(
            bars, bucket_start=bucket_start, policy=policy
        ):
            continue

        volumes = [bar.volume for bar in bars]
        volume = (
            None
            if any(value is None for value in volumes)
            else sum((value for value in volumes if value is not None), Decimal(0))
        )
        output.append(
            ResampledBar(
                instrument_id=instrument_id,
                interval_id=policy.target_interval,
                event_time=bucket_start,
                available_at=max(bar.available_at for bar in bars)
                + policy.compute_latency,
                open=bars[0].open,
                high=max(bar.high for bar in bars),
                low=min(bar.low for bar in bars),
                close=bars[-1].close,
                volume=volume,
                source_raw_bar_ids=tuple(bar.raw_bar_id for bar in bars),
            )
        )

    return tuple(output)
