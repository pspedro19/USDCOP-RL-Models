# Strategic Assessment — 2026-07 (honest state of the system + forward plan)

> **Purpose.** A grounded, evidence-based read of where the system actually stands after the 2026-07
> data-engineering hardening (Gold calendar fix, OHLCV validators, Deflated Sharpe + true OOS). Written to
> be *uncomfortably honest* so future work targets the real bottleneck — **evidence/alpha**, not more
> plumbing. Companion to `AUDIT-2026-07-remediation.md` (code↔spec findings) and the per-asset DS-cycle
> conclusions (`../assets/_ds-cycle-asbuilt.md §8, §10`).
>
> Every claim here is traceable to code or a measured result, cited inline.

---

## 0. The meta-problem (the one that contains the others)

**The infrastructure is far ahead of the strength of the signal.** 25+ Docker services, ~29 DAGs, an
immutable versioned registry, TS/Python contracts, a multi-asset dashboard — sitting on top of models with
**R² < 0** (`_ds-cycle §8`, CLAUDE.md) and strategies whose edge is fragile out-of-sample (BTC's promoted
`btc_trend_b2` went **+351% full-history → −1.4% OOS-2025**; Gold's promoted `gold_trend_b2` has **DSR 0.921
< 0.95**). The sophistication is in the plumbing, not the alpha.

**Implication for prioritization:** the highest-value work is *not* another feature, model, or service. It
is (a) validating that a real edge exists, (b) measuring it honestly at the small sample sizes we have, and
(c) narrowing the system around the one thing that demonstrably works (risk control), while feeding the one
asset (BTC) the data that could actually give it predictive alpha.

---

## 1. Data quality & training data — the foundation is thin

| Problem | Evidence | Consequence |
|---|---|---|
| No end-to-end OHLCV validation (until now) | Gold day-shift bug shipped undetected for months; nothing checked weekday coverage / bars-per-period | Every Gold metric was silently biased. **Partially fixed**: `src/data_quality/ohlcv_validators.py` now gates ingest (weekday/gap/tz/OHLC). Macro + feature paths still lack equivalent gates. |
| Tiny training sets | USD/COP forecasting = **1,686 daily bars** (2020→2026) with 21 features × 9 models; Gold 5-min seed only spans **2026-03→** | Over-fitting near-guaranteed for the ML zoo; no real intraday history for Gold |
| Tiny trade samples | H5 2025 = **34 trades**, 2026 = **1 trade** | No statistical power; strong claims are unsupported (mitigated by DSR/OOS, but N is the hard limit) |
| DB vs parquet duality with no owner | `asset_daily_ohlcv` is written best-effort but **read only by the backup job**; the serving path is 100% parquet→JSON (schema-agent finding) | Two "sources of truth" that can drift; the DB is largely ceremonial for multi-asset |
| Crypto-native data designed but empty | `database/migrations/052_crypto_native_data.sql` defines 5 tables (onchain/derivatives/flows/events/exposure); **no extractor exists** (grep: zero ingest scripts) | BTC model is blind on exactly the data that gives crypto its edge (see §4) |

**Direction:** one source of truth per concern (decide DB *or* parquet, not both half-alive); a validation
gate at *every* boundary (OHLCV done; macro/features next); backfill real history before trusting any
intraday model.

---

## 2. Models — the edge is risk control, not prediction

This is the most important honest conclusion and it is already on record:

- **USD/COP:** the 10-agent audit found **Ridge/BR R² < 0 in both years**. The 9-model zoo's directional
  accuracy is **48.6–54.5%** — a coin flip. The production edge is the **regime gate (knowing when NOT to
  trade) + TP/HS mechanics**, not the point predictions (`smart_simple_v11`).
  > **Trial-aware update (2026-07-06, G2):** the celebrated p=0.0097/p=0.006 came from iterating on the
  > SAME OOS-2025 (42-cell grid, "#8 of 42"). Retroactive trial count (N≈44-70) gives **v11 DSR =
  > 0.50–0.92 < 0.95 in ALL scenarios** (`scripts/analysis/cop_trials_dsr.py`) — held to the same bar as
  > Gold/BTC, v11's backtest does not prove edge after selection. **v11 is FROZEN; the clean judge is the
  > 2026 forward** under `../assets/usdcop/WITHDRAWAL-PROTOCOL.md`. Full ledger:
  > `../assets/usdcop/HYPOTHESIS-REGISTRY.md`.
- **Gold/BTC:** rule-based, no ML. After trial-aware correction (added 2026-07), Gold's promoted
  `gold_trend_b2` **does not clear DSR > 0.95** (0.921), and BTC's `btc_trend_b2` is **negative OOS-2025**.
  The strong headlines are dominated by past trending regimes (Gold 2025, BTC 2017/2021 bulls).

**What this means:** call the thing what it is — a **risk-control strategy** (regime gate × vol-target ×
stops), not a forecasting system. That is a legitimate, defensible edge. Stop investing in the ML zoo as if
it predicts; keep it as a diagnostic surface. Any future model claim must clear **DSR > 0.95 + a positive
OOS slice**, not a lone bootstrap p-value.

---

## 3. Orchestration — much ceremony, partial use

- **Paused/deprioritized tracks:** H1 daily (5 DAGs, `is_paused_upon_creation=True`), RL (not significant,
  p=0.272). Of ~29 DAGs, many are manual/event-triggered.
- **Silent-skip DB steps:** ingest/publish DB writes are best-effort/`graceful:true` and fail quietly, so
  the DB cannot be trusted to reflect reality.
- **Under-used infra:** MLflow only invoked by H5-L3; MinIO only by init-scripts; Jaeger instrumented but
  barely exercised. Deployed ≠ used.

**Direction:** retire/park what doesn't earn its keep; make the remaining DB writes fail-loud *or* drop the
DB from the multi-asset path entirely and own the file-driven model.

---

## 4. BTC deep-dive — why it's *fragile*, not "bad", and what it needs

**BTC is not a money-loser — it's regime-dependent and unvalidated out-of-sample.**

- Full-history `btc_trend_b2`: **+351%, Sharpe 1.40, DSR 0.999** — but that's the 2017/2020-21 bull cycles.
  **OOS-2025: −1.4%, Sharpe −0.05, p=0.62.** A pure price-momentum system gets whipsawed in the 2025 range.

**Root cause = features.** `src/btc_strategy/indicators.py::build_daily_features` is **100% price-derived**
(SMA 20/50/100/200, ADX, ATR, Hurst, realized-vol, z-vs-SMA). In crypto the alpha is largely **non-price**:

| Category | Signals | Status |
|---|---|---|
| **Derivatives** | funding rate (deep), open interest / long-short (30d fwd), basis | ✅ **INGESTED (2026-07-06)** — `scripts/data/ingest_btc_derivatives.py` → `crypto_derivatives_daily` (2492 funding days) + seed; DAG `l0c_ingest_derivatives`. Liquidations still ❌ (no public REST) |
| **On-chain** | exchange netflows, MVRV / realized price, stablecoin supply, active addresses, miner flows | ❌ not ingested |
| **Price technicals** | SMA / ADX / ATR / Hurst | ✅ present |

The system already knew this: `migration 052` defines exactly these tables and the BTC roadmap Fase 1 lists
`BGeo/funding/Farside/DefiLlama` extractors. **Derivatives ingestion is now BUILT** (funding/OI/long-short →
`crypto_derivatives_daily`, plan `../assets/btcusdt/PLAN-binance-derivatives-2026-07.md`); on-chain + the
features/strategy consuming the derivatives are still pending. The regime-gated S3 (the intended "good"
strategy) still *"needs the on-chain HMM"* fed by the remaining on-chain extractors.

**Structural limits beyond features:** spot-only long/flat exposure ∈ [0,1] (can't profit in down/chop years,
best case is sitting out); fixed-threshold rules (no ML, no adaptation); price-only regime classifier
(crypto regime is better defined by MVRV cycle + funding/OI positioning).

### BTC forward plan (ordered by impact / cost)

1. **Derivatives extractor first (cheap, high-ROI).** Funding rate + open interest + liquidations from the
   public Binance/exchange APIs (no paid key) → populate `crypto_derivatives_daily` → add as features
   (`z_funding`, `oi_change`, `liq_imbalance`). This alone tells you if there's real orthogonal signal
   before spending on on-chain. **This is the single biggest lever.**
   > **Concrete plan: [`../assets/btcusdt/PLAN-binance-derivatives-2026-07.md`](../assets/btcusdt/PLAN-binance-derivatives-2026-07.md).**
   > Honest API reality: **only funding has deep history (~2019→now) → backtesteable**; OI/long-short/
   > taker are **forward-only (30-day window)**; liquidations have no public REST (WS/Coinglass). So the
   > first deliverable is `funding_rate → z_funding` + `btc_trend_funding_s4`, gated by DSR>0.95 ∧ OOS-2025+.
2. **On-chain extractor (deeper edge, harder/costlier).** Exchange netflows, stablecoin supply, MVRV/realized
   price → `crypto_onchain_daily`. The genuine crypto alpha, but data is paid/lagged (Glassnode-style / free
   tiers like BGeometrics).
3. **On-chain HMM regime** (roadmap Fase 3, SPEC-01) — replace the price-only regime with an MVRV+funding
   4-state HMM.
4. **Defensive/short capability** — a real sit-out or short leg so bad years aren't just "flat".
5. **Multi-cycle walk-forward evaluation** — never trust one full-history number dominated by past bulls;
   DSR + OOS-2025 already expose the fragility. Use them as the promotion gate.

**Honest verdict:** with **price-only features, no model (ML or rules) will robustly beat HODL** in BTC —
daily price noise is too high (ann.vol 68%, kurtosis 15.7). If the crypto-native data won't be ingested, the
honest posture is to treat BTC as **HODL + vol-targeting** (`btc_hodl_b1`) and stop implying a predictive
model exists.

---

## 5. Frontend / backend — two different worlds

- The dashboard is **file-driven** (reads `public/data` + `public/forecasting` via read-only mounts). The
  API layer is a BFF over files + some DB-live + a proxy. Reasonable, but it means most of what you see is
  **batch-generated JSON, not live** — the "real-time" story is thin.
- **Contracts** (`lib/contracts/*.ts` ↔ `src/contracts/*.py`) are synced **by convention, not enforced** →
  they can silently drift.
- **DB tables the "backend" implies are mostly empty for multi-asset** → backend/DB and frontend/files are
  effectively two systems. Observed drift: a v1.0.0 manifest headline got rewritten while its immutable
  `summary.json` stayed stale (harmless because the active version moved to 1.1.0, but symptomatic).

**Direction:** either make the DB the real serving source or fully embrace file-driven and stop pretending
the DB matters for multi-asset; add a contract-parity check (schema snapshot test) so TS/Python can't drift.

---

## 6. Priorities (what to actually do next)

| # | Action | Why (impact) | Cost |
|---|---|---|---|
| **P1** | Prove alpha before scaling: adopt **DSR > 0.95 + positive OOS** as the promotion bar everywhere | Stops over-claiming; the ML zoo (DA ~50%) likely fails it honestly | Low (mostly done for gold/btc) |
| **P2** | **BTC derivatives extractor** (funding/OI/liquidations → `crypto_derivatives_daily` → features) | The one change that could give BTC real predictive signal; cheap/free data | Medium |
| **P3** | Validation gates at every data boundary (macro + features, mirroring the OHLCV gate) | Prevents the next silent-corruption class of bug | Low-Medium |
| **P4** | Collapse the DB/parquet duality → one source of truth per concern | Removes the drift surface; simplifies everything downstream | Medium |
| **P5** | Narrow scope: park/retire what doesn't earn its keep (ML zoo as diagnostic only, idle infra) | Less surface = less spec↔reality drift, cheaper to operate | Low |
| **P6** | Backfill real history (Gold intraday, more depth) before trusting any intraday model | No data, no model | Medium |

**One-sentence conclusion:** the system does not have an engineering problem — it has an **evidence
problem**. The data entering it wasn't validated end-to-end, the sample sizes are small, and the measurable
alpha is risk-control, not prediction. The most valuable next move is to **narrow the system around what
demonstrably works and feed BTC the non-price data it needs**, not to add more features or services.

---

## Related

- `AUDIT-2026-07-remediation.md` — code↔spec findings backlog (P0/P1/P2)
- `../assets/_ds-cycle-asbuilt.md §8, §10` — per-asset conclusions + systemic read
- `../assets/_strategy-science.md §5` — trial-aware evaluation (DSR/OOS) methodology
- `../assets/btcusdt/IMPLEMENTATION_ROADMAP.md` — BTC phased plan (Fase 1 crypto-native = the unblocker)
- `../../rules/data-governance.md`, `../../rules/data-freshness.md` — OHLCV validator contract (CTR-DQ-OHLCV-001)
