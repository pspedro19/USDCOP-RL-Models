# Emerging Market FX

The gap this fills: across 118 skills, EM FX was four lines calling USD/MXN,
USD/ZAR and USD/TRY "exotic pairs — wider spreads." USDBRL and USDINR did not
appear at all, and carry content was G10-only.

EM is not "G10 with wider spreads." The differences are structural and they
change position sizing, not just execution assumptions.

---

## 1. The carry trade is the trade

EM FX exists in a quant book primarily as a **carry** asset. The mechanics are
the same as AUD/JPY, but the magnitudes and the tail are not.

Long the high-yielder, funded in the low-yielder, earns the rate differential.
The engine handles the sign automatically: `USDMXN` quotes dollars per peso, so
the peso carry trade is **short USDMXN**, which computes as a positive carry to
the short side.

```python
from xasset.carry import fx_carry
fx_carry("USDMXN", rate_base=0.0433, rate_quote=0.1025, is_em=True,
         annual_inflation_base=0.0290)
# -5.92%/yr to a long -> +5.92%/yr to the short (the peso carry trade)
```

### Nominal carry versus real carry

This is the distinction that separates a carry trade from a slow-motion loss.

| | Nominal carry | Inflation | Real carry | Verdict |
|---|---|---|---|---|
| MXN (typical) | ~10% | ~4% | **+6%** | Genuine compensation |
| TRY (2021–23) | ~15–45% | 40–85% | **deeply negative** | Paid to hold a melting asset |

A currency with high nominal rates and higher inflation is not offering edge.
It is offering compensation for expected depreciation, and UIP failure — the
thing that makes carry profitable in G10 — does not reliably hold there.

The engine marks these `reliability="suspect"` and **excludes them from the
cross-sectional z-score by default**. Without that, a single 40%/yr TRY carry
dominates the ranking and pulls the entire book toward the one position most
likely to gap.

### Prefer forward-implied carry

Where capital controls or NDF markets bind, covered interest parity breaks.
The onshore rate differential can show +12% while the tradeable offshore
forward shows +6%. The forward is what you can transact on.

```python
from xasset.carry import fx_carry_from_forward
fx_carry_from_forward("USDBRL", spot=5.09, forward=5.24, days=90)
```

---

## 2. Negative skew is the defining risk

Carry returns are not normally distributed. The shape is: small steady gains,
punctuated by violent losses concentrated in risk-off episodes, with every EM
currency moving together.

Consequences that matter mechanically:

- **Sharpe overstates quality.** The metric assumes symmetric risk. Use
  `sharpe_ratio_stderr`, which applies the Lo/Mertens expansion and widens the
  error bar for skew and kurtosis, rather than a bare Sharpe.
- **Volatility targeting understates tail risk.** Realised vol is measured in
  the calm period that precedes the gap — that is the whole problem. Treat the
  vol-targeted weight as a **ceiling**, not a recommendation.
- **Correlations go to 1 when it matters.** MXN, ZAR, BRL and TRY are
  approximately one trade during a dollar squeeze. Diversification across five
  EM currencies is far less diversification than the covariance matrix,
  estimated on calm data, will tell you.
- **Stops gap through.** A stop-loss is not a risk limit in EM. Position size
  is. Assume the stop fills materially worse than its level.

---

## 3. Deliverable, NDF and controlled

Not all EM currencies trade the same way, and the engine flags this.

| Pair | Type | Notes |
|---|---|---|
| USDMXN | Deliverable | Most liquid EM cross, ~24h. The cleanest EM expression |
| USDZAR | Deliverable | Liquid but thin; high beta to global risk |
| USDBRL | **NDF** offshore | Cash-settled. BCB intervenes via FX swaps |
| USDINR | **NDF**, controlled | RBI manages the rate; realised vol understates true risk |
| USDCNH | Offshore proxy | CNH ≠ CNY. PBoC fixes onshore daily; the two can diverge |
| USDTRY | Controlled | Repeated policy shocks; carry has been erased by step devaluations |

**Non-deliverable forwards** settle in dollars against a fixing. There is no
physical delivery, the fixing methodology is a real risk (fixings have been
disputed and changed), and offshore NDF pricing can decouple from the onshore
rate during stress.

**Managed regimes are the trap that low volatility sets.** A central bank
holding a currency stable produces a low realised volatility, which any
vol-targeting system reads as low risk and rewards with a large position. The
risk did not go away — it moved into a discrete jump when the peg breaks. When
`capital_controls=True`, do not let realised vol drive size.

---

## 4. Value in EM: use PPP carefully

`fx_value_ppp` marks EM estimates `confidence="weak"`, deliberately.

The **Balassa-Samuelson effect** means faster-growing economies sustain
genuinely stronger real exchange rates: productivity gains concentrate in
tradables, wages rise economy-wide, non-tradables get more expensive, and the
real exchange rate appreciates as an *equilibrium* outcome. A persistent PPP
"overvaluation" in a fast-growing EM economy is often fair value, not
mispricing.

PPP in EM is a sanity check against extreme positions, not a signal to trade.

---

## 5. Routes into the market

| Route | Pros | Cons |
|---|---|---|
| **Spot/CFD** | Direct, small size, 24h | Broker-dependent pricing; swap rates worse than interbank; counterparty risk |
| **CME futures** (6M peso, 6L real) | Exchange-cleared, transparent, verifiable specs | Larger minimum size; quoted **inverted** vs spot convention |
| **NDF** | The institutional route for BRL/INR | Not retail-accessible; ISDA required |
| **EM ETFs** (EEM, EMB, CEW) | Simple, no leverage | Equity/credit beta contaminates the FX view |

**The CME inversion is an easy, expensive mistake.** Spot convention is USDMXN
≈ 17.41 (pesos per dollar). The CME 6M future is quoted USD per MXN ≈ 0.0574.
They move in *opposite* directions. Long 6M is long the peso, which is short
USDMXN. The registry carries this in `notes`; check the sign before trading.

---

## 6. Free data reality

`yfinance` covers USDMXN, USDBRL, USDZAR, USDTRY, USDINR and USDCNH as daily
closes — verified working. Real limitations:

- Indicative closes, not tradeable rates. Spot bid-ask in EM is wide and time-varying, and none of that is in the data.
- No intraday. EM liquidity is heavily concentrated in local hours; a daily close hides it entirely.
- **No forward points or NDF curves anywhere free.** This is the real gap: forward-implied carry is the right input, and it needs a paid source.
- Local policy rates are not on FRED. Sourcing those is a manual step.

Do not backtest EM execution on this data. It supports signal research on daily
horizons; it does not support any claim about fills.

---

## Further reading

- Koijen, Moskowitz, Pedersen & Vrugt (2018), "Carry" — the cross-asset framework this module implements
- Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes" — the skew mechanism
- Menkhoff, Sarno, Schmeling & Schrimpf (2012), "Carry Trades and Global FX Volatility" — why carry loads on vol risk
- BIS Triennial Survey — EM turnover, NDF volumes, market structure
