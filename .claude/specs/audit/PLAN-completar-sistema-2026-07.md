# Plan maestro — Completar el sistema: SignalBridge + auditoría (Partes I–IV)

> **Copia durable en el repo** del plan aprobado 2026-07-06. Contiene: roadmap por olas (OLA 0–8)
> + **APÉNDICE A** (propuesta cuant completa) + **APÉNDICE B** (RBAC + monetización completo).

## ESTADO DE IMPLEMENTACIÓN (ledger — actualizar al avanzar)

| Ola | Ítem | Estado | Evidencia |
|---|---|---|---|
| 0 | G1 constitución transversal | ✅ 2026-07-06 | `.claude/rules/quant-constitution.md` + index |
| 0 | G2 registro trials COP + DSR honesto v11 | ✅ 2026-07-06 | `specs/assets/usdcop/HYPOTHESIS-REGISTRY.md` + `scripts/analysis/cop_trials_dsr.py` — **DSR v11 = 0.50–0.92 < 0.95 en TODOS los escenarios** |
| 0 | G3 protocolo retiro COP + freeze v11 | ✅ 2026-07-06 | `specs/assets/usdcop/WITHDRAWAL-PROTOCOL.md` (W1-W6, cortes 26/52 sem) |
| 0 | G4 queue reconciliada + namespaces SPEC | ✅ 2026-07-06 | `EXPERIMENT_QUEUE.md` (active queue nueva) + `specs/assets/_spec-namespaces.md` |
| 0 | G5 test consistencia PRE-REG↔YAML + superseded | ✅ 2026-07-06 | `tests/regression/test_prereg_yaml_consistency.py` (5 pass) + headers SUPERSEDED en xauusd/config/*.yaml |
| 1 | F1/F2 Vote-2 lee del bundle | ✅ 2026-07-06 | `ForecastingBacktestSection.tsx`: `displayGates=approval.gates` siempre; `isDateFiltered` vs auto-range (fix Gold/BTC); badge "PREVIEW DEL REPLAY"; tsc limpio |
| 1 | F3 rebuild + verificación navegador | ✅ 2026-07-06 | imagen 16:48 desplegada; Playwright: login→/dashboard KPIs+gates del bundle OK, **badge preview ausente por defecto** (fix F1 verificado), /forecasting 3 activos OK; /api/data 200 para los bundles de las 8 estrategias (COP `*_2025`, Gold/BTC `*_2026` v1.1.0); mount rw + write OK; únicos 404s = proxies graceful pre-existentes (inference/models/realtime) |
| 3 | S1 sizing (`quantize_amount` ccxt, fix TICK_SIZE MEXC) | ✅ 2026-07-06 | `adapters/{base,mexc}.py`, desplegado en contenedor |
| 3 | S2/S3 PreTradeGate (riesgo+kill-switch+`trading_mode` gate, fail-safe, PAPER default) | ✅ 2026-07-06 | `app/services/pretrade.py` + wiring en `execute_order` + **8 tests pass** (en contenedor) |
| 3 | S4 loop copia per-user / S5 orden real | ⏳ | S4 con R5; S5 diferido (fondeo+confirmación) |
| 8 | R1/R2 contrato RBAC + middleware + migración 055 | ✅ 2026-07-06 | `lib/contracts/rbac.contract.ts`, `middleware.ts` deny-by-default (+fix x-user-id en request), `055_rbac_monetization.sql` aplicada (entitlements+audit_log append-only+multi-tenant), coverage test 53 APIs/14 páginas verde |
| 8 | R3 gating monetizado | ✅ núcleo | edge-gate sesión en `/data`+`/forecasting`; delay por plan en `/api/data` (`lib/auth/entitlements.ts`); PNG/CSV delay per-file pendiente |
| 8 | R4 nav por rol | ✅ hub + navbar | `app/hub/page.tsx` + `GlobalNavbar.tsx` desde el contrato (`roleHasPermission`); subscriber ve "Señales"; copy RL→Smart Simple; `AuthSessionProvider` añadido al layout raíz |
| 8 | R0 surface audit | ✅ 2026-07-06 | `docs/rbac/PHASE0-surface-audit.md` (generado: `scripts/generate-surface-audit.mjs`, 53 APIs + 14 páginas con permiso efectivo) |
| 8 | R6 billing | ✅ núcleo | `lib/billing/` (DIP, Wompi checkout+webhook SHA256), `/api/billing/{checkout,webhook}`, `/pricing`; faltan env keys + `/account/billing` |
| 8 | R2+ tests contrato + CI | ✅ 2026-07-06 | `scripts/test-rbac-contract.mjs` (17 asserts de semántica) + `check-rbac-coverage.mjs` cableados en `.github/workflows/contracts-check.yml`; fix: `/api/execution/auth` público (el login lo necesita pre-sesión) |
| 8 | R5 multi-tenant endpoints / R7 hardening | ⏳ | tablas listas (055); endpoints me/system + rate-limit + Playwright 4 roles pendientes |
| UX | Spec ux-navigation + gobernanza visual P1/P2 | ✅ 2026-07-06 | `specs/platform/ux-navigation.md`; `lib/contracts/ui.contract.ts` (tokens sobrios AA + `canShowRatios` N<20) + `components/ui/MetricBadge.tsx` (LIVE/BACKTEST/PAPER + tooltip procedencia) |
| UX | `/metodologia` pública (arma de ventas) | ✅ 2026-07-06 | página estática server-component; en matriz RBAC + middleware public |
| UX2 | S1 navbar · S3 TrustBar LIVE-del-bundle · S5/S7 TrackRecord+metodología · S9 FAQ incómodas · S10 /legal/* · /admin console · badges LIVE/BACKTEST cableados | ✅ 2026-07-06 | `/api/public/live-stats` (agregados marketing, public en matriz), `TrustBar/TrackRecord.tsx`, `legal/[doc]`, `/api/admin/overview`+`/admin`; 56 APIs/18 páginas cubiertas; tsc limpio |
| UX3 | **QA visual con screenshots revisados** — 9 hallazgos corregidos | ✅ 2026-07-06 | (1) terminal hero fake PPO→ciclo semanal real; (2) Metrics hardcodeado 3.24/92% eliminado; (3-4) copy RL/LLM/tick-by-tick→ensemble+gates; (5) Early-Access/cupos fake→"Empieza gratis"+/pricing; (6) footer→/legal/*; (7) badges sobrios; (8) centering 5 páginas (CSS global rompía mx-auto); (9) hydration guard hub+navbar (flash 'free' para admin — debug confirmó role:"admin" y 6 cards) |
| UX3b | Segunda ronda QA visual (capturas F1-F8 revisadas) | ✅ 2026-07-06 | residuos "PPO,SAC,A2C"/"tick-by-tick"/"vanguardia ML" eliminados; **"Modelo Activo PPO v2.4/67.3%" del hub (stale original §6.5) → `LiveQuickStats`** (bundle real + badge ●LIVE + "no recomputadas" + canShowRatios); centering template-literal (metodologia/admin); probe DOM: pricing centrado 144|1152|144. Landing desplegada con jerga=0 (grep) |
| UX3 | S6 capturas blur · franja estado semana · wizard auto · bottom-tabs · i18n | ⏳ | pulido incremental |
| 2 | H1-H3 verdict DSR-aware + B1′ + cost-stress | ✅ 2026-07-06 | helpers en `services/common/metrics.py`; wired en backtests gold/btc + pipelines; **`gold_trend_b2` degradado PROMOTE→REVIEW en vivo** (DSR 0.921 + no bate B1′) — I-3 resuelto; bundles v1.2.0 |
| 4 | COP-NULL completo + H-COP-V11-01 | ✅ 2026-07-06 | `scripts/analysis/cop_null_suite.py`: **IC95 ΔCalmar(v11,NULL-A) incluye 0** (H0 no rechazada); **NULL-B Calmar 7.67 > v11 4.19** (beta, no timing); DECOMP: TP/HS restó −18.9pp al short incondicional; STRESS-2122 = −3.1% (daño acotado). Registrado (+4 trials) |
| 5 | B1/B2 funding features + `btc_trend_funding_s4` | ✅ 2026-07-06 | `merge_funding_features` (causal shift(1)) + S4 (k=0.25 ex-ante); **H-POS-01: DSR 0.998 ✓ pero OOS-2025 −1.4% no bate B2 ⇒ NO promueve** (registrado); bundles v1.2.1 (4 estrategias) |
| 6 | X1 XAU-TREND-ENS | ✅ 2026-07-06 | `gold_trend_ens` (SMA 63/126/252 long-flat ex-ante): **160.9% vs B2 55.3%, Sharpe 0.587 vs 0.362, DSR 0.989 ✓** — H-XAU-TREND-01 rechaza H0; bundles v1.3.0 |
| 7 | P2 portafolio + H-PORT-01 + P1 H-LATAM-02 | ✅ 2026-07-06 | `scripts/analysis/portfolio_layer.py`: mix Calmar 3.61 (IC incluye 0, N=52 — no probable aún); **TSMOM LATAM NEGATIVO en los 3 pares ⇒ NO se construye** (prima no existe en régimen mean-reverting); H-LATAM-01 carry gated en H-COP-CARRY-00 (medición del operador). +2 trials |
| 3/8 | S4 fan-out + R5 endpoints multi-tenant | ✅ 2026-07-06 | `app/api/routes/tenant.py` desplegado (6 rutas vivas): me/keys (anti-withdraw fail-closed, Vault-encrypted), me/limits (techos solo-hacia-abajo), me/kill, system/kill (admin, audit), `fan_out_signal()` → user_executions; cada orden pasa PreTradeGate |
| 8 | R7 rate-limit + hardening | ✅ 2026-07-06 | rate-limit por usuario en middleware (120/min, prefijos data-heavy, 429); CORS same-origin Next default; headers ya activos |
| UX | Landing honesta (S2 hero) + CTA→/metodologia | ✅ 2026-07-06 | `lib/translations.ts` EN+ES: fuera "RL/maximiza Alpha/escasez fake 3/50"; entra el copy del spec ("verificadas en producción... ganen o pierdan"; "la mayoría de semanas NO se opera"); Hero CTAs → /login + /metodologia |
| 5/7 | B3 on-chain (BGeometrics/Farside) · CLP seed · carry medido (H-COP-CARRY-00) · claves Wompi · gate legal SFC | 🔒 BLOQUEADO EXTERNO | acciones del operador/proveedores — no de código. TODO el código implementable está hecho |

## Context (por qué)
El usuario consolidó una auditoría (I metodología, II diseño cuant, III propuesta de estrategias,
IV verificación dashboard) y pidió **un plan para implementar todo, incluyendo SignalBridge**. La
exploración del código confirmó los hallazgos centrales y el estado real de SignalBridge. Esto NO es
una tarea de una sesión: es un **programa por olas**. El plan prioriza por (a) los P0 de la
auditoría, (b) el orden valor-de-información/costo del §5 del documento, y (c) lo que desbloquea lo
demás. Nada se ejecuta hasta aprobación; disciplina heredada: **un diagnóstico sobre OOS solo genera
hipótesis para el período siguiente, jamás cambios evaluados en el mismo**.

### Hallazgos confirmados en código (grounding)
- **Frontend P0 (I-4):** `components/production/ForecastingBacktestSection.tsx:1572-1615` — `dynamicMetrics`
  recomputa Sharpe/p/gates desde los trades; `displayStats/displayPValue/displayIsSignificant/dynamicGates`
  usan esa recomputación. El Vote-2 humano se emite sobre números que **no** son los del bundle. Recompute
  también en `utils/replayMetrics.ts`, `lib/services/financial-metrics/MetricsCalculator.ts`.
- **Gate DSR no se aplica (I-3):** `scripts/run_gold_pipeline.py:93-116` calcula e inyecta `deflated_sharpe`
  pero el `recommendation` sale del p-value bootstrap (`res["recommendation"]`), **no gateado por DSR>0.95**
  → `gold_trend_b2` queda PROMOTE con DSR 0.921. Mismo patrón en `scripts/pipeline/run_btc_pipeline.py:96-132`.
- **Constitución BTC-only (I-2):** `constitution-modeling.md`, `HYPOTHESIS-REGISTRY.md`, `PRE-REGISTRATION.md`
  viven solo en `.claude/specs/assets/btcusdt/design/`. COP no tiene registro de trials, ni constitución
  transversal, ni protocolo de retiro firmado.
- **SignalBridge:** credenciales **Vault-encriptadas** vía `POST /exchanges/credentials` → `ExchangeService`
  (adapters/factory, `is_testnet` por credencial). Pero `services/execution.py::execute_order` (155-267) va
  **decrypt → adapter → place_order sin RiskCheckChain ni kill-switch ni gate EXECUTION_MODE global**, y sin
  `amount_to_precision` (bug `step_size = 10**-precision.amount` = 1.0 para BTC/USDT en MEXC). Existe
  `services/risk_bridge.py` pero no está cableado al path de orden.
- **Ya hecho esta sesión (no re-hacer):** ingesta derivados BTC (`crypto_derivatives_daily`, 2492 días) + DAG
  `l0c_ingest_derivatives`; fix replay (`/api/data`) + promote (`:rw`); DSR/PSR/trial-aware en
  `services/common/metrics.py`; fix calendario Gold + validador OHLCV. POC MEXC read-only + copy-trade dry-run.
- **Extractores 052 faltantes:** on-chain (`crypto_onchain_daily`) y flows/ETF (`crypto_flows_daily`) sin
  extractor; ningún feature de `src/btc_strategy/` consume aún `crypto_derivatives_daily`.
- **RBAC/monetización (nuevo, grounding confirmado):** el dashboard **ya tiene NextAuth**
  (`app/api/auth/[...nextauth]/route.ts`, `lib/auth/api-auth.ts`, `/login`) y `middleware.ts` con `token.role`,
  **pero** el único chequeo de rol es `admin` para `/admin`|`/api/admin`|`/api/users` (`middleware.ts:34-38,108-125`)
  — **no existen** roles `developer`/`subscriber`/`free`, ni entitlements, ni matriz permiso×ruta, ni
  `lib/contracts/rbac*`. **`/data` y `/forecasting` están en `PUBLIC_ROUTES`** (`middleware.ts:29-30`) y el
  matcher excluye `.json`/`.csv` (`:186`) → los **716+ artefactos monetizables** de `public/data/analysis/**`
  + `public/forecasting/**` se sirven **públicos sin auth**. `/execution` usa auth aparte por localStorage
  (`:21`). SignalBridge es **single-tenant** (llaves del sistema, kill switch único).

---

## OLA 0 — Gobernanza metodológica (P0, barata, desbloquea confianza)
> Convierte las reglas de honestidad (hoy BTC-only) en transversales ANTES de correr trials nuevos.
- **G1. Constitución transversal:** promover reglas anti-selección de `constitution-modeling.md` a
  `.claude/rules/quant-constitution.md` (COP/XAU/BTC): no grid-search sobre test; cada versión/grid/gate
  mirado = 1 trial; diagnóstico OOS → hipótesis del *siguiente* período.
- **G2. Registro de trials COP (retroactivo)** + **recomputar DSR honesto de v11** (conteo v1.0→v11, grids).
- **G3. Protocolo de retiro COP firmado ex-ante** + **congelar v11** (2026 forward = único juez limpio).
- **G4. Reconciliar EXPERIMENT_QUEUE** (congelada feb) + **prefijar specs por activo** (`btc-`/`xau-SPEC-XX`, I-6).
- **G5. Test de consistencia PRE-REGISTRATION↔YAML** (SPEC-11) + marcar YAML de diseño divergentes `superseded` (I-5).

## OLA 1 — Reconciliación frontend↔bundle (P0, I-4)
- **F1.** En `ForecastingBacktestSection.tsx`: separar **replay-preview** (animado desde trades) de **decision
  numbers** (del bundle `summary_*.json`). KPIs junto a Aprobar/Rechazar y `gates` se leen del bundle
  (`approval.gates`), NUNCA de `dynamicMetrics` (que solo rotula la animación, etiquetado "preview de replay").
- **F2.** Verificar `MetricsCalculator.ts`/`replayMetrics.ts` no alimenten cards de decisión.
- **F3.** Verificación navegador (A1–A7): replay 8 estrategias, promoción (Aprobar escribe `approval_state.json`
  sin EROFS), forecasting 3 activos, sin errores de consola.

## OLA 2 — Honest-gate del backtest (P0/P1, I-3 + II-4)
- **H1. Verdict DSR-aware:** en `run_gold_pipeline.py` y `run_btc_pipeline.py` el `recommendation` degrada a
  **REVIEW si DSR<0.95** (o subir bar vía ADR). Resuelve la contradicción `gold_trend_b2`.
- **H2. Baseline exposición emparejada B1′:** helper en `services/common/metrics.py` + `src/{gold_rl,btc_strategy}/backtest.py`
  ("exposición constante = exposición promedio realizada"). **Sin B1′ no hay PROMOTE.** (BTC: B1′ ≈ constante-0.44.)
- **H3. Stress de costos ×1/×2/×3** como gate estándar; REJECT si muere al doble.
- **H4. Descomposición de PnL** (carry/swap + spot + salidas) en cada reporte.

## OLA 3 — SignalBridge apto para copy-trading spot (POC → producción segura)
> Orden: arreglar sizing → cablear riesgo → gate de modo → loop de copia.
- **S1. Fix precision/sizing:** usar `exchange.amount_to_precision(symbol, qty)` (en `adapters/base.py`/`mexc.py`
  y `execution.py::create_execution`) en vez de `step_size=10**-precision.amount`. + test (MEXC BTC/USDT prec==0).
- **S2. Cablear RiskCheckChain + kill-switch** en `execute_order` **antes** de `place_*_order` (fail-safe). Hoy no.
- **S3. Gate de modo global `EXECUTION_MODE`** (`dry`/`paper`/`testnet`/`live`) en `core/config.py`; `dry`/`paper`
  construyen+validan pero no envían (como el POC dry-run). Default seguro `paper`.
- **S4. Concepto "copy":** `signal_bridge_orchestrator` + `redis_streams_bridge` mapean un `UniversalSignalRecord`
  (spot-only [0,1]) → `Execution` por notional. Reusar `src/contracts/signal_adapters.py`.
- **S5. POC live mínimo — DIFERIDO (decisión del usuario: dry-run/paper esta etapa).** Registrar llaves MEXC
  vía `POST /exchanges/credentials` (Vault) queda listo, pero **no se envían órdenes reales** hasta cablear
  S1-S3 + fondeo + confirmación explícita. **Rotar llaves** (expuestas en chat).
- **S6.** Reconciliar `execution-bridge.md`/`risk-management.md` con lo realmente cableado (hoy aspiracional).

## OLA 4 — Auditoría estadística COP (II-2, III-1)
- **C1. Suite COP-NULL** (`scripts/analysis/cop_null_suite.py`): NULL-A (siempre-short 1× + TP/HS v11), NULL-B
  (exposición constante=media v11), **DECOMP** (PnL v11 en carry+spot+salidas), **STRESS-2122** (walk-forward 2021-22).
- **C2. H-COP-V11-01:** bootstrap pareado ΔCalmar(v11, NULL-A). Si IC no excluye 0 → NULL-A ES la estrategia.
- **C3. H-COP-CARRY-00 (0 compute, primero):** medir swap real broker vs IBR−FFR 60d. Traspaso <50% → tesis carry muere.
- **C4. COP-CORE (carry condicionado)** + **COP-TSMOM (4/8/13w)** reusando mecánica TP/HS congelada. Verificar/ingerir
  faltantes vs `config/macro_variables_ssot.yaml`: Brent, COL5Y/10Y, COLCAP, `swap_broker_real`, COT.

## OLA 5 — BTC: consumir derivados + extractores restantes (III-3)
- **B1. Features funding** en `src/btc_strategy/indicators.py`: `funding_rate`/`z_funding` (merge causal `shift(1)`).
- **B2. `btc_trend_funding_s4`** (B2 × funding-gate) + gate honesto (DSR>0.95 ∧ OOS-2025+ ∧ bate B2 Sharpe+Calmar).
- **B3. Extractores 052 restantes:** on-chain MVRV-Z/NUPL/Puell (BGeometrics/CoinMetrics) → `crypto_onchain_daily`;
  ETF flows (Farside D+1) + stablecoin (DefiLlama) → `crypto_flows_daily`. Nuevos `ingest_btc_{onchain,flows}.py`.
- **B4. ADR-0013:** recorte gate liquidez 7→3 ({DFII10, ETF, stablecoin}). **B5.** núcleo+vol-spike-breaker antes de LLM.

## OLA 6 — Gold: ensemble lookbacks + resolver DSR (III-2)
- **X1. XAU-TREND-ENS** (SMA 63/126/252, long-flat, ex-ante) reemplazo propuesto de B2. H-XAU-TREND-01/02.
- **X2. XAU-REAL-COND** (tilt tasas reales condicionado a `corr_gold_real_90`) + falsación H-XAU-REAL-02.
- **X3. XAU-POS** (freno COT asimétrico); ingerir COT (CFTC) + GLD holdings (SSGA). **X4.** DSR resuelto por H1.

## OLA 7 — Amplitud + portafolio (II-3/5, III-4) — retorno marginal más barato
- **P1. LATAM-XS-CARRY / TSMOM** sobre {COP,MXN,BRL,CLP} con la fábrica (MXN/BRL en `aux_pairs`; CLP seed nuevo). H-LATAM-01/02.
- **P2. Capa de portafolio** (`scripts/analysis/portfolio_layer.py`): equal-risk inv-vol 90d + monitor ρ + breaker DD (12%→×0.5, 18%→flat). H-PORT-01.

## OLA 8 — RBAC + Monetización (multi-tenant) — programa de producto (CTR-RBAC-001)
> Convierte el sistema (hoy operador único) en multi-usuario monetizable. **Principio: el rol dice quién
> eres (admin/developer/subscriber/free), el plan dice qué pagaste (entitlements); ambos se validan
> **server-side por request** — la UI solo refleja. Deny-by-default.** Detalle completo en **APÉNDICE B**.
> Absorbe la parte multi-tenant de SignalBridge (OLA 3 = system-account; OLA 8 = own-account por usuario).
- **R0. Auditoría de superficie** (solo lectura): tabla ruta→auth-actual→datos-que-expone; 100% de páginas,
  `/api/**` y artefactos en `public/**` con datos del sistema. → `docs/rbac/PHASE0-surface-audit.md`.
- **R1. Auth + modelo de datos:** extender NextAuth (ya existe) con `users(role ENUM, entitlements JSONB, status)`,
  `sessions`, `audit_log` (append-only); sembrar admin; `requireUser()/requirePermission()`. Gate: sin sesión toda
  página≠landing/login redirige, `/api/**` → 401.
- **R2. Contrato RBAC SSOT + middleware:** `lib/contracts/rbac.contract.ts` (PERMISSIONS, ROLE_PERMISSIONS,
  ROUTE_PERMISSIONS) según APÉNDICE B §2/§3; `middleware.ts` deny-by-default; **test de cobertura que enumera
  rutas reales y falla si alguna no está en la matriz** (al CI).
- **R3. Sacar artefactos monetizados de `public/`:** mover `public/{forecasting,data/analysis}/**` a `data/gated/`
  (o MinIO privado); servir por handlers con permiso + **delay del entitlement** (free ⇒ forecasting T−1 /
  análisis T+7) + `Cache-Control: private`. **Cruza con OLA 1** (el `/api/data` fs route ya es el patrón base).
- **R4. Navegación por rol:** `<Nav/>` + hub desde el contrato; ocultar a subscriber Backtest/Experimentos/
  "Promover"/badges internos/jerga; renombrar la vista cliente de Producción → **"Señales"**; corregir copy
  stale ("modelo RL"/"PPO" → Smart Simple). Playwright 4 sesiones.
- **R5. SignalBridge multi-tenant** (migración `06x`): `user_exchange_keys` (cifradas, rechazar llaves con
  permiso withdraw), `user_risk_limits`, `user_executions` (append-only); cuenta actual → `system`;
  `/api/signalbridge/me/**` (scope por `user_id`) vs `/system/**` (admin); fan-out de señal → usuarios elegibles;
  **paper-first** (`paper_weeks_required`) antes de live; kill switch por usuario + kill global admin que domina.
  **Requiere OLA 3 (S1-S3: sizing, risk-checks, EXECUTION_MODE) hecho primero.**
- **R6. Billing + entitlements:** proveedor Colombia (Wompi/PayU, abstraído en `lib/billing/`); webhook
  firma-verificada actualiza `entitlements`; job diario degrada vencidos → free; `/pricing`, `/account/billing`, paywalls.
- **R7. Endurecimiento + auditoría:** rate-limit por usuario, CORS cerrado, headers; `audit_log` en Vote2/promote/
  kills/plan/llaves/live-enable; página `/admin`; suite RBAC (rol×ruta) como bloqueo de merge; disclaimers §7 + consent.
> **Gate legal (APÉNDICE B §7):** el tier `auto` (auto-ejecución para terceros) puede tocar órbita SFC en
> Colombia → **gate legal obligatorio antes de dinero real de terceros**; hasta entonces `auto` = paper-only.

## Orden de ejecución (§5)
> **PRIMERA OLA APROBADA (esta etapa) = OLA 0 + OLA 1.** El resto queda planificado para olas
> siguientes, con aprobación entre cada una. **SignalBridge: solo dry-run/paper** — S3 default
> `EXECUTION_MODE=paper`, **S5 (orden real) diferido** hasta cablear risk-checks y con fondeo+confirmación.
1. **OLA 0 + OLA 1** (gobernanza + frontend P0) — baratas, desbloquean confianza. **← implementar ahora.**
2. **C3** (medir swap, 0 compute) + **OLA 2** (verdict DSR, B1′, cost-stress).
3. **OLA 4** (COP-NULL/DECOMP/STRESS + H-COP-V11-01).
4. **OLA 3** (SignalBridge S1-S4; S5 solo con fondeo+confirmación).
5. **OLA 5** B1-B2 (funding — único BTC backtesteable hoy) → B3. 6. **OLA 6** Gold. 7. **OLA 7** amplitud+portafolio.
8. **OLA 8 (RBAC + Monetización)** — programa de producto **paralelo/posterior**; R0-R4 pueden arrancar en
   cualquier momento (R3 cruza con OLA 1), pero **R5 (SignalBridge multi-tenant) requiere OLA 3 S1-S3 hecho** y
   el tier `auto` con dinero real de terceros requiere el **gate legal** (APÉNDICE B §7). 9. Verificación E2E.
> ~30 trials nuevos: NO correr todo — cada hipótesis no corrida es deflación que no se paga. RBAC = trabajo de
> ingeniería/producto (no consume trials).

## Critical files
- Frontend: `ForecastingBacktestSection.tsx` (1572-1615), `utils/replayMetrics.ts`, `lib/services/financial-metrics/MetricsCalculator.ts`.
- Honest-gate: `scripts/run_gold_pipeline.py` (93-116), `scripts/pipeline/run_btc_pipeline.py` (96-132), `services/common/metrics.py`, `src/{gold_rl,btc_strategy}/backtest.py`.
- SignalBridge: `services/signalbridge_api/app/services/execution.py` (155, 100), `adapters/{base,mexc}.py`, `services/risk_bridge.py`, `core/config.py`, `api/routes/exchanges.py`.
- BTC: `src/btc_strategy/{indicators,strategies}.py`, nuevos `scripts/data/ingest_btc_{onchain,flows}.py`.
- COP/gobernanza: nuevos `scripts/analysis/cop_null_suite.py`, `.claude/rules/quant-constitution.md`, COP `HYPOTHESIS-REGISTRY`; `config/execution/smart_simple_v1.yaml` (congelado).
- RBAC: `usdcop-trading-dashboard/middleware.ts` (34-38,108-125; `/data`,`/forecasting` en PUBLIC_ROUTES 29-30), `lib/auth/api-auth.ts`, `app/api/auth/[...nextauth]/route.ts`; nuevos `lib/contracts/rbac.contract.ts`, `.claude/rules/rbac.md`, `.claude/specs/platform/rbac-monetization.md`, `lib/billing/**`, migración `06x_signalbridge_multitenant.sql`; mover `public/{forecasting,data/analysis}/**` → `data/gated/`.

## Verificación (E2E)
- Frontend: KPIs/gates de `/dashboard` == bundle (no recompute); replay 8 estrategias; promoción escribe estado; forecasting 3 activos sin errores de consola.
- RBAC: sin sesión toda página≠landing/login redirige y `/api/**`→401; test de cobertura ruta→permiso verde en CI; `curl` sin sesión a URLs viejas de `public/{forecasting,analysis}` → 404; free ⇒ datos retrasados, subscriber ⇒ al día; Playwright 4 roles (admin/dev/subscriber/free) allow/deny por la matriz; subscriber nunca toca `/api/signalbridge/system/**` (403+audit); kill global admin domina.
- Honest-gate: `gold_trend_b2` → REVIEW (DSR<0.95) o ADR; ningún PROMOTE sin B1′ y sin sobrevivir costos ×2; USD/COP H5 sin cambios (regresión).
- SignalBridge: sizing correcto (test MEXC); `execute_order` bloquea si riesgo/kill-switch falla; `EXECUTION_MODE=paper` no envía; POC live solo con fondeo+confirmación (cap duro).
- BTC: `btc_trend_funding_s4` reporta DSR+OOS+B1′; funding con `shift(1)` sin leakage.
- Tests: extractores (schema/anti-leak), COP-NULL, B1′, verdict DSR; suite verde, cobertura 70%.

## Constraints
- Aditivo en `fix/audit-p0-remediation`; **nada se commitea** sin pedirlo. No tocar WIP ajeno.
- `.env`/`secrets/*` gitignored — jamás commitear. **Rotar llaves MEXC + Binance** (expuestas en chat).
- Programa multi-semana: se ejecuta **por olas con aprobación entre ellas**, no todo de una.

> **OLAS 4–7 = implementación de la PROPUESTA completa del apéndice de abajo.** Cada estrategia/
> hipótesis/variable listada ahí es el contenido concreto de esas olas (COP §1 → OLA 4/7; XAU §2 →
> OLA 6; BTC §3 → OLA 5; portafolio+amplitud §4 → OLA 7). Estado: **PROPUESTA — nada corrido ni
> registrado**; cada fila se mueve a `HYPOTHESIS-REGISTRY` solo al comprometerse a correrla (=1 trial DSR),
> y requiere Vote humano. Priors declarados ex-ante, jamás optimizados sobre el test.

---
---

# APÉNDICE — PROPUESTA completa: Hipótesis y Estrategias por Activo (USD/COP · XAU/USD · BTC/USDT)

> **Estado: PROPUESTA.** Nada corrido ni registrado. Las hipótesis se mueven a `HYPOTHESIS-REGISTRY.md`
> solo al comprometerse a correrlas (cada fila = ≥1 trial en el DSR). Convenciones heredadas de la
> constitución: **parámetros = priors económicos declarados, jamás optimizados sobre el test**;
> Calmar/Sortino primarios; features causales (`shift(1)`, lag T−1 en macro); corrección
> Benjamini-Hochberg por familia; baselines obligatorios; sensibilidades completas (cada celda = trial).
> Fecha: 2026-07-06 · Requiere Vote humano antes de entrar a la cola.

## 0. Principio rector
**Cada activo tiene UN motor económico dominante y las estrategias se construyen alrededor de él.**
Regla de admisión: toda señal debe completar *"este retorno existe porque alguien está pagando por ___"*.
Si no se puede completar, es curve-fitting y no entra.

| Activo | Motor económico dominante | Prima que se cosecha | Quién paga |
|---|---|---|---|
| USD/COP | **Carry** (diferencial de tasas) condicionado por riesgo | Prima de crash EM FX | Cobertura cambiaria / quien huye en risk-off |
| XAU/USD | **Tendencia** (flujo persistente) + tasas reales medidas | Prima de momentum / difusión lenta | Bancos centrales acumulando + rebalanceadores tardíos |
| BTC/USDT | **Beta gestionada por vol** + valoración de ciclo | Liquidez en capitulación / ceder upside en euforia | El apalancado liquidado y el FOMO tardío |

### 0.1 Ranking de convicción a priori
- **Alta:** vol-targeting (todas), TSMOM ensemble multi-lookback, carry COP condicionado, amplitud LatAm cross-sectional.
- **Media:** freno funding BTC, ciclo on-chain (N≈4 ciclos), COT extremo oro, tilt tasas reales condicional oro, ETF/stablecoin flows.
- **Media-baja:** lead-lag Brent→COP, lead-lag MXN/CLP→COP (el mecanismo existe pero se arbitra rápido).
- **Baja / descartar:** M2 global como señal, sentimiento LLM como señal primaria, RL táctico (por ahora), añadir features al forecasting COP (evidencia propia ya los mató).

### 0.2 Gates transversales nuevos (aplican a TODA promoción)
1. **B1′ (exposición emparejada):** comparar vs "exposición constante = exposición promedio realizada". Sin B1′ no hay PROMOTE.
2. **Stress de costos ×1/×2/×3:** si muere con costos al doble, REJECT.
3. **Descomposición de PnL obligatoria:** carry/swap + dirección spot + mecánica de salidas.

## 1. USD/COP
### 1.0 Fuente económica
Modelos no predicen (R²<0, DA 48–55%); el edge de `smart_simple_v11` vive en gate de régimen + mecánica
TP/HS + sesgo corto. Económicamente cosecha **(a) carry de estar largo COP** y **(b) prima momentum EM**,
con salidas que recortan la cola izquierda. Riesgo estructural: el carry EM devuelve todo en semanas cuando
el peso se deprecia violentamente (2020, 2021-22). La propuesta hace ese trade explícito, condicionado y medido.

### 1.1 Variables macro (lag T−1 mínimo)
| Variable | Fuente | Transformación causal | Mecanismo |
|---|---|---|---|
| `carry_diff` = IBR (o TPM) − Fed Funds | BanRep + FRED (ya) | nivel pp + z rodante 1a | El motor: pago por mantener COP (señal medida, no supuesto) |
| `swap_broker_real` (long/short) | Broker MT5 (NUEVO, diario) | pp anualizados | **Prueba de fuego**: si el broker no traspasa el carry, la tesis muere en ejecución |
| Brent / WTI | ya | ret 5d/20d, z 63d | Petro-divisa: canal fiscal + términos de intercambio |
| EMBI COL | ya | nivel z 63d + Δ20d z | Prima riesgo país; su *velocidad* = detector risk-off local |
| COL5Y/COL10Y − UST10Y | ya | spread + Δ20d | Riesgo fiscal/soberano local (más limpio que EMBI en episodios locales) |
| VIX | ya | nivel z 252d | Risk-off global: el carry EM muere cuando VIX salta |
| DXY | ya | ret 20d | Ciclo del dólar |
| USDMXN, USDCLP | ya (aux_pairs) | ret t−1 | Lead-lag: descubrimiento de precios en pares líquidos primero |
| COLCAP | ya | ret 20d | Apetito riesgo local / flujos portafolio |
| Calendario BanRep(8/año)/FOMC/CPI + **ciclo electoral 2026 y senda fiscal** | events-taxonomy | flags binarios | Event gate por calendario, no por opinión |

### 1.2 Variables técnicas
**No agregar ninguna** (EXP-REGIME-001 0/7 y 19→36 features ya mostraron que degrada). Reusar `rvol` semanal
(vol-target), ATR semanal (TP/HS), retornos 1/5/21d (TSMOM).

### 1.3 Estrategias
**COP-CORE — carry condicionado por riesgo (principal):**
```
carry_ok  = carry_diff > 2.0 pp nominal            # prior: cubre costos+spread con margen
risk_off  = (z_VIX>1.5) or (z(ΔEMBI_20d)>1.5) or (rvol_20 > p90 propio)
intent    = -1 (short USDCOP) si carry_ok y no risk_off ; 0 si risk_off o no carry_ok
size      = vol_target(tv=0.10, floor=0.04, cap=1.5)
salidas   = mecánica TP/HS de smart_simple_v11 (congelada)
```
Sensibilidades pre-registradas (cada celda=trial): carry {1.5,2.0,2.5}pp; risk-off {1.0,1.5,2.0}σ. No se elige la mejor.

**COP-TSMOM — tendencia semanal explícita:**
```
votes  = sign(ret_4w), sign(ret_8w), sign(ret_13w)
intent = mean(votes)   # LONG USDCOP solo si los 3 acuerdan
size   = vol_target × |intent|
```

**COP-NULL — baselines duros (ANTES que todo):**
| ID | Qué | Propósito |
|---|---|---|
| NULL-A | Siempre-short 1× + mecánica TP/HS v11 | ¿La capa ML/regime agrega sobre "estar corto"? |
| NULL-B | Exposición constante = exposición promedio de v11 | ¿v11 hace timing o solo tiene menos beta? |
| DECOMP | PnL v11 2023-2026 en carry + spot + mecánica | ¿De qué vive el +25%? |
| STRESS-2122 | v11/COP-CORE walk-forward por 2021-22 (≥52 sem) | Daño del sesgo corto en depreciación 30-40% |

### 1.4 Hipótesis
| ID | H0 | Test | Criterio | Convicción |
|---|---|---|---|---|
| H-COP-CARRY-00 | El swap real del broker (short) no traspasa ≥50% del diferencial teórico | Medición 60d swaps broker vs IBR−FFR | <50% ⇒ la tesis carry muere en ejecución | — (medición) |
| H-COP-CARRY-01 | El PnL de short-USDCOP no se explica por carry | DECOMP + bootstrap fracción carry | IC95% fracción_carry>0.25 ⇒ carry real | Alta |
| H-COP-CARRY-02 | Condicionar por risk-off no mejora Calmar vs NULL-A | Bootstrap pareado ΔCalmar OOS | IC95% ΔCalmar>0 tras BH | Alta |
| H-COP-TREND-01 | El TSMOM 4/8/13w no supera al B1′ en Calmar | Bootstrap pareado ΔCalmar | IC95%>0 | Media-alta |
| H-COP-XLEAD-01 | Retornos t−1 de MXN/CLP/Brent no mejoran la DA del intent | Regresión parcial + uplift DA walk-forward | Sobrevive control Y ΔDA>1pp ⇒ tilt ±1/3 | Media-baja |
| H-COP-V11-01 | v11 no supera a NULL-A en Calmar (la capa ML es decoración) | Bootstrap pareado ΔCalmar (2025 + 2026 fwd) | IC95%>0; si no ⇒ NULL-A ES la estrategia | — (auditoría) |

### 1.5 Modos de fallo
Crash de carry (2020-03, 2021-22): short USDCOP pierde 15-30% en semanas ⇒ risk-off gate + STRESS-2122
obligatorios. Riesgo político/fiscal (EMBI se amplía sin VIX) ⇒ EMBI/COL5Y en el gate además de VIX.
Gaps de apertura post-fin-de-semana ⇒ TP/HS asume fill en la apertura.

### 1.6 Qué NO hacer
No features al forecasting. No asumir el carry (medirlo: H-COP-CARRY-00 primero). No re-tunear TP/HS mirando
2025/2026 (congelada; cualquier cambio se evalúa solo en período posterior).

## 2. XAU/USD
### 2.0 Fuente económica
**(a) Momentum:** desde 2022 el comprador marginal son bancos centrales (insensibles al precio) ⇒ tendencias
largas. **(b) Tasas reales como régimen, no constante:** oro↔DFII10 se rompió 2022-2025 por (a) ⇒ señal
condicional, no principio fijo.

### 2.1 Variables macro
| Variable | Fuente | Transformación | Mecanismo |
|---|---|---|---|
| DFII10 (tasa real 10a) | FRED (ya) | nivel, Δ20d, z 252d | Costo de oportunidad — solo válido en régimen macro-driven |
| `corr_gold_real_90` | ya (data.yaml) | rodante 90d | **Interruptor de régimen**: decide si el canal de tasas está vivo |
| DXY (DTWEXBGS) | FRED (ya) | ret 20d, z | Efecto denominador |
| T10YIE (breakevens) | FRED (ya) | Δ20d | Canal inflación (el más débil; solo contexto) |
| GLD holdings | SSGA (NUEVO, diario, gratis) | Δ5d, Δ20d z | Demanda inversión occidental — flujo rápido |
| COT managed-money net long | CFTC (NUEVO, semanal, gratis, lag T+3) | percentil rodante 3a | Crowding especulativo — freno asimétrico en extremos |
| Compras bancos centrales | WGC (mensual) | contexto, NO señal | Explica el régimen post-2022; no entra al modelo |
| SPX drawdown, VIX | ya | DD desde máx 63d; z | Demanda de refugio (solo se testea a nivel portafolio) |

### 2.2 Variables técnicas
Existentes (Wilder ATR/ADX, Hurst, SMAs, z_sma50, rvol_20) + **una** nueva: `dist_max_52w = close/max(close,252)−1`.

### 2.3 Estrategias
**XAU-TREND-ENS — ensemble de lookbacks (reemplazo propuesto de B2):**
```
votes  = sign(close−sma_63), sign(close−sma_126), sign(close−sma_252)
intent = mean(votes)                 # variante long-flat preferida: intent = max(intent,0)
size   = vol_target(0.10, floor 0.06, cap 1.5) × regime_mult
```
**XAU-REAL-COND — tilt tasas reales condicionado al régimen:**
```
si corr_gold_real_90 < -0.3:  tilt = -sign(Δ DFII10_20d) × 1/3   # se SUMA al voto de tendencia
sino:                          tilt = 0                            # régimen CB-driven: canal roto
```
**XAU-POS — freno COT asimétrico:**
```
si percentil_3a(COT_net_long) > 0.90 sostenido 2 sem:  risk_mult ×= 0.6   # solo frena
si percentil < 0.10:                                    nada              # nunca amplifica
```
**XAU-B1 (mantener) + B1′** como comparadores obligatorios.

### 2.4 Hipótesis
| ID | H0 | Test | Criterio | Convicción |
|---|---|---|---|---|
| H-XAU-TREND-01 | El ensemble 3/6/12m no supera en Calmar a B2 OOS | Bootstrap pareado ΔCalmar | IC95%>0 | Media-alta |
| H-XAU-TREND-02 | El ensemble no es más robusto a parámetros que B2 | IQR de Calmar entre celdas (se reporta, no se elige) | IQR_ens < IQR_B2 | Media-alta |
| H-XAU-REAL-01 | El tilt condicionado no agrega Calmar sobre XAU-TREND-ENS | Bootstrap pareado ΔCalmar aislado | IC95%>0 tras BH | Media |
| H-XAU-REAL-02 (falsación) | El tilt incondicional rinde igual que el condicionado | ΔCalmar(cond, incond) | Si IC incluye 0 ⇒ eliminar el condicionamiento | — |
| H-XAU-POS-01 | El COT extremo no predice deterioro del retorno forward ajustado | ΔCVaR forward tras p>0.90 vs resto | IC95% del delta no cruza 0 | Media |
| H-XAU-B2-RESOLVE | `gold_trend_b2` con DSR 0.921 no debería estar PROMOTE | Re-evaluar vs bar 0.95 o cambiar bar vía ADR | Consistencia regla↔acción | — |

### 2.5 Modos de fallo
Whipsaw en laterales largos (vol-target amortigua, no elimina). Reversión del comprador estructural
(`corr_gold_real_90` lo detecta). Gaps de domingo (validador duro ya lo cubre).
### 2.6 Qué NO hacer
No asumir la correlación con tasas reales fija. No shortear contra la deriva secular sin unanimidad de
lookbacks. No meter WGC mensual como señal.

## 3. BTC/USDT
> El diseño existente (constitución + SPEC-01..12 + PRE-REGISTRATION) es sólido. Esta sección **prioriza y
> recorta**, no rediseña. La complejidad se gana su lugar componente por componente, en aislamiento.

### 3.0 Fuente económica
**(a) Ceder upside en euforia / comprar en capitulación** (liquidez vs flujo apalancado, medible on-chain
MVRV-Z/NUPL). **(b) Recorte de cola por vol-targeting** (skew −0.96, kurtosis 15.7). **(c) Momentum**
(btc_trend_b2 ya es el único DSR>0.95 del sistema).

### 3.1 Orden de construcción (ESTO es la recomendación)
| Paso | Qué | Hipótesis | Nota |
|---|---|---|---|
| 1 | Extractores mig-052: funding (Binance), MVRV-Z/NUPL/Puell (BGeometrics/CoinMetrics), ETF flows (Farside D+1), stablecoin (DefiLlama) | todas | Bloqueante único; **funding ya HECHO esta sesión** |
| 2 | Núcleo = vol-target + vol-spike breaker (sin LLM) + B1′ | — | floor=target=0.30 ⇒ exposición típica ~0.44 ⇒ **B1′ = constante-0.44, no HODL 1.0** |
| 3 | z_ciclo en aislamiento | H-REG-01 + H-BTC-CYCLE-02 | Diferenciador de mayor convicción media |
| 4 | z_funding como freno | H-POS-01 | Asimétrico, solo frena |
| 5 | Combinación en riesgo | H-CMB-01/02 | |
| 6 | Liquidez recortada a 3 componentes | H-LIQ-01/02 + H-BTC-LIQ-03 | Ver ADR-0013 |
| 7 | Gate de eventos LLM | H-EVT-01/02 | Solo si el ΔCalmar de 2-6 no basta (el breaker cubre 80% con 5% de complejidad) |
| — | RL táctico (SPEC-09) | H-RL-01 | Congelado hasta que exista un S4 que batir |

### 3.2 ADR-0013 propuesto: recorte del gate de liquidez
7 pesos actuales (tasas 25/DXY 20/VIX 15/ETF 20/stablecoin 10/M2 5/dominancia 5) → **{DFII10, ETF, stablecoin}
40/35/25**. **M2 sale** (correlación espuria), dominancia y VIX salen (redundantes). Cuenta como trial; requiere ADR formal.

### 3.3 Hipótesis nuevas (además de las 15 ya registradas)
| ID | H0 | Test | Criterio | Convicción |
|---|---|---|---|---|
| H-BTC-CYCLE-02 | MVRV-Z → retorno forward NO es estable entre épocas de halving | Kruskal-Wallis por sub-era (reusa subera_report) | Si el efecto vive en una sola era ⇒ degradar peso | Media |
| H-BTC-LIQ-03 | La emisión neta de stablecoins 30d no lidera el retorno de BTC | Regresión con lag + bootstrap, control momentum | Sobrevive control con p<α tras BH | Media |
| H-BTC-B1P-01 | El núcleo vol-targeted no supera a B1′ (constante = exposición promedio) | Bootstrap pareado ΔCalmar | IC95%>0; si no ⇒ "vol-targeting" es solo menos beta | Alta (test) |
| H-BTC-ETF-01 | Los flujos ETF (D+1) no predicen retorno forward 5d ajustado | Regresión con embargo + bootstrap | p<α tras BH y sobrevive control momentum | Media |

### 3.4 Qué NO hacer
No tocar spot-only [0,1]. No calibrar sigmoides del ciclo mirando OOS (congeladas; si H-BTC-CYCLE-02 las mata,
ADR + trial). No usar recall histórico del LLM como estimación (ADR-0011). No construir el paso 7 antes de medir 2-6.

## 4. Portafolio + amplitud (retorno marginal más barato)
### 4.1 Portafolio de sleeves
```
pesos   = equal-risk (inverso a vol realizada 90d), rebalanceo mensual
monitor = |ρ| rodante 90d; si > 0.5 sostenido ⇒ combinar en riesgo (reusar ADR-0009)
breaker = DD agregado >12% ⇒ exposición ×0.5 · >18% ⇒ flat + revisión (NUNCA se relaja en drawdown, ADR-0012)
```
| ID | H0 | Test | Criterio |
|---|---|---|---|
| H-PORT-01 | El mix equal-risk no supera en Calmar al mejor sleeve individual | Bootstrap pareado ΔCalmar | IC95%>0 |

### 4.2 Amplitud LatAm (usa la fábrica)
Sharpe de cartera ~√N con activos poco correlacionados. MXN/BRL ya seedeados (`aux_pairs`); CLP = seed nuevo
(playbook A1). **LATAM-XS-CARRY**: cada viernes rankear {COP,MXN,BRL,CLP} por `carry_diff`; short USD contra
los 2 de mayor carry (>2pp, sin risk-off global); vol-target por par.
| ID | H0 | Test | Criterio | Convicción |
|---|---|---|---|---|
| H-LATAM-01 | No existe spread de carry cross-sectional (top-2 vs bottom-2) | Bootstrap del spread long-short vol-targeted | IC95%>0 tras BH | Alta |
| H-LATAM-02 | El TSMOM 4/8/13w sobre los 4 pares no supera al B1′ de cartera | Bootstrap pareado ΔCalmar | IC95%>0 | Alta |

## 5. Orden de ejecución de la propuesta (valor/costo)
1. H-COP-CARRY-00 (medir swap, 0 compute — puede invalidar la tesis). 2. COP-NULL completo + H-COP-V11-01.
3. Extractores BTC mig-052 (**funding ya hecho**; faltan on-chain/flows). 4. XAU-TREND-ENS vs B2 (resuelve DSR 0.921).
5. LATAM-XS-CARRY/TSMOM. 6. COP-CORE. 7. BTC pasos 2→6. 8. Portafolio (≥2 sleeves limpios). 9. XLEAD/LLM/RL solo si sobra Calmar.

## 6. Presupuesto de trials
~18 filas nuevas + ~12 sensibilidades ≈ **30 trials**. NO correr todo (cada hipótesis no corrida es deflación que
no se paga). El **track COP entra al registro** (retroactivo v1.0→v11, prospectivo para todo esto). Ningún claim
de edge sin recomputar el DSR con el conteo actualizado.

> *Aviso: diseño metodológico y de investigación, no asesoría financiera. Priors declarados sin mirar OOS; si al
> implementar alguno resulta insostenible, se cambia por ADR antes del primer backtest, nunca después.*

---
---

# APÉNDICE B — RBAC + Monetización: Roles, Navegación, Multi-tenant y Prompt de Implementación (CTR-RBAC-001)

> **Estado: PROPUESTA.** Destino: `.claude/specs/platform/rbac-monetization.md`; el §4 (reglas duras) extraído
> como regla thin en `.claude/rules/rbac.md` (auto-cargada). Complementa `platform/authentication.md` +
> `approval-gates.md`. **Principio:** el rol dice *quién eres* (admin/developer/subscriber), el plan dice *qué
> pagaste* (entitlements). Se validan **server-side en cada request** — la UI solo refleja. Deny-by-default.

## B.1 Roles y planes (SSOT)
### Roles (quién eres)
| Rol | Quién | Qué puede |
|---|---|---|
| `admin` | Pedro (operador dueño) | Todo: Vote 2, promover versiones, kill switch **global**, SignalBridge del sistema, gestión usuarios/planes, auditoría |
| `developer` | Quants/devs invitados | Backtest completo (replay, versiones, comparativas), **proponer** experimentos, leer Producción. **NUNCA**: voto final, promover, llaves de exchange, kill switch, datos de usuarios |
| `subscriber` | Cliente pagador | Señales, Forecasting, Análisis según su plan + **su propio** SignalBridge (§B.6) |
| `free` | Registrado sin pago | Contenido con retraso (§B.5). Cero señales en vivo, cero ejecución |
> Anónimo = solo landing pública. No hay rol "operator" separado (admin lo absorbe; 2º operador = por ADR).

### Entitlements (qué pagaste) — JSONB en `users.entitlements`
```json
{ "plan":"auto", "assets":["usdcop","xauusd"], "forecast_delay_hours":0, "analysis_delay_days":0,
  "signals_realtime":true,
  "execution":{ "enabled":true, "mode":"paper", "paper_weeks_required":4,
                "max_notional_usd":5000, "max_daily_loss_pct":3.0, "max_open_positions":2 },
  "expires_at":"2026-08-06T00:00:00Z" }
```
Reglas: `expires_at` vencido ⇒ degradación automática a `free` (job diario + check por request sensible). Los
defaults de riesgo son **techos del sistema**: el usuario puede bajarlos, jamás subirlos. El JWT lleva
`{role,plan}` como cache, pero toda operación sensible (ejecución, promover, datos en vivo) **re-valida contra
DB** para revocación inmediata.

## B.2 Matriz de navegación — página × rol (la lee `<Nav/>` + middleware del mismo contrato)
| Página / módulo | admin | developer | subscriber | free | Notas |
|---|---|---|---|---|---|
| Landing pública (+ pricing) | ✓ | ✓ | ✓ | ✓ | Métrica de marketing = **producción forward**, no el headline de backtest |
| Inicio (hub) | completo | dev | cliente (4 cards) | teaser | Cards desde la matriz. **Fix copy**: quitar "modelo RL"/"PPO v2.4" (prod = Smart Simple) |
| **Backtest** (replay, versiones, gates) | ✓ | ✓ **sin** "Promover" | ✗ | ✗ | Solo admin/developer prueban estrategias |
| **Experimentos** (aprobación 2 votos) | ✓ (Vote 2) | ver + proponer | ✗ | ✗ | Vote 2 al audit log |
| Producción → **"Señales"** para clientes | ✓ + controles | lectura | lectura, solo `assets` del plan | ✗ | Ocultar a subscriber: badges PENDING/PAPER, jerga (L4, gates, votos) |
| Forecasting | ✓ | ✓ | ✓ semana actual, `assets` del plan | semana **T−1**, 1 activo | Retraso lo aplica el **server** (`forecast_delay_hours`) |
| Análisis | ✓ | ✓ | ✓ al día | resumen **T+7** | Escenarios de trading siempre con disclaimer §B.7 |
| **SignalBridge** | **global** (cuenta sistema + kill global) | ✗ | **own-account** (sus llaves, su kill, paper→live) | ✗ | §B.6 |
| Admin (usuarios, planes, auditoría, flags) | ✓ | ✗ | ✗ | ✗ | Página nueva |
| Mi cuenta / billing | ✓ | ✓ | ✓ | ✓ | Perfil + suscripción |

## B.3 Matriz de APIs × permiso (deny-by-default; ruta sin entrada ⇒ **CI falla**)
| Ruta (patrón) | Permiso | Roles | Check extra server-side |
|---|---|---|---|
| `/api/registry`, `/api/backtest/**`, `/api/replay/**` | `research:read` | admin, developer | — |
| `/api/experiments/**` (POST propuesta) | `research:propose` | admin, developer | — |
| `/api/approval/vote`, `/api/registry/promote` | `approval:vote` | **solo admin** | Audit log obligatorio |
| `/api/signals/current`, `/api/production/summary` | `signals:read` | admin, dev, subscriber | `asset ∈ entitlements.assets` y `signals_realtime` |
| `/api/forecasting/**` | `forecast:read` | autenticados | Server aplica `forecast_delay_hours` (free ⇒ T−1) |
| `/api/analysis/**` | `analysis:read` | autenticados | Server aplica `analysis_delay_days` |
| `/api/signalbridge/me/**` (llaves/órdenes/kill propio) | `execution:self` | subscriber(auto), admin | Scope duro por `user_id`; límites del entitlement |
| `/api/signalbridge/system/**`, kill global | `execution:global` | **solo admin** | Audit log |
| `/api/admin/**` | `admin:all` | solo admin | — |
| `/api/market/realtime-price` | `market:read` | autenticados | Rate limit por usuario |
> Servicios backend (:8000/:8085/:8003): solo red Docker o reverse proxy; validan JWT del usuario reenviado por
> Next (audience/issuer) + token servicio-a-servicio. Ningún puerto interno expuesto directo a internet.

## B.4 REGLAS DURAS (extracto para `.claude/rules/rbac.md`)
1. **Deny-by-default:** toda página (≠ landing/pricing/login) y `/api/**` exigen sesión + permiso **server-side** (middleware + handler). Ocultar UI no es control de acceso.
2. **NADA monetizado en `public/`:** mover `public/{forecasting,data/analysis}/**` a `data/gated/` o MinIO privado, servir por handlers con auth + `Cache-Control: private`. `public/` solo marketing.
3. **Rol ≠ plan:** rol autoriza módulos; entitlement autoriza datos (activos/frescura/ejecución). Ambos por request; JWT es cache, DB es la verdad.
4. **Vote 2, "Promover" y kill global: solo `admin`,** siempre con audit log. `developer` propone, nunca promueve.
5. **SignalBridge multi-tenant estricto:** llaves por usuario, cifradas (Vault/KMS o `pgcrypto` con llave fuera de DB), scoped a su `user_id`. El sistema JAMÁS ejecuta por un usuario con las llaves del sistema, ni al revés.
6. **Paper-first obligatorio** para clientes: `paper_weeks_required` semanas con divergencia aceptable antes de live. El flag live lo activa el usuario tras aceptar el risk disclosure (§B.7); el sistema lo puede revocar.
7. **Kill switches en cascada:** el del usuario detiene lo suyo; el global del admin domina todo y no es visible/operable por clientes.
8. **Subscribers ven OUTPUTS, nunca INTERNALS** (ni configs, gates, hiperparámetros, `registry.json` crudo, ni Experimentos). Protege el IP.
9. **Billing por webhook:** verdad de pago = proveedor (Wompi/PayU/Mercado Pago; Stripe solo si se factura desde entidad válida). Webhook actualiza `entitlements`; vencido ⇒ degradación. Nunca confiar en claims del cliente.
10. **Rate limiting por usuario** en APIs de datos/mercado; CORS cerrado al dominio propio; sin API keys de datos en el bundle del frontend.
11. **Audit log append-only** (`audit_log`): aprobaciones, promociones, kills, cambios de plan, altas/bajas de llaves, ejecuciones — con `user_id`, acción, objeto, timestamp, IP. Nunca se edita/borra.
12. **Toda superficie con señales/escenarios lleva el disclaimer §B.7**; el alta de ejecución exige aceptación de ToS + risk disclosure (guardada en audit log).

## B.5 Monetización — tiers propuestos
| | `free` | `signals` (Pro) | `auto` (Premium) |
|---|---|---|---|
| Análisis semanal | resumen T+7 | completo, al día | completo, al día |
| Forecasting | T−1, 1 activo | semana actual | semana actual |
| Señales en vivo | ✗ | ✓ | ✓ |
| Activos | USD/COP | USD/COP (+add-on Oro/BTC) | multi-activo según add-ons |
| SignalBridge propio | ✗ | ✗ | ✓ (paper→live, límites §B.1) |
| Notificaciones (email/telegram) | ✗ | ✓ | ✓ |
> **add-ons por activo** mapean 1:1 al registro dinámico multi-activo (vender Gold/BTC = agregar el asset al
> entitlement, cero código nuevo). Número de marketing = **forward de producción (2026 YTD)**, backtest
> secundario y etiquetado. Precios = decisión comercial, fuera de este plan.

## B.6 SignalBridge multi-tenant (módulo crítico — migración `06x_signalbridge_multitenant.sql`)
```
user_exchange_keys(id, user_id FK, exchange, api_key_enc, api_secret_enc, created_at, last_verified_at, status)
user_risk_limits(user_id PK, max_notional_usd, max_daily_loss_pct, max_open_positions, mode ENUM('paper','live'), kill_switch bool)
user_executions(id, user_id, signal_id, exchange, side, qty, px, status, mode, created_at)  -- particionada, append-only
```
**Flujo cliente:** conectar exchange (verify: trade sí, **withdraw NO** → rechazar llaves con permiso de retiro)
→ paper N semanas → "pasar a live" (gated por `paper_weeks_required` + aceptación de riesgo) → ejecución de
señales del plan con sus límites → su kill switch. **Flujo señal:** una señal publicada se abanica a los
usuarios elegibles (plan+asset+modo) vía cola; cada ejecución corre bajo los límites del usuario y se corta
individualmente si toca `max_daily_loss_pct`. La cuenta actual (mexc/binance del operador) → `system`, solo admin.
> Depende de OLA 3 (S1 sizing, S2 risk-checks, S3 EXECUTION_MODE) ya cableados.

## B.7 Cumplimiento (no es asesoría legal — es la lista de pendientes)
Vender señales con escenarios entrada/stop/target, y más aún **auto-ejecución para terceros**, puede constituir
actividad regulada (en Colombia: órbita SFC — asesoría/administración de recursos). **Gate legal obligatorio
antes de activar `auto` con dinero real de terceros**; hasta entonces `auto` = paper-only. Mínimos de UI desde
el día 1: disclaimer persistente en Señales/Análisis/Forecasting ("contenido informativo y educativo; no es
asesoría financiera; rendimientos pasados no garantizan resultados; riesgo de pérdida total"), ToS + risk
disclosure con aceptación registrada, y política de datos (llaves cifradas, qué se guarda, cómo se borra).

## B.8 Fases de implementación (cada fase = 1 PR con gate verde; aditivo; no tocar lógica de estrategias)
- **Fase 0 — Auditoría de superficie (R0):** inventario páginas/`/api/**`/artefactos `public/**` + qué backend
  llama el cliente; verificar si el login admin valida server-side. → `docs/rbac/PHASE0-surface-audit.md`.
  **Gate:** cubre 100% de rutas y archivos `public/` con datos del sistema.
- **Fase 1 — Auth + datos (R1):** NextAuth v5 + adapter Postgres; `users/sessions/audit_log`; admin sembrado;
  `requireUser()/requirePermission()`. **Gate:** sin sesión toda página≠landing/login redirige, `/api/**`→401.
- **Fase 2 — Contrato RBAC + middleware (R2):** `lib/contracts/rbac.contract.ts` (PERMISSIONS/ROLE_PERMISSIONS/
  ROUTE_PERMISSIONS = §B.2/§B.3); `middleware.ts` deny-by-default; test de cobertura al CI. **Gate:** test verde, matriz sin diffs.
- **Fase 3 — Sacar artefactos de `public/` (R3):** mover JSON/PNG gated a `data/gated/`; handlers con permiso +
  delay del entitlement + `Cache-Control: private`; actualizar componentes. **Gate:** `curl` sin sesión a URLs
  viejas → 404; free ⇒ retrasado; subscriber ⇒ al día.
- **Fase 4 — Nav + UI por rol (R4):** `<Nav/>`/hub desde el contrato; ocultar a subscriber Backtest/Experimentos/
  "Promover"/badges/jerga; Producción→"Señales"; fix copy stale. **Gate:** Playwright 4 sesiones tabla-en-mano.
- **Fase 5 — SignalBridge multi-tenant (R5):** migración `06x`; `me/**` vs `system/**`; fan-out; paper-first;
  kills en cascada. **Gate:** 2 usuarios paper con límites distintos, solo ejecuta el elegible; kill global
  detiene ambos; subscriber → `system/**` = 403+audit.
- **Fase 6 — Billing (R6):** proveedor (Wompi/PayU, abstraído en `lib/billing/`); webhook firma-verificada →
  `entitlements`; job diario degrada vencidos; `/pricing`, `/account/billing`, paywalls. **Gate:** simular
  pago ⇒ entitlement al día; simular vencimiento ⇒ degradación en el siguiente request.
- **Fase 7 — Endurecimiento + auditoría (R7):** rate-limit, CORS, headers; audit_log conectado; `/admin`; suite
  RBAC (rol×ruta) como bloqueo de merge; disclaimers §B.7 + consent flow. **Gate:** suite RBAC verde en CI.
> Al terminar cada fase: actualizar `docs/rbac/IMPLEMENTATION_STATUS.md` con evidencia (comandos, screenshots).

> *Nota: este apéndice define control de acceso y producto; NO resuelve el gate legal §B.7 ni fija precios —
> ambos son decisiones del operador previas al cobro real.*
