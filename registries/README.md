# registries/ — Ledger global de trials FT-/AT- y familias transversales

> **BL-09 + BL-10 + BL-11** (FABRIC §9.4–§9.7, ADR-0022). Creado 2026-07-27 por backfill
> `legacy_estimate`. Validado por `scripts/validation/check_trial_ledger.py` y congelado por
> `tests/regression/test_trial_ledger.py`.

## Qué hay aquí

| Archivo | Qué es |
|---|---|
| `ledger.jsonl` | **Append-only, hasheado.** Una línea por trial (FT-#### predictivo / AT-#### económico) con familia, cluster, asset, cutoff, result y contadores N_family/N_cluster/N_global corrientes. |
| `families/trend_regime.yaml` | **Piloto BL-11**: familia ACTION transversal (SPX/Oro/BTC), bar pre-firmado, celdas reales con status/result de los registries. |
| `families/usdcop_direction.yaml` | Familia FORECAST por activo (§9.5): la dirección COP, **cerrada por escrito** (48 FT). |

## Reglas duras

1. **Append-only.** Cada línea lleva `prev_hash` (line_hash de la anterior; génesis = 64
   ceros) y `line_hash = sha256(JSON canónico del payload sin line_hash)`. Editar una línea
   histórica rompe la cadena y el CI. Corregir = **nueva línea**, jamás editar.
2. **La partición FT/AT no descuenta nada** (ADR-0022 §3): `n_trials_total` de cada activo =
   FT + AT. El DSR se deflacta con el total; los tres N (family/cluster/global) son
   divulgación obligatoria. **El gobierno gatea con DSR_family.**
3. **N_MAX = 989 es cota de GASTO únicamente** (§9.7): commitment device contra el
   p-hacking. **JAMÁS entra en el DSR** ni en ninguna fórmula estadística.
4. **Trial nuevo (env ≠ `legacy_backfill`) exige familia declarada** en `families/*.yaml`
   ANTES de mirar — el validador lo bloquea. Cruzar la muralla forecast→señal cobra su
   propio AT con `provenance` de los FT heredados (ADR-0022 §2 y §4).
5. **Cerrar una familia por escrito es un estado válido** (`closed: true` + `closure_note`),
   no un fracaso. Dividir familias para lavar multiplicidad queda visible vía cluster +
   N_global (decisión rechazada, FABRIC §31).
6. Los HYPOTHESIS-REGISTRY por activo siguen siendo el SSOT narrativo del conteo
   (`n_trials_total` en su front-matter); este ledger es su forma **maquinal**. El CI exige
   igualdad exacta por activo. Nuevos trials se registran en AMBOS en el mismo commit.

## Backfill `legacy_estimate` 2026-07-27 (BL-10) — derivación

Los conteos vienen de los front-matter **reales** leídos ese día:
`usdcop=109, xauusd=77, btcusdt=34, spx500=17` (total 237 = 53 FT + 184 AT).
Regla de clasificación aplicada: **direccional/predictivo = FT, económico = AT; ambiguo →
AT (conservador)**. La predicción de volatilidad (H-VOLF-01, QLIKE) es FT sin ambigüedad:
es la pregunta predictiva del ADR-0022 (calibración), no una hipótesis económica.

| Asset | Familia (cluster) | Linaje | n | IDs | Fuente |
|---|---|---|---|---|---|
| usdcop | `usdcop_direction` (ml_meta) | FT | 48 | FT-0001..0048 | EXP-DIR-001: 27 previos + 14 forward-flow + 7 intraday-LatAm ("48 direccionales / 109 globales") |
| usdcop | `usdcop_vol` (vol) | FT | 1 | FT-0049 | H-VOLF-01 NO_RECHAZA |
| usdcop | `smart_simple` (ml_meta) | AT | 60 | AT-0001..0060 | Residuo exacto 109−49: grid 42 hs/tp, sizing, NULL suite, portfolio, XLEAD, H-META-01… |
| xauusd | `trend_regime` (trend) | AT | 74+1 | AT-0061..0135 | TRIALS_PROGRAM=74 heredado (sin procedencia, conservador) + H-SIMP-GOLD-01 |
| xauusd | `xauusd_vol` (vol) | FT | 1 | FT-0050 | H-VOLF-01 **RECHAZA_H0** (único de 4 activos) |
| xauusd | `vol_sizing` (vol) | AT | 1 | AT-0136 | H-VOLE-01 NO_RECHAZA |
| btcusdt | `exposure_engine` (ml_meta) | AT | 28 | AT-0137..0164 | Registro SPEC-01..11 + sensibilidades; residuo exacto 34−6 |
| btcusdt | `trend_regime` (trend) | AT | 3 | AT-0165..0167 | H-POS-01 (fail gate) + H-ENG-01/02 (declaradas, ya cargadas) |
| btcusdt | `vol_sizing` (vol) | AT | 1 | AT-0168 | H-VOL-01 NO_RECHAZA |
| btcusdt | `btcusdt_vol` (vol) | FT | 1 | FT-0051 | H-VOLF-01 NO_RECHAZA (HAR peor con significancia) |
| btcusdt | `btcusdt_funding_direction` (flow) | FT | 1 | FT-0052 | H-DIR-FUND-01 NO_RECHAZA |
| spx500 | `trend_regime` (trend) | AT | 14+2 | AT-0169..0184 | Grid 12 + B1/B2/S3 (contadas 14) + H-SIMP-SPX-01/02 |
| spx500 | `spx500_vol` (vol) | FT | 1 | FT-0053 | H-VOLF-01 NO_RECHAZA |

Notas de honestidad del backfill:

- **El zoo COP 9×7≈63 celdas queda subsumido** en los 27 "previos" de EXP-DIR-001: el total
  del activo (109) no admite 63 sin romper la suma, y BL-10 prohíbe cambiar el conteo. Es
  la "estimación honesta por escrito" que FABRIC §10.2 exige — un N estimado documentado
  vale infinitamente más que un N=0 falso.
- Las líneas `label: legacy_estimate` son bloques estimados (no mapean 1:1 a una hipótesis);
  las `label: documented` mapean a una hipótesis registrada con nombre en su registry.
- `code_hash`/`data_hash` son `null` en el backfill: no son reconstruibles. Los trials
  nuevos DEBEN llenarlos.
- Los conteos citados en prompts/planes viejos (COP 88, BTC 33) estaban desactualizados; el
  front-matter real del día del backfill manda (109/77/34/17).

## Cómo se añade un trial nuevo

1. Pre-registra la hipótesis en el HYPOTHESIS-REGISTRY del activo (como siempre) y sube su
   `n_trials_total`.
2. Declara (o extiende) la familia en `families/{family}.yaml` — la celda ANTES de mirar.
3. Añade la línea al final de `ledger.jsonl`: mismos campos, `prev_hash` = `line_hash` de la
   última línea, contadores incrementados, `env` real (`screening`/`forward`/…),
   `code_hash`/`data_hash` obligatorios.
4. `python scripts/validation/check_trial_ledger.py` debe salir verde antes de commitear.
