# SPEC-01 — Ingesta de Datos

## Propósito
Descargar y persistir crudo, de forma **incremental e idempotente**, las series necesarias: oro XAUUSD (H1, derivado de M1/tick), DXY, tasas reales/TIPS y calendario macro. Salida: `data/raw/` versionado con DVC.

## Fuentes y herramientas concretas

### 1. Oro — XAUUSD (fuente primaria: Dukascopy)
Dukascopy ofrece tick histórico gratuito desde ~2003, en **GMT/BST (con DST)** — a normalizar en SPEC-02.

**Opción A (recomendada) — CLI `dukascopy-node`:**
```bash
npx dukascopy-node -i xauusd -from 2004-01-01 -to 2025-12-31 \
  -t m1 -f csv --date-format "YYYY-MM-DD HH:mm:ss" -bs 1 -bp 10 -r 5 \
  -dir ./data/raw/gold/_dukascopy_m1
```
`-t m1` descarga velas de 1 minuto (más liviano que tick y suficiente para agregar a H1). `-r 5` reintentos, `-bp` pausa entre lotes.

**Opción B — Python `dukascopy-python`:** (si prefieres todo en Python)
```python
import dukascopy_python as dk
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD
df = dk.fetch(INSTRUMENT_FX_METALS_XAU_USD, dk.INTERVAL_MIN_1,
              dk.OFFER_SIDE_BID, start, end)  # devuelve DataFrame OHLCV
```

**Fuentes de respaldo / cross-check** (para validar Dukascopy, no como primaria):
- HistData.com — M1 gratis por mes (ASCII), útil para verificar velas.
- Export directo del broker vía `MetaTrader5.copy_rates_range(...)` — para reconciliar contra los datos con los que operarás en live.

> Descarga **M1 y agrega a H1/Daily en SPEC-02**, no bajes H1 pre-agregado: necesitas control de la convención de sesión y de las velas de domingo.

### 2. Dólar — DXY
El DXY de ICE es propietario. Proxies válidos:
- **FRED `DTWEXBGS`** (Nominal Broad U.S. Dollar Index, diario) — recomendado por consistencia con el resto de FRED.
- Alternativa: `yfinance` ticker `DX-Y.NYB` (índice) o `DX=F` (futuro), si quieres el DXY clásico.
Decide UNA fuente y documéntala (ADR si cambias). Es contexto Daily.

### 3. Tasas reales / TIPS (FRED, vía `fredapi`)
Series concretas:
| Series ID | Descripción | Uso |
|---|---|---|
| `DFII10` | 10-Year Treasury Inflation-Indexed (real yield) | **tasa real principal** |
| `DFII5` | 5-Year real yield | opcional (curva) |
| `DGS10` | 10-Year nominal Treasury | nivel nominal |
| `T10YIE` | 10-Year Breakeven Inflation | expectativa de inflación |
| `DTWEXBGS` | Broad Dollar Index | proxy DXY |

```python
from fredapi import Fred
fred = Fred(api_key=os.environ["FRED_API_KEY"])
real10 = fred.get_series("DFII10", observation_start="2004-01-01")  # Series diaria
```

### 4. Calendario macro (eventos de alto impacto: CPI, NFP, FOMC)
Necesitas: `timestamp` (UTC), `event`, `currency` (filtrar USD), `impact` (high). Opciones:
- **Trading Economics API** (`tradingeconomics` pip) — robusta, requiere key.
- **Finnhub** `/calendar/economic` — free tier disponible.
- **investpy** / scrape de ForexFactory (XML `?week=`) — gratis pero frágil; si scrapeas, cachea agresivo y valida el schema.

Guarda **datetime exacto del release en UTC** (no solo la fecha): el blackout de SPEC-06 es de minutos.

## Interfaz (contrato de código)

```python
# src/gold_rl/data/ingest/base.py
class Downloader(Protocol):
    def download(self, start: date, end: date) -> pd.DataFrame: ...
    def incremental(self, existing_max_ts: datetime | None) -> pd.DataFrame: ...
    @property
    def raw_path(self) -> Path: ...

# implementaciones: GoldDukascopyDownloader, FredDownloader, CalendarDownloader
```

**Incrementalidad:** cada downloader lee el `max(ts)` ya persistido y descarga solo lo nuevo. Reescribir una partición-año completa es aceptable (idempotente); nunca hacer append ciego que duplique.

## Almacenamiento
- Formato: Parquet, partición por `year`.
- `data/raw/{source}/{symbol}/year=YYYY/part.parquet`.
- Tras cada corrida: `dvc add data/raw/... && dvc push`.
- Metadata de corrida (rango, filas, hash, fuente, versión de lib) en un `_manifest.json` por dataset.

## Criterios de aceptación
- [ ] `GoldDukascopyDownloader` baja M1 de XAUUSD 2004→hoy y persiste Parquet particionado; segunda corrida NO duplica filas (test de idempotencia).
- [ ] `FredDownloader` trae `DFII10`, `DGS10`, `T10YIE`, `DTWEXBGS` con índice diario tz-aware.
- [ ] `CalendarDownloader` devuelve eventos USD high-impact con timestamp UTC a resolución de minuto; test contra ≥3 fechas conocidas de FOMC/NFP.
- [ ] Descarga incremental testeada: dado un `max_ts`, solo trae posterior.
- [ ] Todo crudo bajo control DVC con `_manifest.json` por dataset.
- [ ] Manejo de fallos: reintentos con backoff; una fuente caída no tumba las demás (aislamiento por task, ver SPEC-10).

## Dependencias
SPEC-00 (scaffold, DVC, config).
