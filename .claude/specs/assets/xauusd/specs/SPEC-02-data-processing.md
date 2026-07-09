# SPEC-02 — Procesamiento de Datos

## Propósito
Convertir el crudo en datasets limpios y alineados listos para features: normalizar zona horaria/DST, resolver velas de domingo, auditar calidad (y **fallar** si no cumple umbrales), agregar M1→H1→Daily con la convención de sesión correcta, y alinear H1↔Daily con corrección point-in-time.

## Pasos (pipeline determinista)

### 1. Normalización TZ/DST — el detalle que mata backtests
- Dukascopy entrega **GMT con BST (DST)**. Convertir explícitamente a **UTC tz-aware**. No asumir naive.
- Definir la frontera de la barra **Daily** en **NY-close (17:00 America/New_York)**, la convención estándar de FX/oro. Esto implica: cada "día" de trading va de 17:00 NY a 17:00 NY.
- Test obligatorio: verificar que en los dos cambios de DST anuales (marzo/noviembre) las barras no se corren ni duplican.

```python
df["ts"] = pd.to_datetime(df["ts"], utc=True)            # todo a UTC
# frontera Daily por sesión NY:
ny = df["ts"].dt.tz_convert("America/New_York")
df["session_date"] = (ny - pd.Timedelta(hours=17)).dt.date  # día de sesión
```

### 2. Velas de domingo
El mercado FX abre domingo ~17:00 ET. Decisión (documentar y ser consistente entre backtest y live): **fusionar la sesión de domingo con el lunes** (recomendado) o eliminarla. Nunca dejarla como "día" propio de pocas horas.

### 3. Auditoría de calidad (great-expectations o checks propios)
Produce `reports/data_quality_{date}.html` y **falla el pipeline** si:
- Gaps > umbral: más de `N` barras H1 faltantes consecutivas en horario de mercado (excluyendo fines de semana/feriados). Umbral sugerido: alerta >3, fail >12.
- Duplicados de timestamp: 0 tolerados.
- Velas con `volume == 0` o `high < low` o `close` fuera de `[low, high]`: 0 tolerados.
- Saltos de precio > `k`·ATR entre barras sin evento asociado: marcar como outlier para revisión.
- **Flag pre-2010:** columna `low_quality_period = ts < 2010-01-01` (spreads sintéticos, data rala). No se borra; se marca para ponderar en entrenamiento (SPEC-08) o usar solo para el clasificador de régimen.

### 4. Resampling M1 → H1 → Daily
- M1 → H1: `open`=first, `high`=max, `low`=min, `close`=last, `volume`=sum, sobre ventanas horarias alineadas a UTC.
- H1 → Daily: agregación sobre la `session_date` (NY-close).
- Barras H1 sin ticks (mercado cerrado): NO forward-fill de OHLC; se omiten (el mercado no existía). Distinguir "cerrado" de "faltante por error".

### 5. Alineación H1 ↔ Daily (point-in-time — CRÍTICO)
El feature/label Daily del día `D` (que usa el `close` de D) **solo se conoce tras el NY-close de D**, por lo que se aplica a las barras H1 de `D+1` en adelante:

```python
# regla: shift de 1 sesión al hacer el merge Daily→H1
daily_features = daily_features.assign(available_from=lambda d: d["session_date"] + BDay(1))
h1 = h1.merge_asof(daily_features, left_on="session_date", right_on="available_from", direction="backward")
```
Test anti-look-ahead: para una fila H1 de la sesión `D`, el feature Daily adjunto debe provenir de `D-1` o anterior, nunca de `D`.

## Salida
- `data/processed/gold_h1/` y `data/processed/gold_daily/` (Parquet, part. por año, UTC).
- Reporte de calidad versionado.
- DVC add/push.

## Criterios de aceptación
- [ ] Test DST: barras correctas en los 4 cambios de horario de 2 años de muestra.
- [ ] Test point-in-time: 0 filas con feature Daily del mismo día de sesión.
- [ ] Auditoría falla el DAG ante gaps/duplicados/velas inválidas sobre umbral (test con dataset corrupto sintético).
- [ ] Resample M1→H1→Daily reproduce OHLCV correcto contra un tramo verificado a mano.
- [ ] Columna `low_quality_period` presente y correcta.
- [ ] Domingo tratado consistentemente (test).

## Dependencias
SPEC-01 (crudo).
