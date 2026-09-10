"""Parseo de numeros de Investing.com consciente del LOCALE.

Contract: CTR-DQ-MACRO-SCALE-001 · Date: 2026-08-24

Vive en `src/` y no en el modulo de estrategias para que se pueda TESTEAR sin Airflow
instalado — el mismo motivo por el que `test_seed_ohlcv_integrity.py` no toca Postgres.
Lo importa `airflow/dags/services/macro_extraction_strategies.py`.
"""
from __future__ import annotations


def parse_investing_number(text: str, url: str = "") -> float:
    """Convierte el texto de una celda de Investing.com a float respetando el LOCALE.

    ROOT CAUSE (auditoria 2026-08-24, CTR-DQ-MACRO-SCALE-001)
    ---------------------------------------------------------
    Esta funcion existe porque el codigo anterior hacia, sin mirar la URL:

        value = float(cols[1].get_text(strip=True).replace(',', ''))

    Correcto en el sitio ingles (`1,234.56` -> la coma es separador de miles), y
    DESTRUCTIVO en el espanol, donde la coma es el separador DECIMAL:

        "17,4720" -> "174720"   (x10.000)
        "921,98"  -> "92198"    (x100)

    `config/l0_macro_sources.yaml` manda exactamente dos indicadores a
    `es.investing.com` — `fxrt_spot_usdmxn_mex_d_usdmxn` y
    `fxrt_spot_usdclp_chl_d_usdclp` ("Spanish URL for better availability") — y son
    exactamente los dos que aparecieron corruptos: 16 filas entre 2026-06-29 y
    2026-08-04, todas con el factor 10^(numero de decimales).

    Por que importaba: `usdmxn_change_1d` es la feature #15 de las 20 del
    `FEATURE_ORDER` canonico que consume el pipeline RL, y esas fechas caen dentro del
    hold-out de la tesis. Un salto de escala en una feature de CAMBIO diario mete un
    valor basura el dia que entra y otro el dia que sale.

    Reglas de desambiguacion (en este orden):
      1. Si aparecen AMBOS separadores, manda el ULTIMO: es el decimal.
         "1.234,56" -> 1234.56   ·   "1,234.56" -> 1234.56
      2. Si solo hay uno, decide el locale de la URL (`es.` -> coma decimal).
      3. Sin URL y con un solo separador, se asume convencion inglesa (el default
         historico), que es lo que usan los 17 indicadores del sitio `www.`.
    """
    raw = str(text).strip()
    if not raw or raw in {"-", "--", "N/A"}:
        raise ValueError(f"valor vacio: {raw!r}")
    raw = raw.replace(" ", "").replace(" ", "").replace("%", "")

    is_spanish = "es.investing.com" in (url or "")
    has_dot, has_comma = "." in raw, "," in raw

    if has_dot and has_comma:
        # El separador decimal es el que aparece mas a la derecha.
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif has_comma:
        raw = raw.replace(",", ".") if is_spanish else raw.replace(",", "")
    elif has_dot and is_spanish:
        # En espanol el punto es separador de MILES... salvo que sea un decimal ya
        # normalizado. Se trata como miles solo si el grupo final tiene 3 digitos.
        tail = raw.rsplit(".", 1)[-1]
        if len(tail) == 3 and raw.count(".") >= 1 and len(raw.split(".")[0]) <= 3:
            raw = raw.replace(".", "")
    return float(raw)
