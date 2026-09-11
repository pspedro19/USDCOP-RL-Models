from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.usdcop_forward_macro import (
    ForwardMacroScraper,
    PIT_COLUMNS,
    _causal_rolling_z,
    attach_forward_macro_features,
    parse_daily_forward_text,
    parse_eme_workbook,
    parse_forward_history_workbook,
    parse_monthly_derivatives_text,
    parse_sfc_formato_415,
)


DAILY_SAMPLE = """
Segun el reporte durante el dia se pacto un total de US$4733 millones con una
devaluacion implicita promedio ponderada por monto de 4.95%.
Cuadro No. 1 Contratos Forwards Pactados
3-14 918,8 557,9 337,9 771,2 130,5 58,2 89,5
15-35 652,7 1355,7 1084,7 344,9 32,0 68,8 239,0
36-60 45,9 49,4 0,0 0,0 34,4 30,9 15,0
61-90 491,2 202,4 69,7 319,0 52,7 92,2 80,0
91-180 325,7 358,3 34,2 0,0 24,1 25,7 300,0
>180 440,8 377,5 44,7 89,6 12,7 31,2 320,0
T otal 2875,2 2901,2 1571,2 1524,7 286,5 307,0 1043,5
Cuadro No. 2 Devaluacion
Cuadro No. 3 Vencimientos de Forwards
T otal 62 62 1896 1895 253 253 14542 14505 317 317 15646 15592
"""


def test_causal_rolling_z_uses_only_prior_moments():
    original = pd.Series(np.linspace(-1.0, 1.0, 100))
    revised_future = original.copy()
    revised_future.iloc[-1] = 1_000_000.0
    first = _causal_rolling_z(original, window=60, min_periods=20)
    second = _causal_rolling_z(revised_future, window=60, min_periods=20)
    pd.testing.assert_series_equal(first.iloc[:-1], second.iloc[:-1])
    assert abs(second.iloc[-1]) == 5.0


def test_daily_forward_parser_recovers_flows_and_tenors():
    values = parse_daily_forward_text(DAILY_SAMPLE)
    assert values["br_forward_daily_volume_usd_m"] == 4733.0
    assert values["br_forward_daily_implied_devaluation_pct"] == 4.95
    assert values["br_forward_daily_financial_net_usd_m"] == -26.0
    assert values["br_forward_daily_offshore_net_usd_m"] == 46.5
    assert np.isclose(values["br_forward_daily_short_tenor_share"], 0.6669730392)
    assert values["br_forward_daily_ndf_3w_net_usd_m"] == 92.0


def test_monthly_parser_is_scoped_to_derivatives_section():
    text = """
    El monto promedio diario negociado en el mercado de contado fue USD 744,8 m.
    2.1.1 Tamaño y estructura del mercado
    El monto pactado en el mercado forward se incremento al pasar de USD 75.456,4 m
    a USD 83.388,6 m. El monto promedio diario se redujo de USD 3.969,8 m a USD 3.790,4 m.
    Las negociaciones con plazos inferiores a 36 dıas representaron el 70,2%.
    El promedio de devaluación implı́cita ponderado por monto fue de 6,9%.
    Cuadro 2:
    IMC 44.780,68 53.254,06 50.291,08 53.162,36 -5.510,40 91,70
    Extranjero 31.700,97 20.776,05 29.756,86 25.530,29 1.944,10 -4.754,24
    FPC 3.788,82 5.345,92 4.456,28 5.511,24 -667,46 -165,32
    Total 83.388,56 83.388,56 88.120,73 88.120,73 -4.732,16 -4.732,16
    2.1.2 Plazos negociados
    """
    values = parse_monthly_derivatives_text(text)
    assert values["br_forward_monthly_volume_usd_m"] == 83388.6
    assert values["br_forward_monthly_avg_daily_usd_m"] == 3790.4
    assert np.isclose(values["br_forward_monthly_short_tenor_share"], 0.702)
    assert values["br_forward_monthly_imc_net_usd_m"] < 0
    assert values["br_forward_monthly_foreign_net_usd_m"] > 0


def test_consolidated_forward_history_parser_builds_stationary_daily_fields(
    tmp_path: Path,
):
    observation = pd.Timestamp("2025-01-02")
    position = pd.DataFrame(
        [
            {
                "Fecha": observation,
                "Modalidad": "DF",
                "Rango": "4 a 14",
                "Reportante": "IMC",
                "Contraparte": "Extranjero",
                "ComprasPactadas": 100.0,
                "VentasPactadas": 90.0,
                "ComprasVencidas": 0.0,
                "VentasVencidas": 0.0,
                "PosicionNeta": 10.0,
                "MontosNegociados": 100.0,
            },
            {
                "Fecha": observation,
                "Modalidad": "NDF",
                "Rango": "15 a 35",
                "Reportante": "IMC",
                "Contraparte": "Extranjero",
                "ComprasPactadas": 20.0,
                "VentasPactadas": 25.0,
                "ComprasVencidas": 0.0,
                "VentasVencidas": 0.0,
                "PosicionNeta": -5.0,
                "MontosNegociados": 50.0,
            },
        ]
    )
    balance = pd.DataFrame(
        [
            {
                "Fecha": observation,
                "Aseguradora": 1.0,
                "Extranjero": 1000.0,
                "FPC": -500.0,
                "Fiduciaria": 2.0,
                "Persona Natural": 3.0,
                "Real": 400.0,
                "Resto": 4.0,
                "IMC": -910.0,
            }
        ]
    )
    devaluation = pd.DataFrame(
        [
            {
                "Fecha": observation,
                "Rango": "15 a 35",
                "Reportante": "IMC",
                "IMC": 0.08,
                "Extranjero": 0.09,
                "FPC": 0.085,
                "Aseguradora": np.nan,
                "Fiduciaria": np.nan,
                "Real": 0.075,
                "Resto": np.nan,
                "Persona Natural": np.nan,
                "Mercado": 0.08,
                "Total": 0.081,
            }
        ]
    )
    path = tmp_path / "forward_history.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        position.to_excel(
            writer, sheet_name="3. PosicionDiaria", startrow=8, index=False
        )
        balance.to_excel(
            writer, sheet_name="4. SaldoDiario", startrow=6, index=False
        )
        devaluation.to_excel(
            writer,
            sheet_name="6. DevaluacionesSectorTotal",
            startrow=6,
            index=False,
        )

    parsed = parse_forward_history_workbook(path)
    assert len(parsed) == 1
    row = parsed.iloc[0]
    assert row["observation_date"] == observation
    assert row["br_forward_history_foreign_net_usd_m"] == 5.0
    assert np.isclose(row["br_forward_history_foreign_net_ratio"], 5.0 / 150.0)
    assert row["br_forward_history_foreign_ndf_net_ratio"] == -0.1
    assert row["br_forward_history_foreign_balance_usd_m"] == 1000.0
    assert row["br_forward_history_market_devaluation_1m"] == 0.08
    assert row["br_forward_history_foreign_devaluation_1m"] == 0.09


def test_eme_parser_keeps_target_dates(tmp_path: Path):
    rows = [[None] * 6 for _ in range(18)]
    rows[0][0] = "RESULTADOS DE LA ENCUESTA MENSUAL DE EXPECTATIVAS ECONOMICAS"
    rows[1][0] = "Fecha de realizacion: del 9 de diciembre al 11 de diciembre de 2025"
    rows[5] = [
        "Medidas estadisticas", "el 31 de dic./2025", None,
        "el 31 de dic./2026", None, "el 31 de dic./2027",
    ]
    rows[6][0] = "TODAS LAS ENTIDADES PARTICIPANTES"
    rows[8] = ["Media", 3838.2, None, 3953.0, None, 3974.9]
    rows[9] = ["Mediana", 3830.0, None, 3966.4, None, 4000.0]
    rows[12] = ["Desviacion estandar", 62.7, None, 154.6, None, 238.5]
    rows[16] = ["Numero de participantes", 40, None, 39, None, 36]
    path = tmp_path / "eme.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="TRM", header=False, index=False)
    values, survey_end = parse_eme_workbook(path)
    assert survey_end == pd.Timestamp("2025-12-11")
    assert values["br_eme_usdcop_near_mean"][0] == 3838.2
    assert values["br_eme_usdcop_12m_mean"][1] == pd.Timestamp("2026-12-31")


def test_eme_parser_handles_shifted_legacy_layout_and_separates_year_end(tmp_path: Path):
    rows = [[None] * 10 for _ in range(20)]
    rows[2][1] = "RESULTADOS DE LA ENCUESTA MENSUAL DE EXPECTATIVAS ECONOMICAS"
    rows[3][1] = "Fecha de realizacion: del 9 de julio al 11 de julio de 2019"
    rows[5][1] = "Medidas estadisticas"
    rows[6][2] = "el 31 de jul./2019"
    rows[6][4] = "el 31 de dic./2019"
    rows[6][6] = "el 31 de jul./2020"
    rows[8][1] = "TODAS LAS ENTIDADES PARTICIPANTES"
    rows[10][1], rows[10][2], rows[10][4], rows[10][6] = "Media", 3202.9, 3189.6, 3167.1
    rows[11][1], rows[11][2], rows[11][4], rows[11][6] = "Mediana", 3205, 3175, 3150
    rows[14][1], rows[14][2], rows[14][4], rows[14][6] = (
        "Desviacion estandar", 35.3, 87.5, 125.8
    )
    rows[18][1], rows[18][2], rows[18][4], rows[18][6] = (
        "Numero de participantes", 37, 37, 36
    )
    path = tmp_path / "legacy_eme.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="TRM", header=False, index=False)
    values, survey_end = parse_eme_workbook(path)
    assert survey_end == pd.Timestamp("2019-07-11")
    assert values["br_eme_usdcop_year_end_mean"] == (
        3189.6, pd.Timestamp("2019-12-31")
    )
    assert values["br_eme_usdcop_12m_mean"] == (
        3167.1, pd.Timestamp("2020-07-31")
    )


def test_sfc_formato_415_aggregates_usd_cop_sides(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "Fecha Corte": ["2025-01-31"] * 3,
            "Código moneda derecho": ["USD", "COP", "EUR"],
            "Nominal derecho": [2_000_000, 8_000_000_000, 1_000_000],
            "Código moneda obligación": ["COP", "USD", "COP"],
            "Nominal obligación": [8_000_000_000, 1_000_000, 4_000_000_000],
            "Valor del derecho (COP)": [8_100_000_000, 8_000_000_000, 4_100_000_000],
            "Valor de la obligación (COP)": [8_000_000_000, 4_050_000_000, 4_000_000_000],
        }
    )
    path = tmp_path / "portfolio.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Fmto-415", index=False)
    observation, values = parse_sfc_formato_415(path)
    assert observation == pd.Timestamp("2025-01-31")
    assert values["sfc_pension_deriv_usd_long_m"] == 2.0
    assert values["sfc_pension_deriv_usd_short_m"] == 1.0
    assert values["sfc_pension_deriv_usd_net_m"] == 1.0


def test_asof_join_never_uses_a_future_release(tmp_path: Path):
    pit = pd.DataFrame(
        [
            {
                "series_id": "br_eme_usdcop_near_mean",
                "observation_date": "2025-01-31",
                "reference_date": "2025-12-31",
                "release_date": "2025-02-03",
                "available_at": "2025-02-03T20:30:00Z",  # 15:30 Bogota
                "value": 4100.0,
                "frequency": "monthly",
                "unit": "COP_per_USD",
                "source": "banrep_eme",
                "source_url": "https://example.invalid/eme.xlsx",
                "document_sha256": "a" * 64,
                "retrieved_at": "2025-02-04T00:00:00Z",
                "availability_policy": "official_workbook_modified_timestamp",
                "pit_vintage": True,
                "promotion_eligible": True,
                "metadata_json": "{}",
            }
        ],
        columns=PIT_COLUMNS,
    )
    path = tmp_path / "pit.parquet"
    pit.to_parquet(path, index=False)
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-02-02", "2025-02-03", "2025-02-04", "2025-04-20"]
            ),
            "close": [4050.0, 4060.0, 4070.0, 4200.0],
        }
    )
    enriched, features = attach_forward_macro_features(market, path, decision_hour_bogota=16)
    assert pd.isna(enriched.loc[0, "br_eme_usdcop_near_mean"])
    assert enriched.loc[1, "br_eme_usdcop_near_mean"] == 4100.0
    assert enriched.loc[2, "br_eme_usdcop_near_mean"] == 4100.0
    assert pd.isna(enriched.loc[3, "br_eme_usdcop_near_mean"])
    assert "pit_eme_near_gap_pct" in features
    assert "br_eme_usdcop_near_mean__observation_date" in enriched.columns
    assert "br_eme_usdcop_near_mean__available_at" in enriched.columns
    assert enriched.loc[1, "br_eme_usdcop_near_mean__observation_date"] == pd.Timestamp("2025-01-31")


def test_socrata_uses_official_peso_currency_code(tmp_path: Path, monkeypatch):
    calls = []

    class Response:
        def __init__(self, total: str):
            self.total = total

        def raise_for_status(self):
            return None

        def json(self):
            return [{"fecha_corte": "2022-11-30T00:00:00.000", "total": self.total}]

    scraper = ForwardMacroScraper(
        raw_dir=tmp_path / "raw",
        output_path=tmp_path / "pit.parquet",
        manifest_dir=tmp_path / "manifests",
    )

    def fake_get(url, params=None, timeout=None):
        calls.append(params["$where"])
        return Response("2000000" if "comprada='USD'" in params["$where"] else "1000000")

    monkeypatch.setattr(scraper.session, "get", fake_get)
    rows = scraper.scrape_sfc_socrata(
        pd.Timestamp("2022-11-01"), pd.Timestamp("2022-11-30")
    )
    assert len(rows) == 4
    assert all("PESO" in query for query in calls)
    assert not any("='COP'" in query for query in calls)
