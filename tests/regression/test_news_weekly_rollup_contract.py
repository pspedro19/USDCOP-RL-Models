from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
L8 = ROOT / "airflow" / "dags" / "analysis_l8_daily_generation.py"
NEWS = ROOT / "airflow" / "dags" / "news_daily_pipeline.py"


def test_daily_news_pipeline_runs_three_times_on_weekdays():
    text = NEWS.read_text(encoding="utf-8")
    assert 'schedule="0 7,12,18 * * 1-5"' in text
    assert "ingest >> enrich >> cross_ref >> [features, digest]" in text


def test_l8_updates_current_week_daily_and_closes_on_friday():
    text = L8.read_text(encoding="utf-8")
    assert 'schedule="0 19 * * 1-5"' in text
    assert "generate_for_date(today)" in text
    assert "if today.weekday() == 4" in text
    assert "generate_for_week(year, week)" in text
    assert "daily_entries" in text


def test_news_pipeline_waits_before_analysis():
    text = L8.read_text(encoding="utf-8")
    assert "external_dag_id='news_daily_pipeline'" in text
    assert "soft_fail=True" in text
