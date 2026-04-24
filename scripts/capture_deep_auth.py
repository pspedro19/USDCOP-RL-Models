"""Deep authenticated Playwright captures — Grafana, MinIO, pgAdmin, Airflow DAG runs, Redpanda detail."""
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

OUT = Path("docs/slides/course_screenshots")
OUT.mkdir(parents=True, exist_ok=True)
VP = {"width": 1600, "height": 900}


def safe(name, fn):
    try:
        fn()
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {type(e).__name__}: {str(e)[:120]}")


def airflow(ctx):
    p = ctx.new_page()
    p.goto("http://localhost:8080/login/", wait_until="domcontentloaded", timeout=30000)
    p.wait_for_timeout(2000)
    try:
        p.fill("input[name='username']", "admin")
        p.fill("input[name='password']", "admin123")
        p.click("input[type='submit'], button[type='submit']")
        p.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass

    # DAG list — filter forecast
    safe("airflow_dags_home", lambda: (
        p.goto("http://localhost:8080/home", wait_until="networkidle", timeout=25000),
        p.wait_for_timeout(4000),
        p.screenshot(path=str(OUT / "deep_01_airflow_dags_home.png"))
    ))

    # Filter by forecast_h5
    safe("airflow_h5_filter", lambda: (
        p.goto("http://localhost:8080/home?search=forecast_h5", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(3500),
        p.screenshot(path=str(OUT / "deep_02_airflow_h5_filter.png"))
    ))

    # DAG graph h5 l5
    safe("airflow_graph_h5_l5", lambda: (
        p.goto("http://localhost:8080/dags/forecast_h5_l5_weekly_signal/graph", wait_until="networkidle", timeout=30000),
        p.wait_for_timeout(6000),
        p.screenshot(path=str(OUT / "deep_03_airflow_graph_h5_l5.png"))
    ))

    # DAG code view — show the Kafka publishing task
    safe("airflow_code_view", lambda: (
        p.goto("http://localhost:8080/dags/forecast_h5_l5_weekly_signal/code", wait_until="networkidle", timeout=30000),
        p.wait_for_timeout(5000),
        p.screenshot(path=str(OUT / "deep_04_airflow_code.png"))
    ))

    # Runs for forecast_h5_l3 (more likely to have runs)
    safe("airflow_runs", lambda: (
        p.goto("http://localhost:8080/dags/forecast_h5_l3_weekly_training/grid", wait_until="networkidle", timeout=30000),
        p.wait_for_timeout(5000),
        p.screenshot(path=str(OUT / "deep_05_airflow_runs_grid.png"))
    ))

    p.close()


def mlflow(ctx):
    p = ctx.new_page()
    safe("mlflow_root", lambda: (
        p.goto("http://localhost:5001", wait_until="networkidle", timeout=25000),
        p.wait_for_timeout(5000),
        p.screenshot(path=str(OUT / "deep_06_mlflow_root.png"))
    ))
    # list experiments via API and click a run
    safe("mlflow_experiment0", lambda: (
        p.goto("http://localhost:5001/#/experiments/0", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(5000),
        p.screenshot(path=str(OUT / "deep_07_mlflow_exp0.png"))
    ))
    p.close()


def grafana(ctx):
    p = ctx.new_page()
    # Login
    p.goto("http://localhost:3002/login", wait_until="domcontentloaded", timeout=25000)
    p.wait_for_timeout(3000)
    try:
        p.fill("input[name='user']", "admin")
        p.fill("input[name='password']", "admin")
        p.click("button[type='submit']")
        p.wait_for_load_state("networkidle", timeout=15000)
        p.wait_for_timeout(2000)
        # Skip password change
        try:
            p.click("text=Skip", timeout=3000)
        except Exception:
            pass
    except Exception as e:
        print(f"grafana login err: {e}")

    safe("grafana_home", lambda: (
        p.goto("http://localhost:3002/", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(4000),
        p.screenshot(path=str(OUT / "deep_08_grafana_home.png"))
    ))

    safe("grafana_dashboards", lambda: (
        p.goto("http://localhost:3002/dashboards", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(4000),
        p.screenshot(path=str(OUT / "deep_09_grafana_dashboards_list.png"))
    ))

    # Try to open Trading Performance dashboard
    safe("grafana_trading_perf", lambda: (
        p.goto("http://localhost:3002/d/usdcop-trading/trading-performance?orgId=1", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(5000),
        p.screenshot(path=str(OUT / "deep_10_grafana_trading.png"))
    ))

    safe("grafana_datasources", lambda: (
        p.goto("http://localhost:3002/connections/datasources", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(4000),
        p.screenshot(path=str(OUT / "deep_11_grafana_datasources.png"))
    ))

    p.close()


def minio(ctx):
    p = ctx.new_page()
    p.goto("http://localhost:9001/", wait_until="domcontentloaded", timeout=20000)
    p.wait_for_timeout(3000)
    try:
        p.fill("input[name='accessKey']", "admin")
        p.fill("input[name='secretKey']", "admin123")
        p.click("button[type='submit']")
        p.wait_for_load_state("networkidle", timeout=15000)
        p.wait_for_timeout(3000)
    except Exception as e:
        print(f"minio login err: {e}")

    safe("minio_buckets", lambda: (
        p.goto("http://localhost:9001/browser", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(5000),
        p.screenshot(path=str(OUT / "deep_12_minio_buckets.png"))
    ))

    safe("minio_dashboard", lambda: (
        p.goto("http://localhost:9001/dashboard", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(4000),
        p.screenshot(path=str(OUT / "deep_13_minio_dashboard.png"))
    ))

    p.close()


def redpanda(ctx):
    p = ctx.new_page()
    safe("redpanda_home", lambda: (
        p.goto("http://localhost:8088/overview", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(4000),
        p.screenshot(path=str(OUT / "deep_14_redpanda_overview.png"))
    ))
    safe("redpanda_topics_list", lambda: (
        p.goto("http://localhost:8088/topics", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(4000),
        p.screenshot(path=str(OUT / "deep_15_redpanda_topics.png"))
    ))
    safe("redpanda_topic_signals", lambda: (
        p.goto("http://localhost:8088/topics/signals.h5", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(5000),
        p.screenshot(path=str(OUT / "deep_16_redpanda_signals.png"))
    ))
    # Try clicking messages tab specifically
    try:
        p.click("text=Messages", timeout=5000)
        p.wait_for_timeout(4000)
        p.screenshot(path=str(OUT / "deep_17_redpanda_messages.png"))
        print("  ✓ redpanda_messages_tab")
    except Exception as e:
        print(f"  ✗ redpanda_messages_tab: {e}")
    # Consumer groups
    safe("redpanda_cgroups", lambda: (
        p.goto("http://localhost:8088/groups", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(4000),
        p.screenshot(path=str(OUT / "deep_18_redpanda_groups.png"))
    ))

    p.close()


def prometheus(ctx):
    p = ctx.new_page()
    safe("prom_targets", lambda: (
        p.goto("http://localhost:9090/targets", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(4000),
        p.screenshot(path=str(OUT / "deep_19_prom_targets.png"))
    ))
    safe("prom_rules", lambda: (
        p.goto("http://localhost:9090/rules", wait_until="domcontentloaded", timeout=25000),
        p.wait_for_timeout(6000),
        p.screenshot(path=str(OUT / "deep_20_prom_rules.png"))
    ))
    safe("prom_alerts", lambda: (
        p.goto("http://localhost:9090/alerts", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(4000),
        p.screenshot(path=str(OUT / "deep_21_prom_alerts.png"))
    ))
    safe("prom_graph_up", lambda: (
        p.goto("http://localhost:9090/graph?g0.expr=up&g0.tab=1&g0.range_input=1h", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(5000),
        p.screenshot(path=str(OUT / "deep_22_prom_up_metric.png"))
    ))
    p.close()


def dashboard(ctx):
    p = ctx.new_page()
    pages = [
        ("deep_23_dashboard_hub", "http://localhost:5000/hub"),
        ("deep_24_dashboard_forecasting", "http://localhost:5000/forecasting"),
        ("deep_25_dashboard_dashboard", "http://localhost:5000/dashboard"),
        ("deep_26_dashboard_production", "http://localhost:5000/production"),
        ("deep_27_dashboard_analysis", "http://localhost:5000/analysis"),
        ("deep_28_dashboard_execution", "http://localhost:5000/execution/dashboard"),
    ]
    for name, url in pages:
        safe(name, lambda u=url, n=name: (
            p.goto(u, wait_until="networkidle", timeout=30000),
            p.wait_for_timeout(6000),
            p.screenshot(path=str(OUT / f"{n}.png"))
        ))
    p.close()


def signalbridge(ctx):
    p = ctx.new_page()
    safe("signalbridge_docs", lambda: (
        p.goto("http://localhost:8085/docs", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(5000),
        p.screenshot(path=str(OUT / "deep_29_signalbridge_docs.png"))
    ))
    safe("signalbridge_redoc", lambda: (
        p.goto("http://localhost:8085/redoc", wait_until="networkidle", timeout=20000),
        p.wait_for_timeout(5000),
        p.screenshot(path=str(OUT / "deep_30_signalbridge_redoc.png"))
    ))
    p.close()


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VP, ignore_https_errors=True)
        for name, fn in [
            ("AIRFLOW", airflow),
            ("MLFLOW", mlflow),
            ("GRAFANA", grafana),
            ("MINIO", minio),
            ("REDPANDA", redpanda),
            ("PROMETHEUS", prometheus),
            ("DASHBOARD", dashboard),
            ("SIGNALBRIDGE", signalbridge),
        ]:
            print(f"\n=== {name} ===")
            try:
                fn(ctx)
            except Exception as e:
                print(f"  !! {name} overall failed: {e}")
        browser.close()
    print("\nDeep captures complete.")


if __name__ == "__main__":
    main()
