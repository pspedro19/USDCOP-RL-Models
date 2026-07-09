"""Deeper screenshots: login to Airflow, navigate to DAG detail, MLflow experiment detail."""
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("docs/slides/course_screenshots")
OUT.mkdir(parents=True, exist_ok=True)
VIEWPORT = {"width": 1600, "height": 900}


def airflow_login_and_capture(ctx):
    p = ctx.new_page()
    # Login
    p.goto("http://localhost:8080/login/", wait_until="domcontentloaded", timeout=30000)
    p.wait_for_timeout(2000)
    try:
        p.fill("input[name='username']", "admin")
        p.fill("input[name='password']", "admin123")
        p.click("input[type='submit'], button[type='submit']")
        p.wait_for_load_state("networkidle", timeout=15000)
    except Exception as e:
        print(f"  airflow login error: {e}")
    p.wait_for_timeout(3000)
    # DAG list
    p.goto("http://localhost:8080/home", wait_until="networkidle", timeout=30000)
    p.wait_for_timeout(4000)
    p.screenshot(path=str(OUT / "01_airflow_dags.png"), full_page=False)
    print("  Airflow DAG list captured")
    # Filter by forecast_h5
    try:
        p.goto("http://localhost:8080/home?search=forecast_h5", wait_until="networkidle", timeout=30000)
        p.wait_for_timeout(3000)
        p.screenshot(path=str(OUT / "01b_airflow_h5_dags.png"), full_page=False)
        print("  Airflow H5 filter captured")
    except Exception as e:
        print(f"  H5 filter err: {e}")
    # Specific DAG graph
    try:
        p.goto("http://localhost:8080/dags/forecast_h5_l5_weekly_signal/graph", wait_until="networkidle", timeout=30000)
        p.wait_for_timeout(5000)
        p.screenshot(path=str(OUT / "01c_airflow_dag_graph_h5_l5.png"), full_page=False)
        print("  Airflow DAG graph captured")
    except Exception as e:
        print(f"  graph err: {e}")
    p.close()


def mlflow_capture(ctx):
    p = ctx.new_page()
    p.goto("http://localhost:5001/#/experiments", wait_until="networkidle", timeout=30000)
    p.wait_for_timeout(5000)
    p.screenshot(path=str(OUT / "02_mlflow_experiments.png"), full_page=False)
    print("  MLflow experiments captured")
    # Try to click into first experiment
    try:
        # MLflow has experiments listed in left sidebar
        p.goto("http://localhost:5001/#/experiments/0", wait_until="networkidle", timeout=20000)
        p.wait_for_timeout(4000)
        p.screenshot(path=str(OUT / "02b_mlflow_experiment_detail.png"), full_page=False)
        print("  MLflow experiment detail captured")
    except Exception as e:
        print(f"  MLflow detail err: {e}")
    p.close()


def redpanda_capture(ctx):
    p = ctx.new_page()
    p.goto("http://localhost:8088/topics/signals.h5", wait_until="networkidle", timeout=20000)
    p.wait_for_timeout(4000)
    # Click on "Messages" tab if present
    try:
        p.click("text=Messages", timeout=5000)
        p.wait_for_timeout(3000)
        p.screenshot(path=str(OUT / "04b_redpanda_messages_tab.png"), full_page=False)
        print("  Redpanda messages tab captured")
    except Exception as e:
        print(f"  redpanda tab err: {e}")
    # Topic config
    try:
        p.click("text=Configuration", timeout=3000)
        p.wait_for_timeout(2000)
        p.screenshot(path=str(OUT / "04c_redpanda_config.png"), full_page=False)
    except Exception:
        pass
    p.close()


def prometheus_capture(ctx):
    p = ctx.new_page()
    # Graph view with sample query
    p.goto("http://localhost:9090/graph?g0.expr=up&g0.tab=1", wait_until="networkidle", timeout=20000)
    p.wait_for_timeout(4000)
    p.screenshot(path=str(OUT / "06b_prometheus_up_query.png"), full_page=False)
    print("  Prometheus up query captured")
    # Rules
    try:
        p.goto("http://localhost:9090/rules", wait_until="networkidle", timeout=15000)
        p.wait_for_timeout(3000)
        p.screenshot(path=str(OUT / "06c_prometheus_rules.png"), full_page=False)
        print("  Prometheus rules captured")
    except Exception as e:
        print(f"  rules err: {e}")
    p.close()


def dashboard_deep(ctx):
    p = ctx.new_page()
    # Ensure forecasting is scrolled to show charts
    p.goto("http://localhost:5000/forecasting", wait_until="networkidle", timeout=30000)
    p.wait_for_timeout(8000)
    p.screenshot(path=str(OUT / "08_dashboard_forecasting.png"), full_page=False)
    # Scroll down for model zoo
    p.evaluate("window.scrollTo(0, 600)")
    p.wait_for_timeout(2000)
    p.screenshot(path=str(OUT / "08b_dashboard_forecasting_scrolled.png"), full_page=False)
    print("  Forecasting scrolled captured")
    # Hub page
    try:
        p.goto("http://localhost:5000/hub", wait_until="networkidle", timeout=20000)
        p.wait_for_timeout(4000)
        p.screenshot(path=str(OUT / "07b_dashboard_hub.png"), full_page=False)
        print("  Dashboard hub captured")
    except Exception as e:
        print(f"  hub err: {e}")
    p.close()


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VIEWPORT, ignore_https_errors=True)
        print("-- Airflow")
        try:
            airflow_login_and_capture(ctx)
        except Exception as e:
            print(f"airflow failed: {e}")
        print("-- MLflow")
        try:
            mlflow_capture(ctx)
        except Exception as e:
            print(f"mlflow failed: {e}")
        print("-- Redpanda")
        try:
            redpanda_capture(ctx)
        except Exception as e:
            print(f"redpanda failed: {e}")
        print("-- Prometheus")
        try:
            prometheus_capture(ctx)
        except Exception as e:
            print(f"prometheus failed: {e}")
        print("-- Dashboard")
        try:
            dashboard_deep(ctx)
        except Exception as e:
            print(f"dashboard failed: {e}")
        browser.close()
    print("\nDeep screenshots done.")


if __name__ == "__main__":
    main()
