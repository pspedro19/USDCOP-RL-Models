"""Capture screenshots of all course project services via Playwright.

Evidence for the MLOps final project delivery (2026-04-23).
"""
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

OUT = Path("docs/slides/course_screenshots")
OUT.mkdir(parents=True, exist_ok=True)

VIEWPORT = {"width": 1600, "height": 900}

PAGES = [
    # name, url, wait_selector_or_time, username/password (optional)
    ("01_airflow_dags", "http://localhost:8080", {"wait_time": 6000}),
    ("02_mlflow_experiments", "http://localhost:5001", {"wait_time": 4000}),
    ("03_redpanda_console_topics", "http://localhost:8088/topics", {"wait_time": 4000}),
    ("04_redpanda_console_messages", "http://localhost:8088/topics/signals.h5", {"wait_time": 5000}),
    ("05_grafana_home", "http://localhost:3002/login", {"wait_time": 3000}),
    ("06_prometheus_targets", "http://localhost:9090/targets", {"wait_time": 4000}),
    ("07_dashboard_home", "http://localhost:5000", {"wait_time": 6000}),
    ("08_dashboard_forecasting", "http://localhost:5000/forecasting", {"wait_time": 8000}),
    ("09_dashboard_dashboard", "http://localhost:5000/dashboard", {"wait_time": 8000}),
    ("10_dashboard_production", "http://localhost:5000/production", {"wait_time": 8000}),
    ("11_dashboard_analysis", "http://localhost:5000/analysis", {"wait_time": 8000}),
    ("12_dashboard_execution", "http://localhost:5000/execution", {"wait_time": 6000}),
    ("13_signalbridge_docs", "http://localhost:8085/docs", {"wait_time": 4000}),
    ("14_trading_api_docs", "http://localhost:8000/docs", {"wait_time": 4000}),
    ("15_minio_console", "http://localhost:9001", {"wait_time": 4000}),
    ("16_pgadmin", "http://localhost:5050", {"wait_time": 4000}),
]


def main():
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT, ignore_https_errors=True)
        page = context.new_page()

        for name, url, opts in PAGES:
            out_path = OUT / f"{name}.png"
            wait_time = opts.get("wait_time", 3000)
            try:
                print(f"[capture] {name} <- {url}")
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(wait_time)
                # Try to scroll a bit for dynamic content
                page.evaluate("window.scrollTo(0, 100)")
                page.wait_for_timeout(500)
                page.evaluate("window.scrollTo(0, 0)")
                page.screenshot(path=str(out_path), full_page=False)
                size = out_path.stat().st_size
                results.append((name, "OK", size, url))
                print(f"  -> saved {out_path} ({size} bytes)")
            except PwTimeout as e:
                results.append((name, f"TIMEOUT: {e}", 0, url))
                print(f"  !! timeout")
            except Exception as e:
                results.append((name, f"ERROR: {type(e).__name__}: {e}", 0, url))
                print(f"  !! {e}")

        browser.close()

    # Summary
    print("\n" + "=" * 70)
    print("SCREENSHOT CAPTURE SUMMARY")
    print("=" * 70)
    for name, status, size, url in results:
        flag = "OK" if status == "OK" else "FAIL"
        print(f"  [{flag:4}] {name:40} {url}")
        if status != "OK":
            print(f"         {status}")
    ok = sum(1 for _, s, _, _ in results if s == "OK")
    print(f"\nCaptured: {ok}/{len(results)}")


if __name__ == "__main__":
    main()
