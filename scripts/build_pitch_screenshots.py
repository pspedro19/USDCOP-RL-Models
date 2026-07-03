"""Capture dashboard screenshots for the GlobalMinds pitch deck."""
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("presentation/globalminds_microsoft_pitch_may2026/assets/screenshots")
OUT.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("forecasting", "http://localhost:5000/forecasting"),
    ("dashboard", "http://localhost:5000/dashboard"),
    ("production", "http://localhost:5000/production"),
]

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=2)
        page = ctx.new_page()
        for name, url in PAGES:
            print(f"-> {name}: {url}")
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
            except Exception as e:
                print(f"   networkidle timeout, falling back to load: {e}")
                page.goto(url, wait_until="load", timeout=30000)
            page.wait_for_timeout(3500)
            out = OUT / f"{name}.png"
            page.screenshot(path=str(out), full_page=False)
            print(f"   saved {out} ({out.stat().st_size // 1024} KB)")
        browser.close()

if __name__ == "__main__":
    main()
