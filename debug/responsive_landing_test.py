"""
Responsive Landing Page Tester
==============================
Captures landing page screenshots at multiple viewport sizes.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Installing playwright...")
    os.system("pip install playwright")
    os.system("playwright install chromium")
    from playwright.sync_api import sync_playwright

DASHBOARD_URL = "http://localhost:5000"
DEBUG_DIR = Path(__file__).parent
SCREENSHOTS_DIR = DEBUG_DIR / "screenshots" / "responsive"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Viewport configurations
VIEWPORTS = [
    {"name": "desktop_4k", "width": 2560, "height": 1440},
    {"name": "desktop_hd", "width": 1920, "height": 1080},
    {"name": "laptop", "width": 1366, "height": 768},
    {"name": "tablet_landscape", "width": 1024, "height": 768},
    {"name": "tablet_portrait", "width": 768, "height": 1024},
    {"name": "mobile_large", "width": 428, "height": 926},  # iPhone 13 Pro Max
    {"name": "mobile_medium", "width": 390, "height": 844},  # iPhone 14
    {"name": "mobile_small", "width": 375, "height": 667},  # iPhone SE
    {"name": "mobile_xs", "width": 320, "height": 568},  # iPhone 5
]


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    console_errors = []

    print("=" * 60)
    print("RESPONSIVE LANDING PAGE TESTER")
    print(f"Timestamp: {timestamp}")
    print(f"Target: {DASHBOARD_URL}")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for vp in VIEWPORTS:
            print(f"\n[{vp['name']}] Testing {vp['width']}x{vp['height']}...")

            context = browser.new_context(
                viewport={"width": vp["width"], "height": vp["height"]},
                device_scale_factor=2 if "mobile" in vp["name"] else 1,
            )
            page = context.new_page()

            # Capture console errors
            def handle_console(msg):
                if msg.type == "error":
                    console_errors.append({
                        "viewport": vp["name"],
                        "error": msg.text[:500]
                    })

            page.on("console", handle_console)

            try:
                # Landing page (not logged in)
                page.goto(DASHBOARD_URL, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(2000)

                # Full page screenshot
                screenshot_path = SCREENSHOTS_DIR / f"{timestamp}_{vp['name']}_landing_full.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"  Full page: {screenshot_path.name}")

                # Above the fold screenshot
                screenshot_path = SCREENSHOTS_DIR / f"{timestamp}_{vp['name']}_landing_fold.png"
                page.screenshot(path=str(screenshot_path), full_page=False)
                print(f"  Above fold: {screenshot_path.name}")

                # Scroll down and capture features section
                page.evaluate("window.scrollTo(0, window.innerHeight)")
                page.wait_for_timeout(500)
                screenshot_path = SCREENSHOTS_DIR / f"{timestamp}_{vp['name']}_landing_features.png"
                page.screenshot(path=str(screenshot_path), full_page=False)
                print(f"  Features: {screenshot_path.name}")

                # Test mobile menu if mobile viewport
                if "mobile" in vp["name"] or vp["name"] == "tablet_portrait":
                    page.goto(DASHBOARD_URL, wait_until="networkidle", timeout=30000)
                    page.wait_for_timeout(1000)

                    # Try to click mobile menu button
                    menu_btn = page.locator('button[aria-label*="menu" i], button:has(svg.w-6)')
                    if menu_btn.count() > 0:
                        menu_btn.first.click()
                        page.wait_for_timeout(500)
                        screenshot_path = SCREENSHOTS_DIR / f"{timestamp}_{vp['name']}_mobile_menu.png"
                        page.screenshot(path=str(screenshot_path), full_page=False)
                        print(f"  Mobile menu: {screenshot_path.name}")

            except Exception as e:
                print(f"  Error: {e}")

            context.close()

        browser.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Screenshots saved to: {SCREENSHOTS_DIR}")
    print(f"Total viewports tested: {len(VIEWPORTS)}")
    print(f"Console errors: {len(console_errors)}")

    if console_errors:
        print("\nConsole errors by viewport:")
        for err in console_errors[:10]:
            print(f"  [{err['viewport']}] {err['error'][:80]}...")

    print("=" * 60)
    return {"timestamp": timestamp, "errors": len(console_errors)}


if __name__ == "__main__":
    result = main()
    print(f"\nTest complete: {result}")
