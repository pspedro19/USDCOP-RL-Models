"""
Dashboard Visual Analyzer - Playwright
=======================================
Captures screenshots and console logs from the trading dashboard.

Usage: python debug/dashboard_analyzer.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Try to import playwright
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Playwright not installed. Installing...")
    os.system("pip install playwright")
    os.system("playwright install chromium")
    from playwright.sync_api import sync_playwright

# Configuration
DASHBOARD_URL = "http://localhost:5000"  # Docker maps 3000 -> 5000
LOGIN_CREDENTIALS = {"username": "admin", "password": "admin123"}
DEBUG_DIR = Path(__file__).parent
SCREENSHOTS_DIR = DEBUG_DIR / "screenshots"
LOGS_DIR = DEBUG_DIR / "logs"

# Create directories
SCREENSHOTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Pages to capture - CORRECT ROUTES
PAGES_TO_CAPTURE = [
    {"name": "login", "path": "/login", "wait_for": None, "login_required": False},
    {"name": "dashboard", "path": "/", "wait_for": "networkidle", "login_required": True},
    {"name": "trade-history", "path": "/trade-history", "wait_for": "networkidle", "login_required": True},
    {"name": "risk", "path": "/risk", "wait_for": "networkidle", "login_required": True},
    {"name": "trading", "path": "/trading", "wait_for": "networkidle", "login_required": True},
    {"name": "ml-analytics", "path": "/ml-analytics", "wait_for": "networkidle", "login_required": True},
    {"name": "agent-trading", "path": "/agent-trading", "wait_for": "networkidle", "login_required": True},
]


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    console_logs = []
    network_errors = []
    page_errors = []

    print("=" * 60)
    print("DASHBOARD VISUAL ANALYZER")
    print(f"Timestamp: {timestamp}")
    print(f"Target: {DASHBOARD_URL}")
    print("=" * 60)

    with sync_playwright() as p:
        # Launch browser
        print("\n[1] Launching browser...")
        browser = p.chromium.launch(
            headless=True,  # Set to False for visual debugging
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )

        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
        )

        page = context.new_page()

        # Capture console logs
        def handle_console(msg):
            log_entry = {
                "type": msg.type,
                "text": msg.text,
                "url": page.url,
                "timestamp": datetime.now().isoformat()
            }
            console_logs.append(log_entry)
            # Print important logs
            if msg.type in ["error", "warning"]:
                print(f"  [{msg.type.upper()}] {msg.text[:100]}...")

        # Capture page errors
        def handle_pageerror(err):
            error_entry = {
                "error": str(err),
                "url": page.url,
                "timestamp": datetime.now().isoformat()
            }
            page_errors.append(error_entry)
            print(f"  [PAGE ERROR] {str(err)[:100]}...")

        # Capture failed requests
        def handle_request_failed(request):
            error_entry = {
                "url": request.url,
                "method": request.method,
                "failure": request.failure,
                "timestamp": datetime.now().isoformat()
            }
            network_errors.append(error_entry)
            print(f"  [NETWORK FAIL] {request.method} {request.url[:80]}...")

        page.on("console", handle_console)
        page.on("pageerror", handle_pageerror)
        page.on("requestfailed", handle_request_failed)

        # Step 2: Login
        print("\n[2] Navigating to login page...")
        try:
            page.goto(f"{DASHBOARD_URL}/login", wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2000)

            # Take login page screenshot
            screenshot_path = SCREENSHOTS_DIR / f"{timestamp}_01_login_page.png"
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  Screenshot saved: {screenshot_path.name}")

            # Find and fill login form
            print("\n[3] Attempting login...")

            # Try different selectors for username/password fields
            username_selectors = [
                'input[name="username"]',
                'input[type="text"]',
                'input[placeholder*="user" i]',
                'input[placeholder*="email" i]',
                '#username',
            ]

            password_selectors = [
                'input[name="password"]',
                'input[type="password"]',
                '#password',
            ]

            username_input = None
            password_input = None

            for selector in username_selectors:
                try:
                    if page.locator(selector).count() > 0:
                        username_input = page.locator(selector).first
                        print(f"  Found username field: {selector}")
                        break
                except:
                    continue

            for selector in password_selectors:
                try:
                    if page.locator(selector).count() > 0:
                        password_input = page.locator(selector).first
                        print(f"  Found password field: {selector}")
                        break
                except:
                    continue

            if username_input and password_input:
                username_input.fill(LOGIN_CREDENTIALS["username"])
                password_input.fill(LOGIN_CREDENTIALS["password"])

                # Screenshot after filling form
                screenshot_path = SCREENSHOTS_DIR / f"{timestamp}_02_login_filled.png"
                page.screenshot(path=str(screenshot_path), full_page=True)

                # Find and click submit button
                submit_selectors = [
                    'button[type="submit"]',
                    'button:has-text("Login")',
                    'button:has-text("Sign in")',
                    'button:has-text("Iniciar")',
                    'input[type="submit"]',
                ]

                for selector in submit_selectors:
                    try:
                        if page.locator(selector).count() > 0:
                            page.locator(selector).first.click()
                            print(f"  Clicked submit: {selector}")
                            break
                    except:
                        continue

                # Wait for navigation
                page.wait_for_timeout(3000)
                page.wait_for_load_state("networkidle")

                # Screenshot after login
                screenshot_path = SCREENSHOTS_DIR / f"{timestamp}_03_after_login.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"  Screenshot saved: {screenshot_path.name}")
                print(f"  Current URL: {page.url}")
            else:
                print("  Could not find login form fields!")

        except Exception as e:
            print(f"  Login error: {e}")
            screenshot_path = SCREENSHOTS_DIR / f"{timestamp}_error_login.png"
            page.screenshot(path=str(screenshot_path), full_page=True)

        # Step 4: Capture dashboard pages
        print("\n[4] Capturing dashboard pages...")

        page_num = 4
        for page_config in PAGES_TO_CAPTURE:
            if page_config["name"] == "login":
                continue  # Already captured

            try:
                print(f"\n  Navigating to: {page_config['path']}")
                page.goto(f"{DASHBOARD_URL}{page_config['path']}", timeout=30000)

                if page_config["wait_for"]:
                    page.wait_for_load_state(page_config["wait_for"])

                page.wait_for_timeout(3000)  # Extra time for JS rendering

                # Full page screenshot
                screenshot_path = SCREENSHOTS_DIR / f"{timestamp}_{page_num:02d}_{page_config['name']}_full.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"  Full screenshot: {screenshot_path.name}")

                # Viewport screenshot
                screenshot_path = SCREENSHOTS_DIR / f"{timestamp}_{page_num:02d}_{page_config['name']}_viewport.png"
                page.screenshot(path=str(screenshot_path), full_page=False)
                print(f"  Viewport screenshot: {screenshot_path.name}")

                page_num += 1

            except Exception as e:
                print(f"  Error on {page_config['name']}: {e}")
                screenshot_path = SCREENSHOTS_DIR / f"{timestamp}_error_{page_config['name']}.png"
                try:
                    page.screenshot(path=str(screenshot_path), full_page=True)
                except:
                    pass

        # Close browser
        browser.close()

    # Save logs
    print("\n[5] Saving logs...")

    # Console logs
    logs_file = LOGS_DIR / f"{timestamp}_console_logs.json"
    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump(console_logs, f, indent=2, ensure_ascii=False)
    print(f"  Console logs: {logs_file.name} ({len(console_logs)} entries)")

    # Network errors
    if network_errors:
        errors_file = LOGS_DIR / f"{timestamp}_network_errors.json"
        with open(errors_file, "w", encoding="utf-8") as f:
            json.dump(network_errors, f, indent=2, ensure_ascii=False)
        print(f"  Network errors: {errors_file.name} ({len(network_errors)} entries)")

    # Page errors
    if page_errors:
        errors_file = LOGS_DIR / f"{timestamp}_page_errors.json"
        with open(errors_file, "w", encoding="utf-8") as f:
            json.dump(page_errors, f, indent=2, ensure_ascii=False)
        print(f"  Page errors: {errors_file.name} ({len(page_errors)} entries)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Screenshots saved to: {SCREENSHOTS_DIR}")
    print(f"Logs saved to: {LOGS_DIR}")
    print(f"Console logs: {len(console_logs)}")
    print(f"Console errors: {len([l for l in console_logs if l['type'] == 'error'])}")
    print(f"Console warnings: {len([l for l in console_logs if l['type'] == 'warning'])}")
    print(f"Network errors: {len(network_errors)}")
    print(f"Page errors: {len(page_errors)}")
    print("=" * 60)

    # Return summary for programmatic use
    return {
        "timestamp": timestamp,
        "screenshots_dir": str(SCREENSHOTS_DIR),
        "logs_dir": str(LOGS_DIR),
        "console_logs": len(console_logs),
        "console_errors": len([l for l in console_logs if l['type'] == 'error']),
        "network_errors": len(network_errors),
        "page_errors": len(page_errors),
    }


if __name__ == "__main__":
    result = main()
    print(f"\nAnalysis complete: {result}")
