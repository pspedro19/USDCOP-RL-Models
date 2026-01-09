#!/usr/bin/env python3
"""
Script to install the monitor router into multi_model_trading_api.py

Run this script once to add the monitoring endpoints:
    python install_monitor.py
"""

import os

API_FILE = os.path.join(os.path.dirname(__file__), 'multi_model_trading_api.py')

ROUTER_CODE = '''
# Include model monitoring router
try:
    from monitor_router import router as monitor_router
    app.include_router(monitor_router, prefix="/api/monitor", tags=["monitoring"])
    logger.info("Model monitoring router included successfully")
except ImportError as e:
    logger.warning(f"Could not import monitor_router: {e}")

'''

def main():
    # Read the file
    with open(API_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Check if already installed
    content = ''.join(lines)
    if 'monitor_router' in content:
        print("Monitor router already installed!")
        return

    # Find the insertion point (after CORS middleware closing paren)
    insert_idx = None
    for i, line in enumerate(lines):
        if 'allow_headers' in line:
            # Next line should be )
            if i + 1 < len(lines) and lines[i + 1].strip() == ')':
                insert_idx = i + 2  # After the )
                break

    if insert_idx is not None:
        # Insert the router code
        lines.insert(insert_idx, ROUTER_CODE)

        # Write back
        with open(API_FILE, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        print("Monitor router installed successfully!")
        print("\nNew endpoints available:")
        print("  GET  /api/monitor/health")
        print("  GET  /api/monitor/{model_id}/health")
        print("  POST /api/monitor/{model_id}/record-action")
        print("  POST /api/monitor/{model_id}/record-pnl")
        print("  POST /api/monitor/{model_id}/set-baseline")
        print("  POST /api/monitor/{model_id}/reset")
    else:
        print("Could not find insertion point. Please add manually.")
        print("\nAdd the following after the CORS middleware configuration:")
        print(ROUTER_CODE)

if __name__ == '__main__':
    main()
