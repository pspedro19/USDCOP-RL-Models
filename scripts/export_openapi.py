#!/usr/bin/env python3
"""
OpenAPI Spec Export Script
==========================

Exports the OpenAPI specification from the Inference API to YAML format.

Usage:
    python scripts/export_openapi.py
    python scripts/export_openapi.py --output docs/api/openapi.yaml

Author: Trading Team
Date: 2026-01-14
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_openapi(output_path: str = "docs/api/openapi_v2.yaml") -> str:
    """
    Export OpenAPI spec from the FastAPI application.

    Args:
        output_path: Output file path (relative to project root)

    Returns:
        Path to the exported file
    """
    try:
        # Import the app to get OpenAPI schema
        from services.inference_api.main import app

        # Get OpenAPI schema
        openapi_schema = app.openapi()

        # Ensure output directory exists
        output_file = PROJECT_ROOT / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Export based on file extension
        if output_path.endswith('.yaml') or output_path.endswith('.yml'):
            if not YAML_AVAILABLE:
                print("Warning: PyYAML not installed. Exporting as JSON instead.")
                output_path = output_path.replace('.yaml', '.json').replace('.yml', '.json')
                output_file = PROJECT_ROOT / output_path

                with open(output_file, 'w') as f:
                    json.dump(openapi_schema, f, indent=2)
            else:
                with open(output_file, 'w') as f:
                    yaml.dump(
                        openapi_schema,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False
                    )
        else:
            with open(output_file, 'w') as f:
                json.dump(openapi_schema, f, indent=2)

        print(f"OpenAPI spec exported to: {output_file}")
        return str(output_file)

    except ImportError as e:
        print(f"Error: Could not import FastAPI app: {e}")
        print("Make sure all dependencies are installed.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Export OpenAPI specification to YAML/JSON'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='docs/api/openapi_v2.yaml',
        help='Output file path (default: docs/api/openapi_v2.yaml)'
    )

    args = parser.parse_args()

    export_openapi(args.output)


if __name__ == '__main__':
    main()
