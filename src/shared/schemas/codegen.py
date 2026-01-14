"""
Schema Code Generator
=====================

Generates TypeScript types and Zod schemas from Pydantic models.
Ensures frontend-backend contract synchronization.

Usage:
    python -m shared.schemas.codegen --output ../dashboard/types/generated/

Contract: CTR-SHARED-CODEGEN-001
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

# Import all schemas
from . import (
    # Core
    SignalType,
    TradeSide,
    TradeStatus,
    OrderSide,
    MarketStatus,
    DataSource,
    # Features
    OBSERVATION_DIM,
    FEATURE_ORDER,
    NamedFeatures,
    ObservationSchema,
    FeatureSnapshotSchema,
    NormalizationStatsSchema,
    # Trading
    CandlestickSchema,
    SignalSchema,
    TradeSchema,
    TradeSummarySchema,
    TradeMetadataSchema,
    # API
    BacktestRequestSchema,
    InferenceRequestSchema,
    BacktestResponseSchema,
    HealthResponseSchema,
    ErrorResponseSchema,
    ApiMetadataSchema,
    ApiResponseSchema,
    ModelInfoSchema,
    ModelsResponseSchema,
)


# =============================================================================
# CONSTANTS
# =============================================================================


GENERATED_HEADER = '''/**
 * GENERATED FILE - DO NOT EDIT DIRECTLY
 *
 * Generated from Pydantic schemas in shared/schemas/
 * Run: python -m shared.schemas.codegen
 *
 * Contract: CTR-SHARED-CODEGEN-001
 * Generated: {timestamp}
 */

'''

TYPESCRIPT_IMPORTS = '''import { z } from 'zod';

'''


# =============================================================================
# TYPE MAPPING
# =============================================================================


PYTHON_TO_TS: Dict[str, str] = {
    "str": "string",
    "int": "number",
    "float": "number",
    "bool": "boolean",
    "None": "null",
    "NoneType": "null",
    "Any": "unknown",
    "datetime": "string",
    "date": "string",
    "Decimal": "number",
}


def python_type_to_ts(python_type: str) -> str:
    """Convert Python type annotation to TypeScript."""
    # Handle Optional
    if python_type.startswith("Optional["):
        inner = python_type[9:-1]
        return f"{python_type_to_ts(inner)} | null"

    # Handle List
    if python_type.startswith("List[") or python_type.startswith("list["):
        inner = python_type[5:-1]
        return f"{python_type_to_ts(inner)}[]"

    # Handle Dict
    if python_type.startswith("Dict[") or python_type.startswith("dict["):
        # Simplified - just use Record<string, unknown>
        return "Record<string, unknown>"

    # Handle Tuple
    if python_type.startswith("Tuple[") or python_type.startswith("tuple["):
        # Convert to array for simplicity
        return "unknown[]"

    # Handle Literal
    if python_type.startswith("Literal["):
        inner = python_type[8:-1]
        values = [v.strip() for v in inner.split(",")]
        return " | ".join(values)

    # Direct mapping
    return PYTHON_TO_TS.get(python_type, python_type)


def python_type_to_zod(python_type: str, field_info: Dict[str, Any]) -> str:
    """Convert Python type to Zod schema."""
    # Handle Optional
    if python_type.startswith("Optional["):
        inner = python_type[9:-1]
        inner_zod = python_type_to_zod(inner, field_info)
        return f"{inner_zod}.nullable().optional()"

    # Handle List
    if python_type.startswith("List[") or python_type.startswith("list["):
        inner = python_type[5:-1]
        inner_zod = python_type_to_zod(inner, {})
        return f"z.array({inner_zod})"

    # Handle basic types
    type_map = {
        "str": "z.string()",
        "int": "z.number().int()",
        "float": "z.number()",
        "bool": "z.boolean()",
        "Any": "z.unknown()",
        "datetime": "z.string().datetime()",
        "date": "z.string()",
    }

    base = type_map.get(python_type, "z.unknown()")

    # Add constraints from field_info
    if python_type in ("int", "float"):
        if "ge" in field_info:
            base = f"{base}.min({field_info['ge']})"
        if "le" in field_info:
            base = f"{base}.max({field_info['le']})"
        if "gt" in field_info:
            base = f"{base}.gt({field_info['gt']})"
        if "lt" in field_info:
            base = f"{base}.lt({field_info['lt']})"

    if python_type == "str":
        if "min_length" in field_info:
            base = f"{base}.min({field_info['min_length']})"
        if "max_length" in field_info:
            base = f"{base}.max({field_info['max_length']})"
        if "pattern" in field_info:
            base = f"{base}.regex(/{field_info['pattern']}/)"

    return base


# =============================================================================
# GENERATORS
# =============================================================================


def generate_enum_ts(enum_class: Type) -> str:
    """Generate TypeScript enum from Python enum."""
    name = enum_class.__name__
    values = [f"  {m.name} = '{m.value}'," for m in enum_class]
    return f"export enum {name} {{\n" + "\n".join(values) + "\n}\n"


def generate_enum_zod(enum_class: Type) -> str:
    """Generate Zod enum from Python enum."""
    name = enum_class.__name__
    values = [f"'{m.value}'" for m in enum_class]
    return f"export const {name}Schema = z.enum([{', '.join(values)}]);\n"


def generate_const_ts(name: str, value: Any) -> str:
    """Generate TypeScript constant."""
    if isinstance(value, (list, tuple)):
        items = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
        return f"export const {name} = [{items}] as const;\n"
    elif isinstance(value, int):
        return f"export const {name} = {value};\n"
    elif isinstance(value, str):
        return f"export const {name} = '{value}';\n"
    else:
        return f"export const {name} = {json.dumps(value)};\n"


def get_field_type(field) -> str:
    """Extract type string from Pydantic field."""
    annotation = field.annotation
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def generate_interface_ts(model: Type[BaseModel]) -> str:
    """Generate TypeScript interface from Pydantic model."""
    name = model.__name__
    fields = []

    for field_name, field in model.model_fields.items():
        ts_type = python_type_to_ts(get_field_type(field))
        optional = "?" if not field.is_required() else ""
        description = field.description or ""
        comment = f"  /** {description} */\n" if description else ""
        fields.append(f"{comment}  {field_name}{optional}: {ts_type};")

    return f"export interface {name} {{\n" + "\n".join(fields) + "\n}\n"


def generate_zod_schema(model: Type[BaseModel]) -> str:
    """Generate Zod schema from Pydantic model."""
    name = model.__name__
    fields = []

    for field_name, field in model.model_fields.items():
        field_info = {}
        if hasattr(field, "metadata"):
            for m in field.metadata:
                if hasattr(m, "ge"):
                    field_info["ge"] = m.ge
                if hasattr(m, "le"):
                    field_info["le"] = m.le
                if hasattr(m, "gt"):
                    field_info["gt"] = m.gt
                if hasattr(m, "lt"):
                    field_info["lt"] = m.lt
                if hasattr(m, "min_length"):
                    field_info["min_length"] = m.min_length
                if hasattr(m, "max_length"):
                    field_info["max_length"] = m.max_length

        zod_type = python_type_to_zod(get_field_type(field), field_info)
        fields.append(f"  {field_name}: {zod_type},")

    return f"export const {name}Schema = z.object({{\n" + "\n".join(fields) + "\n}});\n"


# =============================================================================
# MAIN GENERATOR
# =============================================================================


def generate_all_types(output_dir: Path) -> None:
    """Generate all TypeScript types and Zod schemas."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all content
    enums_content = GENERATED_HEADER.format(timestamp=datetime.utcnow().isoformat())
    enums_content += TYPESCRIPT_IMPORTS

    types_content = GENERATED_HEADER.format(timestamp=datetime.utcnow().isoformat())
    types_content += TYPESCRIPT_IMPORTS

    schemas_content = GENERATED_HEADER.format(timestamp=datetime.utcnow().isoformat())
    schemas_content += TYPESCRIPT_IMPORTS

    # Generate enums
    enums_content += "// =============================================================================\n"
    enums_content += "// ENUMS\n"
    enums_content += "// =============================================================================\n\n"

    for enum_class in [SignalType, TradeSide, TradeStatus, OrderSide, MarketStatus, DataSource]:
        enums_content += generate_enum_ts(enum_class) + "\n"
        enums_content += generate_enum_zod(enum_class) + "\n"

    # Generate constants
    enums_content += "// =============================================================================\n"
    enums_content += "// CONSTANTS\n"
    enums_content += "// =============================================================================\n\n"
    enums_content += generate_const_ts("OBSERVATION_DIM", OBSERVATION_DIM)
    enums_content += generate_const_ts("FEATURE_ORDER", FEATURE_ORDER)

    # Generate interfaces
    types_content += "// =============================================================================\n"
    types_content += "// FEATURE TYPES\n"
    types_content += "// =============================================================================\n\n"

    for model in [NamedFeatures, ObservationSchema, NormalizationStatsSchema, FeatureSnapshotSchema]:
        types_content += generate_interface_ts(model) + "\n"

    types_content += "// =============================================================================\n"
    types_content += "// TRADING TYPES\n"
    types_content += "// =============================================================================\n\n"

    for model in [CandlestickSchema, SignalSchema, TradeMetadataSchema, TradeSchema, TradeSummarySchema]:
        types_content += generate_interface_ts(model) + "\n"

    types_content += "// =============================================================================\n"
    types_content += "// API TYPES\n"
    types_content += "// =============================================================================\n\n"

    for model in [BacktestRequestSchema, InferenceRequestSchema, ApiMetadataSchema,
                  BacktestResponseSchema, HealthResponseSchema, ErrorResponseSchema,
                  ModelInfoSchema, ModelsResponseSchema]:
        types_content += generate_interface_ts(model) + "\n"

    # Generate Zod schemas
    schemas_content += "// =============================================================================\n"
    schemas_content += "// FEATURE SCHEMAS\n"
    schemas_content += "// =============================================================================\n\n"

    for model in [NamedFeatures, ObservationSchema, NormalizationStatsSchema]:
        schemas_content += generate_zod_schema(model) + "\n"

    schemas_content += "// =============================================================================\n"
    schemas_content += "// TRADING SCHEMAS\n"
    schemas_content += "// =============================================================================\n\n"

    for model in [CandlestickSchema, TradeMetadataSchema, TradeSchema, TradeSummarySchema]:
        schemas_content += generate_zod_schema(model) + "\n"

    schemas_content += "// =============================================================================\n"
    schemas_content += "// API SCHEMAS\n"
    schemas_content += "// =============================================================================\n\n"

    for model in [BacktestRequestSchema, InferenceRequestSchema,
                  BacktestResponseSchema, HealthResponseSchema, ErrorResponseSchema]:
        schemas_content += generate_zod_schema(model) + "\n"

    # Write files
    (output_dir / "enums.ts").write_text(enums_content)
    (output_dir / "types.ts").write_text(types_content)
    (output_dir / "schemas.ts").write_text(schemas_content)

    # Generate index.ts
    index_content = GENERATED_HEADER.format(timestamp=datetime.utcnow().isoformat())
    index_content += '''export * from './enums';
export * from './types';
export * from './schemas';
'''
    (output_dir / "index.ts").write_text(index_content)

    print(f"Generated TypeScript files in {output_dir}:")
    print(f"  - enums.ts")
    print(f"  - types.ts")
    print(f"  - schemas.ts")
    print(f"  - index.ts")


def generate_openapi_spec(output_path: Path) -> None:
    """Generate OpenAPI spec from FastAPI app."""
    try:
        # Import FastAPI app
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from services.inference_api.main import app

        # Get OpenAPI schema
        openapi_schema = app.openapi()

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(openapi_schema, indent=2))

        print(f"Generated OpenAPI spec: {output_path}")

    except ImportError as e:
        print(f"Warning: Could not import FastAPI app: {e}")
        print("Skipping OpenAPI generation")


# =============================================================================
# CLI
# =============================================================================


def main():
    """Main entry point for code generation."""
    parser = argparse.ArgumentParser(
        description="Generate TypeScript types from Pydantic schemas"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("usdcop-trading-dashboard/types/generated"),
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--openapi",
        type=Path,
        default=None,
        help="Path to output OpenAPI spec JSON"
    )
    parser.add_argument(
        "--skip-types",
        action="store_true",
        help="Skip TypeScript generation"
    )

    args = parser.parse_args()

    # Generate TypeScript types
    if not args.skip_types:
        generate_all_types(args.output)

    # Generate OpenAPI spec
    if args.openapi:
        generate_openapi_spec(args.openapi)


if __name__ == "__main__":
    main()
