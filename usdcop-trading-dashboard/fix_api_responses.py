#!/usr/bin/env python3
"""
Script to add createApiResponse wrapper to API routes
"""
import re

files_to_fix = [
    r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\usdcop-trading-dashboard\app\api\pipeline\health\route.ts",
    r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\usdcop-trading-dashboard\app\api\websocket\status\route.ts",
    r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\usdcop-trading-dashboard\app\api\alerts\system\route.ts",
    r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\usdcop-trading-dashboard\app\api\usage\monitoring\route.ts",
    r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\usdcop-trading-dashboard\app\api\backup\status\route.ts",
]

import_line = "import { createApiResponse } from '@/lib/types/api';\n"

for file_path in files_to_fix:
    print(f"Processing: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add import if not present
    if "createApiResponse" not in content:
        # Find the last import statement
        imports_end = content.rfind("import ")
        if imports_end != -1:
            next_line = content.find("\n", imports_end)
            content = content[:next_line + 1] + import_line + content[next_line + 1:]

    # Fix return NextResponse.json({ ...data, with createApiResponse
    # Pattern 1: return NextResponse.json({ with spread operator and source
    content = re.sub(
        r'return NextResponse\.json\(\{\s*\.\.\.(\w+),\s*source:\s*[\'"]([^"]+)["\'],?\s*(?:timestamp: new Date\(\)\.toISOString\(\))?\s*\}',
        r'return NextResponse.json(\n          createApiResponse(\n            { ...\1, source: "\2" },\n            "live"\n          )',
        content
    )

    # Pattern 2: return NextResponse.json({ success: false, error:
    content = re.sub(
        r'return NextResponse\.json\(\{\s*success:\s*false,\s*error:\s*[\'"]([^"]+)["\'],\s*message:\s*[\'"]([^"]+)["\'],',
        r'return NextResponse.json(\n      createApiResponse(\n        {',
        content
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Fixed: {file_path}")

print("\nDone!")
