#!/usr/bin/env python3
"""Fix escaped quotes in TSX files."""

import os
import re
from pathlib import Path

def fix_escaped_quotes(file_path):
    """Fix escaped quotes in a TSX file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace escaped quotes with regular quotes
    original_content = content
    content = content.replace('className=\\"', 'className="')
    content = content.replace('\\">', '">')
    content = content.replace('\\"}', '"}')
    content = content.replace('\\"', '"')
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {file_path}")
        return True
    return False

def main():
    """Main function."""
    views_dir = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\usdcop-trading-dashboard\components\views")
    
    files_to_fix = [
        "APIUsagePanel.tsx",
        "L6BacktestResults.tsx",
        "L5ModelDashboard.tsx",
        "L4RLReadyData.tsx",
        "L3CorrelationMatrix.tsx",
        "L1FeatureStats.tsx",
        "L0RawDataDashboard.tsx"
    ]
    
    fixed_count = 0
    for file_name in files_to_fix:
        file_path = views_dir / file_name
        if file_path.exists():
            if fix_escaped_quotes(file_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()