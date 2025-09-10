#!/usr/bin/env python3
"""Fix all corrupted TSX files by restoring proper formatting."""

import os
import re
from pathlib import Path

def fix_tsx_file(file_path):
    """Fix a corrupted TSX file by properly formatting it."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace literal \n with actual newlines
    content = content.replace('\\n', '\n')
    
    # Fix escaped quotes
    content = content.replace('\\"', '"')
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {file_path.name}")
    return True

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
    
    for file_name in files_to_fix:
        file_path = views_dir / file_name
        if file_path.exists():
            fix_tsx_file(file_path)
    
    print("\nAll files fixed!")

if __name__ == "__main__":
    main()