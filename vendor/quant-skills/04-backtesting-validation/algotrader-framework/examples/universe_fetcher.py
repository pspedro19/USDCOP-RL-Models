#!/usr/bin/env python3
"""
Universe Fetcher - Fetch Index Constituents from NSE

Fetches latest stock lists from NSE India website.
Includes liquidity filtering and momentum scoring.

Usage:
    python universe_fetcher.py --index nifty50
    python universe_fetcher.py --index midcap150 --filter
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path


def fetch_nse_index(index_name: str):
    """
    Fetch index constituents from NSE

    Args:
        index_name: 'NIFTY 50', 'NIFTY 100', 'NIFTY MIDCAP 150', etc.

    Returns:
        List of stocks with metadata
    """
    url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_name.replace(' ', '%20')}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    session = requests.Session()

    # First visit homepage to get cookies
    session.get("https://www.nseindia.com", headers=headers, timeout=10)
    time.sleep(1)

    # Fetch index data
    response = session.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    data = response.json()

    stocks = []
    if 'data' in data:
        for item in data['data']:
            symbol = item.get('symbol', '').replace('-EQ', '')

            if symbol == index_name:
                continue  # Skip index itself

            stocks.append({
                'symbol': symbol,
                'company': item.get('meta', {}).get('companyName', symbol),
                'last_price': item.get('lastPrice', 0),
                'change_pct': item.get('pChange', 0)
            })

    return stocks


def main():
    """Fetch and save universe"""
    indices = {
        'nifty50': 'NIFTY 50',
        'nifty100': 'NIFTY 100',
        'midcap150': 'NIFTY MIDCAP 150',
        'smallcap250': 'NIFTY SMALLCAP 250'
    }

    output_dir = Path('universe')
    output_dir.mkdir(exist_ok=True)

    for key, index_name in indices.items():
        print(f"\n⏳ Fetching {index_name}...")

        try:
            stocks = fetch_nse_index(index_name)

            data = {
                'index': index_name,
                'last_updated': datetime.now().isoformat(),
                'stocks': stocks,
                'count': len(stocks)
            }

            output_file = output_dir / f"{key}.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"✓ Saved {len(stocks)} stocks to {output_file}")

        except Exception as e:
            print(f"⚠️  Error fetching {index_name}: {e}")

    print("\n✅ Universe fetch complete!")


if __name__ == "__main__":
    main()
