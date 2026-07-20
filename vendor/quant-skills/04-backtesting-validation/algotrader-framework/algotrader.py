#!/usr/bin/env python3
"""
AlgoTrader: Quantitative Trading Expert for Claude Code

This is the main CLI interface with interactive bot generation wizard,
universe fetcher, signal analyzer, and code optimizer.

Usage:
    /algotrader                    # Interactive wizard
    /algotrader wizard             # Bot generation wizard
    /algotrader universe           # Fetch index constituents
    /algotrader signal RELIANCE    # Generate signal for symbol
    /algotrader check ./bot.py     # Analyze code for issues
    /algotrader optimize ./bot.py  # Optimize for performance

Author: Trading Systems Lab
Version: 1.0.0
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

# ============================================================================
# SECTION: Imports and Dependencies
# PURPOSE: Lazy imports to avoid loading heavy dependencies unless needed
# CRITICAL: Only import what's needed for current command
# TOKEN-SAVING: Don't load Polars/KiteConnect unless actually trading
# ============================================================================

def check_dependencies():
    """
    Check if required dependencies are installed
    Guide user to install if missing
    """
    required = {
        'polars': 'pip install polars',
        'kiteconnect': 'pip install kiteconnect',
        'requests': 'pip install requests',
        'bs4': 'pip install beautifulsoup4'
    }

    missing = []
    for package, install_cmd in required.items():
        try:
            __import__(package)
        except ImportError:
            missing.append((package, install_cmd))

    if missing:
        print("⚠️  Missing dependencies:")
        for package, cmd in missing:
            print(f"   - {package}: {cmd}")
        print("\nInstall all: pip install polars kiteconnect requests beautifulsoup4")
        return False

    return True


# ============================================================================
# SECTION: CLI Menu System
# PURPOSE: Beautiful, interactive menu-driven experience
# UX: Users can type numbers or commands
# ============================================================================

class Colors:
    """ANSI color codes for beautiful terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print beautiful header"""
    width = 66
    print(f"\n{Colors.CYAN}╔{'═' * width}╗{Colors.ENDC}")
    padding = (width - len(text)) // 2
    print(f"{Colors.CYAN}║{' ' * padding}{Colors.BOLD}{text}{Colors.ENDC}{' ' * (width - padding - len(text))}{Colors.CYAN}║{Colors.ENDC}")
    print(f"{Colors.CYAN}╚{'═' * width}╝{Colors.ENDC}\n")


def print_menu(title: str, options: List[str], sections: Optional[Dict[str, List[int]]] = None):
    """
    Print menu with numbered options

    Args:
        title: Menu title
        options: List of option strings
        sections: Optional dict of section names to option indices
    """
    print(f"\n{Colors.BOLD}{title}{Colors.ENDC}\n")

    if sections:
        for section_name, indices in sections.items():
            print(f"{Colors.YELLOW}{section_name}{Colors.ENDC}")
            for i in indices:
                print(f"  {i}. {options[i-1]}")
            print()
    else:
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")


def get_user_choice(prompt: str = "Choice", valid_range: Optional[range] = None) -> str:
    """
    Get user input with validation

    Args:
        prompt: Prompt text
        valid_range: Optional range of valid numeric choices

    Returns:
        User input (string)
    """
    while True:
        try:
            choice = input(f"\n{Colors.GREEN}{prompt}:{Colors.ENDC} ").strip()

            if not choice:
                continue

            # Check if numeric choice
            if valid_range and choice.isdigit():
                choice_num = int(choice)
                if choice_num in valid_range:
                    return choice
                else:
                    print(f"{Colors.RED}Invalid choice. Please enter {valid_range.start}-{valid_range.stop-1}{Colors.ENDC}")
                    continue

            return choice

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Operation cancelled{Colors.ENDC}")
            sys.exit(0)


# ============================================================================
# SECTION: Folder Scanner
# PURPOSE: Scan current directory for existing trading code
# LLM-FRIENDLY: Detects patterns like signal_generator, backtest, zerodha_client
# RETURNS: Dict with found files and what they likely do
# ============================================================================

def scan_folder_for_trading_code(folder: str = ".") -> Dict[str, Any]:
    """
    Scan folder for existing trading code

    Returns:
        {
            'has_trading_code': bool,
            'files': {
                'signal_generator': 'path/to/file.py',
                'backtest': 'path/to/file.py',
                'zerodha_client': 'path/to/file.py',
                'config': 'path/to/config.json',
                'universe': 'path/to/universe.json'
            },
            'issues': ['tick size not rounded', 'no symbol cooldown']
        }
    """
    folder_path = Path(folder)

    # Patterns to look for
    patterns = {
        'signal_generator': ['signal', 'strategy', 'indicator'],
        'backtest': ['backtest', 'test', 'simulation'],
        'zerodha_client': ['zerodha', 'kite', 'broker', 'api'],
        'data_manager': ['data', 'cache', 'fetch'],
        'risk_manager': ['risk', 'position', 'capital'],
        'config': ['config', 'settings', 'params'],
        'universe': ['universe', 'stocks', 'symbols']
    }

    found_files = {}

    # Scan Python files (skip venv, .venv, __pycache__, etc.)
    exclude_dirs = {'venv', '.venv', '__pycache__', 'node_modules', '.git'}
    for py_file in folder_path.rglob("*.py"):
        # Skip if in excluded directories
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue

        file_name_lower = py_file.stem.lower()
        file_content = py_file.read_text(errors='ignore').lower()

        for category, keywords in patterns.items():
            if any(kw in file_name_lower for kw in keywords):
                found_files[category] = str(py_file)
            elif any(kw in file_content for kw in keywords):
                if category not in found_files:
                    found_files[category] = str(py_file)

    # Scan JSON files
    for json_file in folder_path.rglob("*.json"):
        file_name_lower = json_file.stem.lower()

        if 'config' in file_name_lower or 'settings' in file_name_lower:
            found_files['config'] = str(json_file)
        elif 'universe' in file_name_lower or 'stocks' in file_name_lower:
            found_files['universe'] = str(json_file)

    has_trading_code = len(found_files) > 0

    return {
        'has_trading_code': has_trading_code,
        'files': found_files,
        'issues': []  # Populated by analyze_code if needed
    }


# ============================================================================
# SECTION: Interactive Wizard
# PURPOSE: Guide user through bot generation with smart questions
# FLOW: Scan folder → Ask questions → Generate complete bot
# TOKEN-SAVING: Generates production-ready code, not just examples
# ============================================================================

def wizard_main():
    """
    Main wizard flow for generating trading bot
    """
    print_header("ALGOTRADER BOT GENERATION WIZARD")

    print("Scanning current directory for existing trading code...")
    scan_result = scan_folder_for_trading_code(".")

    if scan_result['has_trading_code']:
        print(f"\n{Colors.GREEN}✓ Found existing trading code:{Colors.ENDC}")
        for category, filepath in scan_result['files'].items():
            print(f"  - {category}: {filepath}")

        print("\nWhat would you like to do?")
        print("  1. Generate new trading bot (will create in new folder)")
        print("  2. Enhance existing code (fix issues, optimize)")
        print("  3. Just analyze existing code")
        print("  4. Exit")

        choice = get_user_choice("Choice", range(1, 5))

        if choice == "1":
            wizard_generate_new_bot()
        elif choice == "2":
            wizard_enhance_existing(scan_result['files'])
        elif choice == "3":
            wizard_analyze_code(scan_result['files'])
        else:
            print("Goodbye!")
            return
    else:
        print(f"\n{Colors.YELLOW}No existing trading code found.{Colors.ENDC}")
        print("Let's create a new trading bot!\n")
        wizard_generate_new_bot()


def wizard_generate_new_bot():
    """
    Generate new trading bot from scratch
    Ask strategic questions to customize generation
    """
    print_header("NEW BOT CONFIGURATION")

    config = {}

    # Question 1: Trading style
    print(f"{Colors.BOLD}1. What trading style?{Colors.ENDC}")
    print("  1. Intraday (same-day, high frequency)")
    print("  2. Swing (multi-day, 3-10 days)")
    print("  3. Positional (multi-week, 2-8 weeks)")

    style_choice = get_user_choice("Trading style", range(1, 4))
    config['trading_style'] = ['intraday', 'swing', 'positional'][int(style_choice) - 1]

    # Question 2: Universe
    print(f"\n{Colors.BOLD}2. Which stock universe?{Colors.ENDC}")
    print("  1. Nifty 50 (largecap, high liquidity)")
    print("  2. Nifty 100 (largecap + midcap mix)")
    print("  3. Nifty Midcap 150 (midcap, higher volatility)")
    print("  4. Custom (I'll provide my own list)")

    universe_choice = get_user_choice("Universe", range(1, 5))
    config['universe'] = ['nifty50', 'nifty100', 'midcap150', 'custom'][int(universe_choice) - 1]

    # Question 3: Strategy preference
    print(f"\n{Colors.BOLD}3. Preferred strategy type?{Colors.ENDC}")
    print("  1. Momentum (trend-following, breakouts)")
    print("  2. Mean Reversion (VWAP pullback, oversold)")
    print("  3. Hybrid (Fortress - multi-factor confirmation)")

    strategy_choice = get_user_choice("Strategy", range(1, 4))
    config['strategy'] = ['momentum', 'mean_reversion', 'fortress'][int(strategy_choice) - 1]

    # Question 4: Capital
    print(f"\n{Colors.BOLD}4. Starting capital (in Lakhs)?{Colors.ENDC}")
    capital_input = get_user_choice("Capital (e.g., 10 for ₹10L)")
    try:
        config['capital'] = float(capital_input) * 100000
    except:
        print(f"{Colors.YELLOW}Invalid input, defaulting to ₹10L{Colors.ENDC}")
        config['capital'] = 1000000

    # Question 5: Risk tolerance
    print(f"\n{Colors.BOLD}5. Risk tolerance per trade?{Colors.ENDC}")
    print("  1. Conservative (0.5% per trade)")
    print("  2. Balanced (1.0% per trade) - Recommended")
    print("  3. Aggressive (2.0% per trade)")

    risk_choice = get_user_choice("Risk tolerance", range(1, 4))
    config['risk_pct'] = [0.5, 1.0, 2.0][int(risk_choice) - 1]

    # Confirmation
    print(f"\n{Colors.BOLD}Configuration Summary:{Colors.ENDC}")
    print(f"  Trading Style: {config['trading_style']}")
    print(f"  Universe: {config['universe']}")
    print(f"  Strategy: {config['strategy']}")
    print(f"  Capital: ₹{config['capital']:,.0f}")
    print(f"  Risk per trade: {config['risk_pct']}%")

    confirm = get_user_choice("\nProceed with generation? (y/n)")

    if confirm.lower() == 'y':
        print(f"\n{Colors.GREEN}Generating your trading bot...{Colors.ENDC}\n")
        generate_trading_bot(config)
    else:
        print("Generation cancelled. Run wizard again to reconfigure.")


def generate_trading_bot(config: Dict[str, Any]):
    """
    Generate complete trading bot based on configuration

    Creates:
        trading_bot/
        ├── config.json
        ├── main.py
        ├── signal_generator.py
        ├── data_manager.py
        ├── risk_manager.py
        ├── zerodha_client.py
        └── universe/
            └── {universe}.json
    """
    # Create folder
    folder_name = "trading_bot_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    bot_path = Path(folder_name)
    bot_path.mkdir(exist_ok=True)

    print(f"✓ Created folder: {folder_name}")

    # Generate config.json
    config_data = {
        'trading_style': config['trading_style'],
        'universe': config['universe'],
        'strategy': config['strategy'],
        'capital': config['capital'],
        'risk_pct': config['risk_pct'],
        'parameters': get_default_parameters(config['strategy'], config['trading_style'])
    }

    with open(bot_path / 'config.json', 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"✓ Generated config.json")

    # Fetch universe if needed
    if config['universe'] != 'custom':
        print(f"\n⏳ Fetching {config['universe']} constituents from NSE...")
        universe_data = fetch_index_constituents(config['universe'])

        universe_path = bot_path / 'universe'
        universe_path.mkdir(exist_ok=True)

        with open(universe_path / f"{config['universe']}.json", 'w') as f:
            json.dump(universe_data, f, indent=2)

        print(f"✓ Saved universe: {len(universe_data['stocks'])} stocks")

    # Generate code files
    print(f"\n⏳ Generating code files...")

    # Use templates based on configuration
    template_path = Path(__file__).parent / 'templates'

    if config['trading_style'] == 'intraday':
        template_file = template_path / 'minimal_intraday.py'
    else:
        template_file = template_path / 'minimal_positional.py'

    if template_file.exists():
        # Copy and customize template
        template_content = template_file.read_text()
        main_py = bot_path / 'main.py'
        main_py.write_text(template_content)
        print(f"✓ Generated main.py (from {template_file.name})")
    else:
        print(f"⚠️  Template not found, generating basic structure...")
        generate_basic_structure(bot_path, config)

    # Success message
    print(f"\n{Colors.GREEN}✅ Trading bot generated successfully!{Colors.ENDC}\n")
    print(f"Location: {bot_path.absolute()}")
    print(f"\nNext steps:")
    print(f"  1. cd {folder_name}")
    print(f"  2. Configure Zerodha API credentials in .env")
    print(f"  3. python main.py --mode paper (test with paper trading)")
    print(f"  4. python main.py --mode live (go live after testing)")
    print(f"\nRecommended: Read NUANCES.md for critical production gotchas")


def get_default_parameters(strategy: str, trading_style: str) -> Dict[str, Any]:
    """
    Get default parameters based on strategy and trading style
    These are battle-tested values from KNOWLEDGE.md
    """
    # Base parameters (common across strategies)
    params = {
        'rsi_period': 14,
        'rsi_long_min': 45,
        'rsi_long_max': 65,
        'adx_min': 25,
        'ema_fast': 9,
        'ema_slow': 21,
        'volume_mult': 1.5,
        'symbol_cooldown_minutes': 45
    }

    # Strategy-specific
    if strategy == 'fortress':
        params['min_confidence'] = 0.50
        params['factors'] = ['trend', 'strength', 'momentum', 'vwap', 'volume', 'body']
    elif strategy == 'momentum':
        params['breakout_period'] = 20
        params['volume_mult'] = 2.0
    elif strategy == 'mean_reversion':
        params['vwap_band_std'] = 2.0
        params['rsi_long_min'] = 30  # Buy oversold
        params['rsi_long_max'] = 50

    # Trading style specific
    if trading_style == 'intraday':
        params['target_pct'] = 1.0
        params['stop_loss_pct'] = 0.5
        params['max_hold_minutes'] = 45
    elif trading_style == 'swing':
        params['target_pct'] = 5.0
        params['stop_loss_pct'] = 2.5
        params['max_hold_days'] = 7
    elif trading_style == 'positional':
        params['target_pct'] = 8.0
        params['stop_loss_pct'] = 4.0
        params['max_hold_days'] = 30

    return params


def generate_basic_structure(bot_path: Path, config: Dict[str, Any]):
    """
    Generate basic file structure if templates not found
    """
    # main.py stub
    main_content = f'''#!/usr/bin/env python3
"""
Trading Bot - {config['strategy']} strategy
Auto-generated by AlgoTrader

Trading Style: {config['trading_style']}
Universe: {config['universe']}
Capital: ₹{config['capital']:,.0f}
Risk per trade: {config['risk_pct']}%
"""

# TODO: Import required modules
# TODO: Load configuration
# TODO: Initialize Zerodha client
# TODO: Main trading loop

if __name__ == "__main__":
    print("Trading bot starting...")
    print("⚠️  Complete the implementation using examples from algotrader/examples/")
'''

    (bot_path / 'main.py').write_text(main_content)
    print(f"✓ Generated main.py (basic structure)")

    # .env template
    env_content = '''# Zerodha API Credentials
# Get from: https://kite.trade/
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_ACCESS_TOKEN=your_access_token_here

# Never commit this file to Git!
'''
    (bot_path / '.env.template').write_text(env_content)
    print(f"✓ Generated .env.template")


# ============================================================================
# SECTION: Universe Fetcher
# PURPOSE: Fetch latest index constituents from NSE
# ONLINE: Always uses fresh data from internet
# RETURNS: JSON with symbols, market cap, sector, liquidity
# CRITICAL: This validates stale information by fetching live data
# ============================================================================

def fetch_index_constituents(index_name: str) -> Dict[str, Any]:
    """
    Fetch index constituents from NSE website

    Args:
        index_name: 'nifty50', 'nifty100', 'midcap150', 'smallcap250'

    Returns:
        {
            'index': 'NIFTY 50',
            'last_updated': '2026-02-14T14:35:00',
            'stocks': [
                {
                    'symbol': 'RELIANCE',
                    'company': 'Reliance Industries Ltd',
                    'sector': 'Energy',
                    'mcap': 1500000,  # Crores
                    'avg_volume': 5000000,
                    'isin': 'INE002A01018'
                },
                ...
            ]
        }
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print(f"{Colors.RED}Error: Missing dependencies (requests, beautifulsoup4){Colors.ENDC}")
        print("Install: pip install requests beautifulsoup4")
        return {'index': index_name, 'stocks': [], 'error': 'Missing dependencies'}

    # Map index names to NSE index codes
    index_map = {
        'nifty50': 'NIFTY 50',
        'nifty100': 'NIFTY 100',
        'midcap150': 'NIFTY MIDCAP 150',
        'smallcap250': 'NIFTY SMALLCAP 250'
    }

    nse_index = index_map.get(index_name, 'NIFTY 50')

    print(f"⏳ Fetching {nse_index} constituents from NSE...")

    # NSE website URL (as of Feb 2026)
    # NOTE: This URL may change. If broken, update from latest NSE website
    url = f"https://www.nseindia.com/api/equity-stockIndices?index={nse_index.replace(' ', '%20')}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    try:
        # Create session to handle cookies
        session = requests.Session()

        # First visit homepage to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(1)

        # Now fetch index data
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Parse data
        stocks = []
        if 'data' in data:
            for item in data['data']:
                symbol = item.get('symbol', '').replace('-EQ', '')

                # Skip index itself
                if symbol == nse_index:
                    continue

                stocks.append({
                    'symbol': symbol,
                    'company': item.get('meta', {}).get('companyName', symbol),
                    'sector': item.get('meta', {}).get('industry', 'Unknown'),
                    'last_price': item.get('lastPrice', 0),
                    'change_pct': item.get('pChange', 0),
                    'isin': item.get('meta', {}).get('isin', '')
                })

        print(f"{Colors.GREEN}✓ Fetched {len(stocks)} stocks{Colors.ENDC}")

        return {
            'index': nse_index,
            'last_updated': datetime.now().isoformat(),
            'stocks': stocks,
            'count': len(stocks)
        }

    except requests.exceptions.RequestException as e:
        print(f"{Colors.RED}⚠️  Error fetching from NSE: {e}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Falling back to hardcoded list...{Colors.ENDC}")
        return get_fallback_universe(index_name)
    except Exception as e:
        print(f"{Colors.RED}⚠️  Unexpected error: {e}{Colors.ENDC}")
        return get_fallback_universe(index_name)


def get_fallback_universe(index_name: str) -> Dict[str, Any]:
    """
    Fallback to hardcoded universe if NSE fetch fails
    This list is from Jan 2026, may be slightly outdated
    """
    # Hardcoded Nifty 50 (as of Jan 2026)
    nifty50 = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "LT", "BAJFINANCE", "ASIANPAINT", "MARUTI", "HCLTECH",
        "AXISBANK", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
        "TATAMOTORS", "NTPC", "POWERGRID", "ONGC", "ADANIPORTS",
        "WIPRO", "M&M", "COALINDIA", "JSWSTEEL", "TATASTEEL",
        "BAJAJFINSV", "HINDALCO", "INDUSINDBK", "TECHM", "GRASIM",
        "CIPLA", "DRREDDY", "EICHERMOT", "HEROMOTOCO", "BRITANNIA",
        "UPL", "DIVISLAB", "ADANIENT", "APOLLOHOSP", "SHRIRAMFIN",
        "BAJAJ-AUTO", "HDFCLIFE", "SBILIFE", "BPCL", "TATACONSUM"
    ]

    stocks = [{'symbol': s, 'company': s, 'sector': 'Unknown'} for s in nifty50]

    return {
        'index': 'NIFTY 50 (Fallback)',
        'last_updated': datetime.now().isoformat(),
        'stocks': stocks,
        'count': len(stocks),
        'warning': 'Using fallback list. NSE fetch failed.'
    }


def wizard_enhance_existing(files: Dict[str, str]):
    """
    Enhance existing trading code
    """
    print_header("ENHANCE EXISTING CODE")
    print("This feature will be implemented in next version.")
    print("For now, use '/algotrader check <file>' to analyze code.")


def wizard_analyze_code(files: Dict[str, str]):
    """
    Analyze existing code for issues
    """
    print_header("CODE ANALYSIS")
    print("This feature will be implemented in next version.")
    print("For now, manually review against NUANCES.md checklist.")


# ============================================================================
# SECTION: Main Entry Point
# PURPOSE: Route commands to appropriate functions
# ============================================================================

def main():
    """
    Main entry point for algotrader CLI
    """
    # Check dependencies first
    if not check_dependencies():
        print(f"\n{Colors.YELLOW}Install dependencies and try again.{Colors.ENDC}")
        return

    # Parse command line arguments
    if len(sys.argv) == 1:
        # No arguments → interactive wizard
        wizard_main()
    else:
        command = sys.argv[1]

        if command == 'wizard':
            wizard_main()

        elif command == 'universe':
            # Fetch universe
            print_header("UNIVERSE FETCHER")

            # Determine which indices to fetch
            if len(sys.argv) > 2 and sys.argv[2] == '--indices':
                indices = sys.argv[3].split(',') if len(sys.argv) > 3 else ['nifty50']
            else:
                # Default: all indices
                indices = ['nifty50', 'nifty100', 'midcap150']

            universe_path = Path('universe')
            universe_path.mkdir(exist_ok=True)

            for index in indices:
                data = fetch_index_constituents(index.strip())

                # Save to JSON
                output_file = universe_path / f"{index}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"✓ Saved: {output_file} ({data['count']} stocks)")

            print(f"\n{Colors.GREEN}✅ Universe files created in ./universe/{Colors.ENDC}")

        elif command == 'signal':
            # Generate signal for symbol
            if len(sys.argv) < 3:
                print(f"{Colors.RED}Usage: /algotrader signal <SYMBOL>{Colors.ENDC}")
                return

            symbol = sys.argv[2].upper()
            print_header(f"SIGNAL ANALYSIS - {symbol}")
            print("This feature requires live market data.")
            print("Implement signal generation using templates in examples/")

        elif command == 'check':
            # Analyze code
            if len(sys.argv) < 3:
                print(f"{Colors.RED}Usage: /algotrader check <file.py>{Colors.ENDC}")
                return

            filepath = sys.argv[2]
            print_header(f"CODE ANALYSIS - {filepath}")
            print("This feature will be implemented in next version.")
            print("For now, manually review against NUANCES.md checklist.")

        elif command == 'optimize':
            # Optimize code
            if len(sys.argv) < 3:
                print(f"{Colors.RED}Usage: /algotrader optimize <file.py>{Colors.ENDC}")
                return

            filepath = sys.argv[2]
            print_header(f"CODE OPTIMIZATION - {filepath}")
            print("This feature will be implemented in next version.")

        elif command in ['help', '--help', '-h']:
            print_usage()

        else:
            print(f"{Colors.RED}Unknown command: {command}{Colors.ENDC}")
            print_usage()


def print_usage():
    """Print usage information"""
    print(f"\n{Colors.BOLD}AlgoTrader - Quantitative Trading Expert{Colors.ENDC}\n")
    print("Usage:")
    print("  /algotrader                       Interactive wizard")
    print("  /algotrader wizard                Bot generation wizard")
    print("  /algotrader universe              Fetch all index constituents")
    print("  /algotrader universe --indices nifty50,midcap150")
    print("  /algotrader signal <SYMBOL>       Generate signal for symbol")
    print("  /algotrader check <file.py>       Analyze code for issues")
    print("  /algotrader optimize <file.py>    Optimize code performance")
    print("  /algotrader help                  Show this help")
    print("\nDocumentation:")
    print("  README.md     - Comprehensive guide")
    print("  KNOWLEDGE.md  - All 16 domains (1,780 lines)")
    print("  NUANCES.md    - 30+ token-burning gotchas")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Operation cancelled by user{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
