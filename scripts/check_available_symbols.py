"""
Check Available Symbols on MT5
==============================
Find what forex symbols are available on the broker.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_symbols():
    """Check available symbols on MT5"""
    try:
        from dotenv import load_dotenv
        import MetaTrader5 as mt5
        
        # Load environment
        load_dotenv(project_root / '.env')
        
        # Connect
        if not mt5.initialize():
            print("Could not initialize MT5")
            return
        
        login = int(os.getenv('MT5_LOGIN'))
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')
        
        if not mt5.login(login, password=password, server=server):
            print("Could not login")
            return
        
        print("Searching for forex symbols...")
        
        # Get all symbols
        symbols = mt5.symbols_get()
        if not symbols:
            print("No symbols found")
            return
        
        print(f"Found {len(symbols)} total symbols")
        
        # Filter forex symbols
        forex_symbols = []
        usd_symbols = []
        
        for symbol in symbols:
            symbol_name = symbol.name
            
            # Look for forex pairs (typically 6 characters, no spaces/dots)
            if len(symbol_name) == 6 and symbol_name.isalpha():
                forex_symbols.append(symbol_name)
                
            # Look for USD pairs specifically
            if 'USD' in symbol_name.upper():
                usd_symbols.append(symbol_name)
        
        print(f"\nFound {len(forex_symbols)} forex pairs:")
        for symbol in sorted(forex_symbols)[:20]:  # Show first 20
            print(f"   {symbol}")
        if len(forex_symbols) > 20:
            print(f"   ... and {len(forex_symbols) - 20} more")
        
        print(f"\nFound {len(usd_symbols)} USD-related symbols:")
        for symbol in sorted(usd_symbols):
            print(f"   {symbol}")
        
        # Look specifically for COP or Colombian peso related
        cop_symbols = [s.name for s in symbols if 'COP' in s.name.upper()]
        if cop_symbols:
            print(f"\nFound COP symbols:")
            for symbol in cop_symbols:
                info = mt5.symbol_info(symbol)
                print(f"   {symbol} - {info.description if info else 'N/A'}")
        else:
            print("\nNo COP (Colombian Peso) symbols found")
            print("This broker may not offer USDCOP trading")
            
            # Suggest alternatives
            print("\nSuggested alternatives for testing:")
            test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
            for test_symbol in test_symbols:
                info = mt5.symbol_info(test_symbol)
                if info:
                    print(f"   [OK] {test_symbol} - {info.description}")
                else:
                    print(f"   [NO] {test_symbol} - Not available")
        
        mt5.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_symbols()