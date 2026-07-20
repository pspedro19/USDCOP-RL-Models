#!/bin/bash
# Start script for AlgoTrader - Ensures venv is active and runs with proper environment
# Usage: ./start.sh [bot_directory] [--mode paper|live]

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
VENV_PIP="$SCRIPT_DIR/venv/bin/pip"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print header
print_header() {
    echo ""
    print_msg "$CYAN" "╔══════════════════════════════════════════════════════════════════╗"
    print_msg "$CYAN" "║                  ALGOTRADER STARTUP SYSTEM                       ║"
    print_msg "$CYAN" "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
}

# Check if venv exists
check_venv() {
    if [ ! -f "$VENV_PYTHON" ]; then
        print_msg "$YELLOW" "⚠️  Virtual environment not found at: $SCRIPT_DIR/venv"
        print_msg "$BLUE" "Creating virtual environment..."
        python3 -m venv "$SCRIPT_DIR/venv"

        print_msg "$BLUE" "Installing dependencies..."
        "$VENV_PIP" install -q -r "$SCRIPT_DIR/requirements.txt"

        print_msg "$GREEN" "✅ Virtual environment created and dependencies installed"
    else
        print_msg "$GREEN" "✅ Virtual environment found"
    fi
}

# Verify dependencies
verify_dependencies() {
    print_msg "$BLUE" "Checking dependencies..."

    local required_packages=("polars" "kiteconnect" "requests" "beautifulsoup4")
    local missing=()

    for package in "${required_packages[@]}"; do
        if ! "$VENV_PYTHON" -c "import $package" 2>/dev/null; then
            missing+=("$package")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        print_msg "$YELLOW" "⚠️  Missing packages: ${missing[*]}"
        print_msg "$BLUE" "Installing missing dependencies..."
        "$VENV_PIP" install -q -r "$SCRIPT_DIR/requirements.txt"
        print_msg "$GREEN" "✅ Dependencies installed"
    else
        print_msg "$GREEN" "✅ All dependencies present"
    fi
}

# Show Python and package versions
show_versions() {
    print_msg "$BLUE" "Environment info:"
    "$VENV_PYTHON" --version | sed 's/^/   /'
    "$VENV_PIP" list 2>/dev/null | grep -E "polars|kiteconnect" | sed 's/^/   /'
}

# Activate venv and run algotrader CLI
run_algotrader() {
    print_header
    check_venv
    verify_dependencies
    show_versions
    echo ""

    print_msg "$GREEN" "🚀 Starting AlgoTrader CLI..."
    echo ""

    # Run algotrader with all arguments
    "$VENV_PYTHON" "$SCRIPT_DIR/algotrader.py" "$@"
}

# Start a generated trading bot
start_bot() {
    local bot_dir=$1
    local mode=${2:-paper}  # Default to paper trading

    print_header

    if [ ! -d "$bot_dir" ]; then
        print_msg "$RED" "❌ Bot directory not found: $bot_dir"
        echo ""
        print_msg "$YELLOW" "Usage: ./start.sh <bot_directory> [--mode paper|live]"
        print_msg "$YELLOW" "Example: ./start.sh trading_bot_20260214_143000 --mode paper"
        echo ""
        exit 1
    fi

    # Check if bot has main.py
    if [ ! -f "$bot_dir/main.py" ]; then
        print_msg "$RED" "❌ No main.py found in $bot_dir"
        exit 1
    fi

    check_venv
    verify_dependencies

    # Check if .env exists
    if [ ! -f "$bot_dir/.env" ] && [ ! -f "$SCRIPT_DIR/.env" ]; then
        print_msg "$YELLOW" "⚠️  No .env file found"
        print_msg "$YELLOW" "    Create $bot_dir/.env with your Zerodha API credentials"
        echo ""
        print_msg "$BLUE" "Template .env file:"
        echo "    KITE_API_KEY=your_api_key"
        echo "    KITE_API_SECRET=your_api_secret"
        echo "    KITE_ACCESS_TOKEN=your_access_token"
        echo ""

        if [ "$mode" == "live" ]; then
            print_msg "$RED" "❌ Cannot start in LIVE mode without API credentials"
            exit 1
        fi
    fi

    show_versions
    echo ""

    print_msg "$GREEN" "🚀 Starting trading bot: $bot_dir"
    print_msg "$BLUE" "Mode: $mode"
    echo ""

    # Change to bot directory and run
    cd "$bot_dir"
    "$VENV_PYTHON" main.py --mode "$mode"
}

# Main logic
main() {
    # If no arguments, show usage
    if [ $# -eq 0 ]; then
        print_header
        print_msg "$YELLOW" "AlgoTrader Startup System"
        echo ""
        echo "Usage:"
        echo "  ./start.sh                           Run AlgoTrader CLI (interactive)"
        echo "  ./start.sh <bot_dir>                 Start trading bot in paper mode"
        echo "  ./start.sh <bot_dir> --mode live     Start trading bot in live mode"
        echo "  ./start.sh wizard                    Run bot generation wizard"
        echo "  ./start.sh universe                  Fetch universe data"
        echo "  ./start.sh help                      Show help"
        echo ""
        echo "Examples:"
        echo "  ./start.sh wizard"
        echo "  ./start.sh universe --indices nifty50"
        echo "  ./start.sh trading_bot_20260214_143000"
        echo "  ./start.sh trading_bot_20260214_143000 --mode live"
        echo ""
        exit 0
    fi

    # List of known AlgoTrader commands (not bot directories)
    local known_commands=("wizard" "universe" "signal" "check" "optimize" "help" "--help" "-h")

    # Check if first argument is a known command
    local is_command=false
    for cmd in "${known_commands[@]}"; do
        if [ "$1" == "$cmd" ]; then
            is_command=true
            break
        fi
    done

    # If it's a known command, run algotrader CLI
    if [ "$is_command" == "true" ]; then
        run_algotrader "$@"
    # Otherwise, check if it's a directory (trading bot)
    elif [ -d "$1" ]; then
        # Extract mode if provided
        local mode="paper"
        if [ "$2" == "--mode" ] && [ -n "$3" ]; then
            mode=$3
        fi
        start_bot "$1" "$mode"
    else
        # Unknown command or non-existent directory
        print_msg "$RED" "❌ Unknown command or directory not found: $1"
        echo ""
        print_msg "$YELLOW" "Run ./start.sh for usage information"
        exit 1
    fi
}

# Trap Ctrl+C for graceful shutdown
trap 'echo ""; print_msg "$YELLOW" "🛑 Shutting down..."; exit 0' INT

# Run main function
main "$@"
