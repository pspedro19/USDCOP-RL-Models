# start.sh - Comprehensive Startup Script Guide

## Overview

`start.sh` is an intelligent startup script that ensures AlgoTrader runs with the correct virtual environment and all dependencies. It handles both:
1. **AlgoTrader CLI commands** (wizard, universe, check, etc.)
2. **Trading bot execution** (paper mode or live mode)

## Features

### ✅ Automatic Environment Management
- Checks if virtual environment exists
- Creates venv if missing
- Verifies and installs dependencies automatically
- Shows version information

### ✅ Dual Mode Operation
- **CLI Mode**: Run AlgoTrader commands (wizard, universe, etc.)
- **Bot Mode**: Start generated trading bots with mode selection

### ✅ Safety Features
- Requires .env file for live trading
- Defaults to paper mode for safety
- Graceful shutdown on Ctrl+C
- Colorful, informative output

### ✅ Dependency Validation
- Checks for required packages (polars, kiteconnect, requests, beautifulsoup4)
- Auto-installs missing packages
- Shows installed versions

## Usage

### 1. Show Help (No Arguments)
```bash
./start.sh
```
**Output:**
```
╔══════════════════════════════════════════════════════════════════╗
║                  ALGOTRADER STARTUP SYSTEM                       ║
╚══════════════════════════════════════════════════════════════════╝

AlgoTrader Startup System

Usage:
  ./start.sh                           Run AlgoTrader CLI (interactive)
  ./start.sh <bot_dir>                 Start trading bot in paper mode
  ./start.sh <bot_dir> --mode live     Start trading bot in live mode
  ./start.sh wizard                    Run bot generation wizard
  ./start.sh universe                  Fetch universe data
  ./start.sh help                      Show help
```

### 2. Run AlgoTrader Commands

#### Generate Trading Bot
```bash
./start.sh wizard
```

#### Fetch Universe Data
```bash
# Fetch all default indices
./start.sh universe

# Fetch specific indices
./start.sh universe --indices nifty50,midcap150,smallcap250
```

#### Get Help
```bash
./start.sh help
```

#### Check Code for Issues
```bash
./start.sh check ./my_bot.py
```

### 3. Start Trading Bots

#### Paper Trading (Safe, Default)
```bash
./start.sh trading_bot_20260214_143000
```

**What happens:**
1. ✅ Checks virtual environment
2. ✅ Verifies dependencies
3. ⚠️ Warns if .env is missing (non-blocking for paper mode)
4. ✅ Shows environment info
5. 🚀 Starts bot in paper mode

#### Live Trading (Requires .env)
```bash
./start.sh trading_bot_20260214_143000 --mode live
```

**What happens:**
1. ✅ Checks virtual environment
2. ✅ Verifies dependencies
3. ❌ **BLOCKS** if .env file is missing
4. ✅ Shows environment info
5. 🚀 Starts bot in live mode

## Environment Setup

### Virtual Environment
The script automatically creates a virtual environment if it doesn't exist:

```bash
# Automatic on first run
./start.sh wizard
# Creates: venv/
# Installs: all requirements.txt packages
```

### Dependency Installation
Missing packages are detected and installed automatically:

```
Checking dependencies...
⚠️  Missing packages: beautifulsoup4
Installing missing dependencies...
✅ Dependencies installed
```

### API Credentials (.env)

For **live trading**, create `.env` file in your bot directory:

```bash
# In your trading bot directory
cat > .env << EOF
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_ACCESS_TOKEN=your_access_token_here
EOF
```

**Security Note:** Never commit .env to git (already in .gitignore)

## Examples

### Example 1: First-Time Setup
```bash
cd /home/rakesh/work/skills/algotrader

# First run - creates venv and installs dependencies
./start.sh wizard

# Follow prompts to generate bot
# Output: trading_bot_20260214_143000/
```

### Example 2: Fetch Latest Market Data
```bash
# Fetch Nifty 50 and Midcap 150
./start.sh universe --indices nifty50,midcap150

# Output:
# ✓ Fetched 50 stocks
# ✓ Saved: universe/nifty50.json
# ✓ Fetched 150 stocks
# ✓ Saved: universe/midcap150.json
```

### Example 3: Test Bot in Paper Mode
```bash
# Generate bot first
./start.sh wizard
# ... follow prompts ...

# Start in paper mode (safe, no real trades)
./start.sh trading_bot_20260214_143000

# Output:
# 🚀 Starting trading bot: trading_bot_20260214_143000
# Mode: paper
# [Bot runs...]
```

### Example 4: Go Live (After Testing)
```bash
# Configure API credentials
cd trading_bot_20260214_143000
cat > .env << EOF
KITE_API_KEY=your_real_key
KITE_API_SECRET=your_real_secret
KITE_ACCESS_TOKEN=your_real_token
EOF
cd ..

# Start in live mode
./start.sh trading_bot_20260214_143000 --mode live

# Output:
# ✅ API credentials found
# 🚀 Starting trading bot: trading_bot_20260214_143000
# Mode: live
# [Bot runs with real money!]
```

## Output Explained

### Successful Startup
```
╔══════════════════════════════════════════════════════════════════╗
║                  ALGOTRADER STARTUP SYSTEM                       ║
╚══════════════════════════════════════════════════════════════════╝

✅ Virtual environment found          # venv exists
✅ All dependencies present            # All packages installed
Environment info:                      # Version information
   Python 3.12.3
   kiteconnect        5.0.1
   polars             1.38.1

🚀 Starting trading bot: trading_bot_20260214_143000
Mode: paper
```

### Missing Dependencies
```
✅ Virtual environment found
⚠️  Missing packages: beautifulsoup4  # Auto-detected
Installing missing dependencies...     # Auto-fixed
✅ Dependencies installed
```

### Missing .env (Paper Mode)
```
⚠️  No .env file found                          # Warning only
    Create test_bot/.env with credentials
Template .env file:
    KITE_API_KEY=your_api_key
    KITE_API_SECRET=your_api_secret
    KITE_ACCESS_TOKEN=your_access_token
```

### Missing .env (Live Mode)
```
⚠️  No .env file found
❌ Cannot start in LIVE mode without API credentials  # BLOCKED
```

## Comparison: start.sh vs run.sh

| Feature | `start.sh` | `run.sh` |
|---------|-----------|----------|
| **Purpose** | Full startup system | Simple command wrapper |
| **Venv Check** | ✅ With auto-create | ✅ With auto-create |
| **Dependency Check** | ✅ Auto-install missing | ❌ No check |
| **Bot Execution** | ✅ With mode selection | ❌ CLI only |
| **Environment Info** | ✅ Shows versions | ❌ No |
| **Safety Checks** | ✅ .env validation | ❌ No |
| **Colorful Output** | ✅ Yes | ❌ No |
| **Best For** | Production use | Quick CLI commands |

**Recommendation:**
- Use `start.sh` for running trading bots
- Use `run.sh` for quick CLI commands

## Troubleshooting

### Issue: "Virtual environment not found"
**Fix:** Script auto-creates it
```bash
./start.sh wizard
# Automatically creates venv and installs dependencies
```

### Issue: "Missing packages: beautifulsoup4"
**Fix:** Script auto-installs it
```
Checking dependencies...
⚠️  Missing packages: beautifulsoup4
Installing missing dependencies...
✅ Dependencies installed
```

### Issue: "Cannot start in LIVE mode without API credentials"
**Fix:** Create .env file
```bash
cd your_bot_directory
cat > .env << EOF
KITE_API_KEY=your_key
KITE_API_SECRET=your_secret
KITE_ACCESS_TOKEN=your_token
EOF
```

### Issue: "No main.py found in bot_directory"
**Fix:** Ensure you're pointing to a valid bot directory
```bash
# Check if directory has main.py
ls bot_directory/main.py

# If not, regenerate bot
./start.sh wizard
```

## Advanced Usage

### Custom Python Version
Edit `start.sh` and change:
```bash
python3 -m venv "$SCRIPT_DIR/venv"
# to
python3.11 -m venv "$SCRIPT_DIR/venv"  # Use specific version
```

### Skip Dependency Check (Not Recommended)
Comment out the verification:
```bash
# verify_dependencies  # Skip this line
```

### Force Dependency Reinstall
```bash
rm -rf venv/
./start.sh wizard  # Recreates venv and reinstalls all
```

## Integration with Systemd

Create a systemd service for auto-start:

```bash
# Create service file
sudo nano /etc/systemd/system/algotrader.service

# Add:
[Unit]
Description=AlgoTrader Bot
After=network.target

[Service]
Type=simple
User=rakesh
WorkingDirectory=/home/rakesh/work/skills/algotrader
ExecStart=/home/rakesh/work/skills/algotrader/start.sh trading_bot_20260214_143000 --mode paper
Restart=on-failure

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable algotrader
sudo systemctl start algotrader
```

## Security Best Practices

1. **Never commit .env** - Already in .gitignore
2. **Test with paper mode first** - Default behavior
3. **Review bot code** - Before going live
4. **Start small capital** - When testing live
5. **Monitor logs** - Check for errors
6. **Use stop loss** - Always

## Summary

`start.sh` is your all-in-one startup solution:

✅ **Intelligent** - Auto-detects and fixes environment issues
✅ **Safe** - Defaults to paper mode, validates .env for live
✅ **Versatile** - Handles both CLI and bot execution
✅ **Informative** - Colorful output with version info
✅ **Production-ready** - Suitable for systemd integration

**Quick Start:**
```bash
./start.sh wizard          # Generate bot
./start.sh universe        # Fetch data
./start.sh my_bot          # Run in paper mode
./start.sh my_bot --mode live  # Go live (with .env)
```

---

**Last Updated:** 2026-02-14
**Script Version:** 1.0
