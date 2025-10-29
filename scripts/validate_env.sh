#!/bin/bash

# ==============================================
# USDCOP Trading System - Environment Validation Script
# ==============================================
# This script validates that:
# 1. .env file exists
# 2. All required variables are set
# 3. No weak/default passwords are used
# 4. API keys are not placeholders
# 5. File permissions are secure
#
# Usage: ./scripts/validate_env.sh
# Exit codes:
#   0 = All validations passed
#   1 = Critical validation failures found
# ==============================================

set -e

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
ERRORS=0
WARNINGS=0
PASSED=0

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}USDCOP Trading System - Environment Validation${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# ==============================================
# VALIDATION 1: Check if .env file exists
# ==============================================
echo -e "${BLUE}[1/7] Checking if .env file exists...${NC}"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}✗ CRITICAL: .env file not found at $ENV_FILE${NC}"
    echo -e "${YELLOW}  → Create it by copying .env.example: cp .env.example .env${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ .env file exists${NC}"
    PASSED=$((PASSED + 1))
fi
echo ""

# ==============================================
# VALIDATION 2: Check file permissions
# ==============================================
echo -e "${BLUE}[2/7] Checking .env file permissions...${NC}"
if [ -f "$ENV_FILE" ]; then
    PERMS=$(stat -c "%a" "$ENV_FILE" 2>/dev/null || stat -f "%A" "$ENV_FILE" 2>/dev/null || echo "unknown")
    if [ "$PERMS" = "600" ] || [ "$PERMS" = "400" ]; then
        echo -e "${GREEN}✓ File permissions are secure ($PERMS)${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${YELLOW}⚠ WARNING: File permissions are $PERMS (should be 600)${NC}"
        echo -e "${YELLOW}  → Fix with: chmod 600 .env${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}⚠ Skipping (no .env file)${NC}"
fi
echo ""

# ==============================================
# VALIDATION 3: Check required variables
# ==============================================
echo -e "${BLUE}[3/7] Checking required environment variables...${NC}"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"

    REQUIRED_VARS=(
        "POSTGRES_USER"
        "POSTGRES_PASSWORD"
        "POSTGRES_DB"
        "REDIS_PASSWORD"
        "MINIO_ACCESS_KEY"
        "MINIO_SECRET_KEY"
        "AIRFLOW_USER"
        "AIRFLOW_PASSWORD"
        "AIRFLOW_FERNET_KEY"
        "AIRFLOW_SECRET_KEY"
        "GRAFANA_USER"
        "GRAFANA_PASSWORD"
        "PGADMIN_EMAIL"
        "PGADMIN_PASSWORD"
        "TWELVEDATA_API_KEY_1"
        "API_KEY_G1_1"
        "API_KEY_G1_2"
        "API_KEY_G1_3"
        "API_KEY_G1_4"
        "API_KEY_G1_5"
        "API_KEY_G1_6"
        "API_KEY_G1_7"
        "API_KEY_G1_8"
        "API_KEY_G2_1"
        "API_KEY_G2_2"
        "API_KEY_G2_3"
        "API_KEY_G2_4"
        "API_KEY_G2_5"
        "API_KEY_G2_6"
        "API_KEY_G2_7"
        "API_KEY_G2_8"
    )

    MISSING_VARS=0
    for VAR in "${REQUIRED_VARS[@]}"; do
        if [ -z "${!VAR}" ]; then
            echo -e "${RED}✗ Missing required variable: $VAR${NC}"
            ERRORS=$((ERRORS + 1))
            MISSING_VARS=$((MISSING_VARS + 1))
        fi
    done

    if [ $MISSING_VARS -eq 0 ]; then
        echo -e "${GREEN}✓ All ${#REQUIRED_VARS[@]} required variables are set${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ $MISSING_VARS required variable(s) missing${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping (no .env file)${NC}"
fi
echo ""

# ==============================================
# VALIDATION 4: Check for weak passwords
# ==============================================
echo -e "${BLUE}[4/7] Checking for weak/default passwords...${NC}"
if [ -f "$ENV_FILE" ]; then
    WEAK_PASSWORDS=(
        "admin123"
        "password"
        "admin"
        "redis123"
        "minioadmin123"
        "your-secret-key"
        "CHANGE_ME"
    )

    WEAK_FOUND=0
    # Only check actual password fields, not emails or usernames
    CRITICAL_PASSWORD_VARS=(
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "MINIO_SECRET_KEY"
        "AIRFLOW_PASSWORD"
        "GRAFANA_PASSWORD"
    )

    for VAR in "${CRITICAL_PASSWORD_VARS[@]}"; do
        VAR_VALUE=$(grep "^${VAR}=" "$ENV_FILE" | cut -d'=' -f2)
        for WEAK in "${WEAK_PASSWORDS[@]}"; do
            if [ "$VAR_VALUE" == "$WEAK" ]; then
                echo -e "${RED}✗ CRITICAL: $VAR has weak password: '$WEAK'${NC}"
                echo -e "${YELLOW}  → Generate strong password: openssl rand -base64 32${NC}"
                ERRORS=$((ERRORS + 1))
                WEAK_FOUND=$((WEAK_FOUND + 1))
            fi
        done
    done

    if [ $WEAK_FOUND -eq 0 ]; then
        echo -e "${GREEN}✓ No weak passwords detected in critical fields${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ $WEAK_FOUND weak password(s) found - MUST be changed!${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping (no .env file)${NC}"
fi
echo ""

# ==============================================
# VALIDATION 5: Check for placeholder API keys (CRITICAL VARIABLES ONLY)
# ==============================================
echo -e "${BLUE}[5/7] Checking for placeholder API keys...${NC}"
if [ -f "$ENV_FILE" ]; then
    # Only check CRITICAL variables (TwelveData API keys)
    # Optional variables (MT5, JWT, Slack) can have placeholders
    CRITICAL_VARS=(
        "TWELVEDATA_API_KEY_1"
        "API_KEY_G1_1"
        "API_KEY_G2_1"
    )

    PLACEHOLDER_FOUND=0
    for VAR in "${CRITICAL_VARS[@]}"; do
        VAR_VALUE=$(grep "^${VAR}=" "$ENV_FILE" | cut -d'=' -f2)
        if [[ "$VAR_VALUE" == *"YOUR_"* ]] || [[ "$VAR_VALUE" == *"PLACEHOLDER"* ]]; then
            echo -e "${RED}✗ CRITICAL: $VAR has placeholder value${NC}"
            echo -e "${YELLOW}  → Replace with real TwelveData API key from https://twelvedata.com/apikey${NC}"
            ERRORS=$((ERRORS + 1))
            PLACEHOLDER_FOUND=$((PLACEHOLDER_FOUND + 1))
        fi
    done

    if [ $PLACEHOLDER_FOUND -eq 0 ]; then
        echo -e "${GREEN}✓ All critical API keys are configured${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ $PLACEHOLDER_FOUND critical placeholder(s) found - system will NOT work!${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping (no .env file)${NC}"
fi
echo ""

# ==============================================
# VALIDATION 6: Check for exposed API keys
# ==============================================
echo -e "${BLUE}[6/7] Checking for previously exposed API keys...${NC}"
if [ -f "$ENV_FILE" ]; then
    EXPOSED_KEYS=(
        "7bb5ad48501b4a6ea9fbedba2b0247f9"
        "21de7ed341dc4ad5a20c292cee4652cb"
        "06f4622c4cfd4fd78795777367bb07d6"
        "0cabdba116f2415c82f7922c704427d9"
        "93104476650b4d25aaf033ce06585ba2"
        "9c61d83999c949d9866d92661f0f7c25"
        "3d465022bf0c489f84270d5f1cd771b9"
        "46d5e594b4af4282a935e1b1da8edc19"
        "bf67096c644f4d20821540de2ad344dc"
        "a752322b9f3d44328a61cb53c24666b6"
        "69f2c0d75209451981be8d208204ea13"
        "b3faeff15a634855aab906858d2c8486"
        "e56b2aec322e4fe1b2b43bb419d5fde8"
        "fabe96c251694ab38a0c4a794244ae58"
        "c6d60eefb2b347f8a94f3cbc919fc33a"
        "e0cd278e177a469ba91992f1487c9c0e"
    )

    EXPOSED_FOUND=0
    for KEY in "${EXPOSED_KEYS[@]}"; do
        if grep -q "$KEY" "$ENV_FILE"; then
            EXPOSED_FOUND=$((EXPOSED_FOUND + 1))
        fi
    done

    if [ $EXPOSED_FOUND -eq 0 ]; then
        echo -e "${GREEN}✓ No exposed API keys detected${NC}"
    else
        echo -e "${YELLOW}⚠ WARNING: $EXPOSED_FOUND previously exposed key(s) detected${NC}"
        echo -e "${YELLOW}  → These keys were exposed in git history${NC}"
        echo -e "${YELLOW}  → Recommendation: Revoke and replace when possible${NC}"
        echo -e "${YELLOW}  → System will function but keys are compromised${NC}"
        WARNINGS=$((WARNINGS + EXPOSED_FOUND))
    fi
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠ Skipping (no .env file)${NC}"
fi
echo ""

# ==============================================
# VALIDATION 7: Check .gitignore
# ==============================================
echo -e "${BLUE}[7/7] Checking .gitignore configuration...${NC}"
GITIGNORE="$PROJECT_ROOT/.gitignore"
if [ -f "$GITIGNORE" ]; then
    if grep -q "^\.env$" "$GITIGNORE" || grep -q "^\.env" "$GITIGNORE"; then
        echo -e "${GREEN}✓ .env is in .gitignore${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ CRITICAL: .env is NOT in .gitignore!${NC}"
        echo -e "${YELLOW}  → Add it to prevent committing secrets: echo '.env' >> .gitignore${NC}"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${RED}✗ .gitignore file not found${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ==============================================
# SUMMARY
# ==============================================
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Passed:   $PASSED${NC}"
echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
echo -e "${RED}Errors:   $ERRORS${NC}"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL VALIDATIONS PASSED!${NC}"
    echo -e "${GREEN}  Your .env file is properly configured and secure.${NC}"
    echo -e "${GREEN}  You can safely run: docker-compose up -d${NC}"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ WARNINGS DETECTED${NC}"
    echo -e "${YELLOW}  Your .env file is functional but has minor issues.${NC}"
    echo -e "${YELLOW}  Please review the warnings above.${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}✗ CRITICAL ERRORS DETECTED!${NC}"
    echo -e "${RED}  Your .env file has $ERRORS critical issue(s).${NC}"
    echo -e "${RED}  Fix these issues before running docker-compose.${NC}"
    echo ""
    echo -e "${YELLOW}Quick fixes:${NC}"
    echo -e "${YELLOW}  1. Generate passwords: openssl rand -base64 32${NC}"
    echo -e "${YELLOW}  2. Generate Fernet key: python3 -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"${NC}"
    echo -e "${YELLOW}  3. Get TwelveData keys: https://twelvedata.com/apikey${NC}"
    echo -e "${YELLOW}  4. Set permissions: chmod 600 .env${NC}"
    echo ""
    exit 1
fi
