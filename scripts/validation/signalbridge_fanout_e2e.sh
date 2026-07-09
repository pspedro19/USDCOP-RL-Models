#!/usr/bin/env bash
# SignalBridge multi-tenant S4 fan-out E2E (R5). Seeds two PAPER tenants with verified keys +
# different notional caps, has the admin publish one signal, and asserts the fan-out honours
# eligibility (verified key + limits + kill off), per-user caps, cascade kills, and admin-only
# gating (subscriber → 403 + audit). Paper-only: PENDING rows never reach an exchange (PreTradeGate).
# Self-cleaning. Usage: bash scripts/validation/signalbridge_fanout_e2e.sh
set -uo pipefail

SB="${QA_SB:-http://localhost:8085}"
PG="docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading -t -A"
PRO="f638c28b-856a-41a9-9457-63ae1cb8f636"   # pro@test.com  (subscriber) cap 5000
FREE="91e16b77-6b0d-49e1-9ecd-43266bbb5d1e"  # free@test.com (free)       cap 1000
SIG="fanout_e2e_$(date +%s)"
pass=0; fail=0
chk(){ if [ "$2" = "$3" ]; then echo "PASS: $1 ($2)"; pass=$((pass+1)); else echo "FAIL: $1 (got '$2' want '$3')"; fail=$((fail+1)); fi; }

tok(){ curl -s -X POST "$SB/api/auth/login" -H "Content-Type: application/json" \
  -d "{\"email\":\"$1\",\"password\":\"$2\"}" | python -c "import json,sys;print(json.load(sys.stdin).get('access_token','') or '')" 2>/dev/null; }

echo "=== seed two paper tenants (verified key + limits, kill off) ==="
$PG -c "INSERT INTO user_exchange_keys (user_id, exchange, api_key_enc, api_secret_enc, status)
        VALUES ('$PRO','mexc','enc','enc','verified'),('$FREE','mexc','enc','enc','verified')
        ON CONFLICT (user_id, exchange) DO UPDATE SET status='verified';" >/dev/null
$PG -c "INSERT INTO user_risk_limits_v2 (user_id, max_notional_usd, mode, kill_switch)
        VALUES ('$PRO',5000,'paper',false),('$FREE',1000,'paper',false)
        ON CONFLICT (user_id) DO UPDATE SET max_notional_usd=EXCLUDED.max_notional_usd, kill_switch=false, mode='paper';" >/dev/null
$PG -c "DELETE FROM user_executions WHERE signal_id LIKE 'fanout_e2e_%';" >/dev/null

ADMIN=$(tok "admin@trading.usdcop.com" "Admin2026!")
SUB=$(tok "pro@test.com" "Test2026!")
chk "admin token present" "$([ -n "$ADMIN" ] && echo yes || echo no)" "yes"
chk "subscriber token present" "$([ -n "$SUB" ] && echo yes || echo no)" "yes"

echo "=== 1. admin fan-out (notional 5000 @ px 100) → both eligible, capped ==="
R=$(curl -s -X POST "$SB/api/tenant/system/fan-out" -H "Authorization: Bearer $ADMIN" \
  -H "Content-Type: application/json" \
  -d "{\"signal_id\":\"$SIG\",\"symbol\":\"BTC/USDT\",\"side\":\"BUY\",\"notional_usd\":5000,\"price\":100}")
FANNED=$(echo "$R" | python -c "import json,sys;print(json.load(sys.stdin).get('fanned_out_to',''))" 2>/dev/null)
chk "fanned_out_to == 2" "$FANNED" "2"
PRO_QTY=$($PG -c "SELECT ROUND(qty::numeric,2) FROM user_executions WHERE signal_id='$SIG' AND user_id='$PRO';" | tr -d ' ')
FREE_QTY=$($PG -c "SELECT ROUND(qty::numeric,2) FROM user_executions WHERE signal_id='$SIG' AND user_id='$FREE';" | tr -d ' ')
chk "pro qty = 50.00 (5000/100, within cap)" "$PRO_QTY" "50.00"
chk "free qty = 10.00 (1000/100, capped at free's limit)" "$FREE_QTY" "10.00"
PEND=$($PG -c "SELECT count(*) FROM user_executions WHERE signal_id='$SIG' AND status='pending';" | tr -d ' ')
chk "both rows PENDING (never sent to exchange)" "$PEND" "2"

echo "=== 2. free kills own switch → re-fan → only pro eligible ==="
SIG2="${SIG}_b"
$PG -c "UPDATE user_risk_limits_v2 SET kill_switch=true WHERE user_id='$FREE';" >/dev/null
R2=$(curl -s -X POST "$SB/api/tenant/system/fan-out" -H "Authorization: Bearer $ADMIN" \
  -H "Content-Type: application/json" \
  -d "{\"signal_id\":\"$SIG2\",\"symbol\":\"BTC/USDT\",\"side\":\"BUY\",\"notional_usd\":5000,\"price\":100}")
FANNED2=$(echo "$R2" | python -c "import json,sys;print(json.load(sys.stdin).get('fanned_out_to',''))" 2>/dev/null)
chk "fanned_out_to == 1 (free killed, excluded)" "$FANNED2" "1"
FREE_ROWS=$($PG -c "SELECT count(*) FROM user_executions WHERE signal_id='$SIG2' AND user_id='$FREE';" | tr -d ' ')
chk "free got 0 rows on re-fan" "$FREE_ROWS" "0"

echo "=== 3. subscriber (non-admin) calls /system/fan-out → 403 + audit ==="
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$SB/api/tenant/system/fan-out" \
  -H "Authorization: Bearer $SUB" -H "Content-Type: application/json" \
  -d "{\"signal_id\":\"deny\",\"symbol\":\"BTC/USDT\",\"side\":\"BUY\",\"notional_usd\":100,\"price\":100}")
chk "subscriber fan-out DENIED 403" "$CODE" "403"
DENY_AUDIT=$($PG -c "SELECT count(*) FROM audit_log WHERE action='fanout_denied';" | tr -d ' ')
chk "fanout_denied audit row present (>=1)" "$([ "${DENY_AUDIT:-0}" -ge 1 ] && echo yes || echo no)" "yes"
FAN_AUDIT=$($PG -c "SELECT count(*) FROM audit_log WHERE action='fanout';" | tr -d ' ')
chk "fanout audit row present (>=2)" "$([ "${FAN_AUDIT:-0}" -ge 2 ] && echo yes || echo no)" "yes"

echo "=== cleanup ==="
$PG -c "DELETE FROM user_executions WHERE signal_id LIKE 'fanout_e2e_%';
        DELETE FROM user_exchange_keys WHERE user_id IN ('$PRO','$FREE') AND api_key_enc='enc';
        UPDATE user_risk_limits_v2 SET kill_switch=false WHERE user_id='$FREE';" >/dev/null
echo ""
echo "FANOUT E2E: $pass PASS, $fail FAIL"
exit 0
