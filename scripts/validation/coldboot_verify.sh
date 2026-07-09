#!/usr/bin/env bash
# Cold-boot verification (CTR-DQ-OPS-001): after `docker compose down -v` → `up -d --build`,
# assert the platform reconstructed itself — schema (migrations 043-055), seeded data
# (OHLCV/macro from seeds+backups, news/analysis/H5/asset via feature restore), bootstrap
# users (admin + QA roles), and live services (dashboard/SB/airflow health + login).
# Usage: bash scripts/validation/coldboot_verify.sh [expected_counts_file]
set -uo pipefail
PG="docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading -t -A"
pass=0; fail=0
chk(){ if [ "$2" = "1" ]; then echo "PASS: $1"; pass=$((pass+1)); else echo "FAIL: $1 ($3)"; fail=$((fail+1)); fi; }

echo "── services healthy"
for c in usdcop-postgres-timescale usdcop-redis usdcop-signalbridge usdcop-dashboard usdcop-airflow-scheduler usdcop-airflow-webserver; do
  st=$(docker ps --filter "name=$c" --format "{{.Status}}" 2>/dev/null)
  case "$st" in *healthy*|*Up*) ok=1;; *) ok=0;; esac
  chk "container $c up" "$ok" "$st"
done

echo "── schema: migration-created tables exist (043-055)"
for t in forecast_h5_signals forecast_h5_subtrades news_articles weekly_analysis \
         asset_daily_ohlcv user_exchange_keys user_risk_limits_v2 user_executions audit_log sb_users; do
  n=$($PG -c "SELECT count(*) FROM information_schema.tables WHERE table_name='$t';" 2>/dev/null | tr -d ' ')
  chk "table $t exists" "$([ "${n:-0}" = "1" ] && echo 1 || echo 0)" "found=$n"
done
uq=$($PG -c "SELECT count(*) FROM pg_constraint WHERE conname='uq_h5_subtrade';" | tr -d ' ')
chk "migration 054 constraint uq_h5_subtrade" "$([ "$uq" = "1" ] && echo 1 || echo 0)" "$uq"
trg=$($PG -c "SELECT count(*) FROM pg_trigger WHERE tgname LIKE '%audit%block%' OR tgname LIKE '%no_update%';" | tr -d ' ')
ent=$($PG -c "SELECT count(*) FROM information_schema.columns WHERE table_name='sb_users' AND column_name='entitlements';" | tr -d ' ')
chk "migration 055 entitlements column" "$([ "$ent" = "1" ] && echo 1 || echo 0)" "$ent"

echo "── data restored (timeseries + derived/news)"
declare -A MIN=( [usdcop_m5_ohlcv]=100000 [macro_indicators_daily]=9000 [news_articles]=1 \
                 [forecast_h5_signals]=1 [forecast_h5_executions]=1 [asset_daily_ohlcv]=9000 )
for t in "${!MIN[@]}"; do
  n=$($PG -c "SELECT count(*) FROM $t;" 2>/dev/null | tr -d ' ')
  chk "$t rows >= ${MIN[$t]}" "$([ "${n:-0}" -ge "${MIN[$t]}" ] && echo 1 || echo 0)" "rows=$n"
done

echo "── bootstrap users (admin + QA roles)"
for pair in "admin@trading.usdcop.com:admin" "dev@test.com:developer" "pro@test.com:subscriber" "free@test.com:free"; do
  em="${pair%%:*}"; ro="${pair##*:}"
  n=$($PG -c "SELECT count(*) FROM sb_users WHERE email='$em' AND role='$ro' AND status='approved';" | tr -d ' ')
  chk "user $em role=$ro approved" "$([ "$n" = "1" ] && echo 1 || echo 0)" "$n"
done

echo "── live auth + app"
code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/login)
chk "dashboard /login 200" "$([ "$code" = "200" ] && echo 1 || echo 0)" "$code"
tok=$(curl -s -X POST http://localhost:8085/api/auth/login -H "Content-Type: application/json" \
  -d '{"email":"admin@trading.usdcop.com","password":"Admin2026!"}' | python -c "import json,sys;print(json.load(sys.stdin).get('access_token','') or '')" 2>/dev/null)
chk "SB admin login" "$([ -n "$tok" ] && echo 1 || echo 0)" "no token"
qa=$(curl -s -X POST http://localhost:8085/api/auth/login -H "Content-Type: application/json" \
  -d '{"email":"pro@test.com","password":"Test2026!"}' | python -c "import json,sys;print(json.load(sys.stdin).get('access_token','') or '')" 2>/dev/null)
chk "SB QA subscriber login" "$([ -n "$qa" ] && echo 1 || echo 0)" "no token"

echo ""
echo "COLDBOOT VERIFY: $pass PASS, $fail FAIL"
[ "$fail" = "0" ]
