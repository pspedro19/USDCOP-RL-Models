"""Static assurance contracts for critical cross-cutting controls.

These tests intentionally fail when a production-safety control is absent. They do not
claim runtime security; they prevent known architectural regressions and make gaps visible.
"""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]
DASH = ROOT / "usdcop-trading-dashboard"


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8", errors="replace")


def test_playwright_retains_failure_evidence() -> None:
    config = read("usdcop-trading-dashboard/playwright.config.ts")
    assert "screenshot: 'only-on-failure'" in config
    assert "video: 'retain-on-failure'" in config
    assert "trace: 'on-first-retry'" in config


def test_payment_webhook_verifies_signature_before_grant() -> None:
    route = read("usdcop-trading-dashboard/app/api/billing/webhook/route.ts")
    verify_at = route.index("verifyWebhook")
    update_at = route.index("UPDATE sb_users SET entitlements")
    assert verify_at < update_at
    assert "status: 401" in route


def test_cart_checkout_never_trusts_client_addons() -> None:
    route = read("usdcop-trading-dashboard/app/api/cart/checkout/route.ts")
    assert "SELECT asset_id FROM user_cart WHERE user_id = $1" in route
    assert "body.addOn" not in route


def test_audit_log_is_append_only() -> None:
    migration = read("database/migrations/055_rbac_monetization.sql")
    assert "BEFORE UPDATE OR DELETE ON audit_log" in migration
    assert "audit_log is append-only" in migration


def test_no_frontend_payment_test_gap_is_silently_accepted() -> None:
    tests = list((DASH / "tests").rglob("*"))
    payment_tests = [p for p in tests if p.is_file() and any(
        token in p.name.lower() for token in ("billing", "wompi", "checkout", "cart")
    )]
    assert payment_tests, (
        "No dedicated billing/Wompi/checkout/cart tests found; PAY-P0 controls need "
        "unit + integration + E2E coverage before production billing."
    )


def test_billing_uses_server_side_order_ledger() -> None:
    migrations = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in (ROOT / "database" / "migrations").glob("*.sql")
    ).lower()
    assert "checkout_orders" in migrations
    assert "billing_events" in migrations
    assert "provider_event_id" in migrations


def test_ai_context_has_untrusted_content_guardrail() -> None:
    prompt_files = [
        read("src/analysis/prompt_templates.py"),
        read("usdcop-trading-dashboard/lib/chat/context.ts"),
    ]
    combined = "\n".join(prompt_files).lower()
    markers = ("untrusted", "no confiable", "ignore instructions", "treat as data")
    assert any(marker in combined for marker in markers), (
        "LLM context must explicitly identify retrieved news/user content as untrusted data."
    )


def test_feature_registry_matches_active_experiment_dimension() -> None:
    """A legacy registry must never silently disagree with the active training SSOT."""
    registry = read("config/feature_registry.yaml")
    experiment = read("config/experiment_ssot.yaml")
    registry_dim = int(re.search(r"total_dimension:\s*(\d+)", registry).group(1))
    experiment_dim = int(re.search(r"observation_dim:\s*(\d+)", experiment).group(1))
    assert registry_dim == experiment_dim, (
        f"Feature-contract drift: feature_registry={registry_dim}, "
        f"experiment_ssot={experiment_dim}. Deprecate the registry or generate it from SSOT."
    )


def test_data_quality_gate_does_not_embed_legacy_feature_contract() -> None:
    gate = read("src/validation/data_quality_gate.py")
    assert '"expected_feature_count": 15' not in gate
    assert "EXPECTED_FEATURES = [" not in gate, (
        "DataQualityGate embeds the old feature list; it must load the active feature contract."
    )


def test_strategy_manifest_carries_decision_grade_evidence() -> None:
    manifest = read("src/contracts/strategy_manifest.py").lower()
    required = ("deflated_sharpe", "out_of_sample", "cost_sensitivity", "benchmark")
    missing = [field for field in required if field not in manifest]
    assert not missing, (
        "A production/promoted strategy manifest lacks decision-grade evidence: "
        + ", ".join(missing)
    )


def test_production_promotion_thresholds_are_economically_positive() -> None:
    config = read("config/experiment_ssot.yaml")
    gates = config.split("gates:", 1)[1].split("output:", 1)[0]
    min_return = float(re.search(r"min_return_pct:\s*(-?[\d.]+)", gates).group(1))
    min_sharpe = float(re.search(r"min_sharpe:\s*(-?[\d.]+)", gates).group(1))
    assert min_return > 0, "Promotion cannot accept a negative OOS return."
    assert min_sharpe > 0, "Promotion cannot accept a negative OOS Sharpe ratio."


def test_marketplace_does_not_claim_to_sell_models_without_model_skus() -> None:
    contract = read("usdcop-trading-dashboard/lib/contracts/catalog.contract.ts")
    required = ("model_version", "license", "risk_disclosure", "performance_snapshot")
    missing = [field for field in required if field not in contract]
    assert not missing, (
        "Current catalog sells asset access/add-ons, not auditable model products. "
        "Missing model-SKU fields: " + ", ".join(missing)
    )


def test_sp500_is_not_left_in_frontend_as_coming_soon_only() -> None:
    analysis = read("usdcop-trading-dashboard/lib/contracts/analysis-assets.ts")
    catalog = read("usdcop-trading-dashboard/lib/services/catalog-registry.ts")
    pricing = read("usdcop-trading-dashboard/lib/billing/prices.ts")
    assert "asset_id: 'spx500'" in analysis
    assert "spx500: 39_000" in pricing
    assert "asset_id: 'spx500'" in catalog  # registry will override teaser when manifest exists


def test_sp500_routes_and_internal_entitlements_are_wired() -> None:
    forecasting = read("usdcop-trading-dashboard/app/api/forecasting/[...path]/route.ts")
    data_route = read("usdcop-trading-dashboard/app/api/data/[...path]/route.ts")
    entitlements = read("usdcop-trading-dashboard/lib/auth/entitlements.ts")
    assert "btcusdt|spx500" in forecasting
    assert "btcusdt|spx500" in data_route
    assert "'spx500'" in entitlements
