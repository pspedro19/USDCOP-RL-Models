#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DSR family/cluster/global + gate de gobierno sobre registries/ (BL-09 re-entrega).

Remedia el rechazo CXD-009 ("09 sin DSR×3 ... ni gate"): el ledger ya divulgaba los tres N
(N_family/N_cluster/N_global) pero nadie los convertía en DSR ni gateaba con ellos.

Reglas (quant-constitution §2 + registries/README regla 2):

1. Para cada familia con un candidato con `dsr_inputs` COMPLETOS (sharpe_per_period, n_obs,
   skew, kurtosis, trials_sharpe_std, source) se recomputa el DSR TRES veces con
   `services.common.metrics.deflated_sharpe_ratio` (SSOT constitucional):
   n_trials = N_family, N_cluster, N_global del ledger — jamás un N hardcodeado ni stale.
2. Candidato solo con `published_dsr` (valor + n_trials_used + source): como el DSR es
   monótonamente NO-creciente en n_trials (sr0 crece con N), si n_trials_used <= N_family el
   valor publicado es COTA SUPERIOR de DSR_family/cluster/global. Una cota jamás habilita
   claim.
3. Sin candidato o sin insumos: DSR = null y claim_allowed = False. FAIL-CLOSED.
4. El gobierno gatea con DSR_family: `claims_edge: true` sin DSR_family COMPUTADO > 0.95
   es una violación (exit 1).
5. Política de cutoff (BL-09): toda fila nueva (env != legacy_backfill) exige cutoff
   no-nulo. Los 218 nulos del backfill son legacy declarados irrecuperables por escrito.
6. N_MAX=989 es cota de GASTO: NO aparece en ninguna fórmula de este script (§9.7).

Uso: python scripts/validation/report_ledger_dsr.py   (exit 0 = verde, 1 = violaciones)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # permite `python scripts/validation/report_ledger_dsr.py`
    sys.path.insert(0, str(ROOT))

from scripts.validation.check_trial_ledger import (  # noqa: E402
    FAMILIES_DIR,
    LEDGER_PATH,
    load_ledger,
)
from services.common.metrics import deflated_sharpe_ratio  # noqa: E402  (SSOT del DSR)

DSR_BAR = 0.95  # quant-constitution §2; cambiarlo requiere ADR
REQUIRED_DSR_INPUTS = ("sharpe_per_period", "n_obs", "skew", "kurtosis", "source")


def _sigma_grid(inputs: dict) -> list[float]:
    """Rejilla de sigma_trials declarada. Escalar o lista; el gate usa la MENOS favorable.

    `approval-gates.md` gate 6 lo exige literalmente ("sigma menos favorable"). Como el DSR
    es decreciente en sigma_trials (sr0 crece), la menos favorable es la sigma MAYOR — pero
    no se asume: se evalúan TODAS y se toma el DSR mínimo.
    """
    grid = inputs.get("trials_sharpe_std_grid")
    if isinstance(grid, list) and grid:
        return [float(value) for value in grid]
    single = inputs.get("trials_sharpe_std")
    return [float(single)] if single is not None else []


def load_families(families_dir: Path = FAMILIES_DIR) -> dict[str, dict]:
    families: dict[str, dict] = {}
    for yaml_path in sorted(families_dir.glob("*.yaml")):
        families[yaml_path.stem] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return families


def final_counts(records: list[dict]) -> tuple[Counter, Counter, int, dict[str, str]]:
    """(N_family, N_cluster, N_global, family->cluster) recomputados del ledger completo."""
    n_family: Counter = Counter()
    n_cluster: Counter = Counter()
    cluster_of: dict[str, str] = {}
    for record in records:
        n_family[record["family"]] += 1
        n_cluster[record["cluster"]] += 1
        cluster_of[record["family"]] = record["cluster"]
    return n_family, n_cluster, len(records), cluster_of


def compute_candidate_dsr(candidate: dict, n_family: int, n_cluster: int,
                          n_global: int, scope: str = "family") -> dict:
    """DSR×3 de un candidato. method: computed | bounded | none. Fail-closed siempre.

    `scope` (family|cluster) decide QUÉ DSR gatea: una familia que replica la misma mecánica
    en varios activos (sibling_families) deflacta con el cluster — más estricto, nunca menos.
    """
    result = {
        "candidate": candidate.get("id"),
        "n_x3": {"family": n_family, "cluster": n_cluster, "global": n_global},
        "method": "none",
        "scope": scope,
        "dsr_family": None, "dsr_cluster": None, "dsr_global": None,
        "dsr_grid": {},
        "dsr_family_upper_bound": None,
        "claim_allowed": False,
    }
    inputs = candidate.get("dsr_inputs")
    sigmas = _sigma_grid(inputs) if inputs else []
    if inputs and sigmas and all(inputs.get(key) is not None for key in REQUIRED_DSR_INPUTS):
        for level, n_trials in (("family", n_family), ("cluster", n_cluster),
                                ("global", n_global)):
            per_sigma = {}
            for sigma in sigmas:
                dsr = deflated_sharpe_ratio(
                    sharpe_per_period=float(inputs["sharpe_per_period"]),
                    n_obs=int(inputs["n_obs"]),
                    n_trials=int(n_trials),
                    trials_sharpe_std=sigma,
                    skew=float(inputs["skew"]),
                    kurtosis=float(inputs["kurtosis"]),
                )
                per_sigma[sigma] = dsr["dsr"]
            result["dsr_grid"][level] = per_sigma
            # sigma MENOS favorable = el DSR mínimo de la rejilla (approval-gates.md gate 6)
            result[f"dsr_{level}"] = min(per_sigma.values())
        result["method"] = "computed"
        gate_value = result["dsr_cluster"] if scope == "cluster" else result["dsr_family"]
        result["claim_allowed"] = bool(gate_value > DSR_BAR)
        return result
    published = candidate.get("published_dsr")
    if published and published.get("n_trials_used") and published.get("source"):
        if int(published["n_trials_used"]) <= n_family:
            # DSR no-creciente en n_trials => el publicado (con N menor) acota por arriba.
            result["method"] = "bounded"
            result["dsr_family_upper_bound"] = float(published["value"])
        else:
            # publicado con MÁS trials que N_family => solo acota por abajo: no habilita nada.
            result["method"] = "bounded_below"
            result["dsr_family_lower_bound"] = float(published["value"])
    return result


def check_cutoff_policy(records: list[dict]) -> list[str]:
    """BL-09 fail-closed: fila nueva (env != legacy_backfill) exige cutoff no-nulo."""
    errors = []
    for record in records:
        if record.get("env") != "legacy_backfill" and record.get("cutoff") in (None, "", "null"):
            errors.append(
                f"{record.get('trial_id')}: env='{record.get('env')}' exige cutoff no-nulo "
                "(los únicos cutoff nulos admitidos son el backfill legacy declarado)"
            )
    return errors


def check_governance(families: dict[str, dict], records: list[dict]) -> list[str]:
    """El gate DSR_family manda: claims_edge sin DSR_family computado > bar = violación."""
    errors = []
    n_family, n_cluster, n_global, cluster_of = final_counts(records)
    for family_id, family in families.items():
        governance = family.get("governance")
        if not governance:
            errors.append(f"{family_id}: sin bloque governance (gate DSR_family obligatorio)")
            continue
        scope_declared = family.get("deflation_scope", "family")
        expected_token = "DSR_cluster" if scope_declared == "cluster" else "DSR_family"
        if expected_token not in str(governance.get("gate", "")):
            errors.append(
                f"{family_id}: deflation_scope={scope_declared} pero el gate no referencia "
                f"{expected_token}"
            )
        if governance.get("claims_edge") is None:
            errors.append(f"{family_id}: claims_edge ausente (debe ser explícito)")
        candidates = governance.get("candidates", [])
        scope = family.get("deflation_scope", "family")
        passing = []
        for candidate in candidates:
            result = compute_candidate_dsr(
                candidate,
                n_family.get(family_id, 0),
                n_cluster.get(cluster_of.get(family_id, ""), n_global),
                n_global,
                scope,
            )
            if result["method"] == "computed" and result["claim_allowed"]:
                passing.append(candidate.get("id"))
        if governance.get("claims_edge") and not passing:
            errors.append(
                f"{family_id}: claims_edge=True pero ningún candidato tiene DSR_family "
                f"COMPUTADO > {DSR_BAR} (fail-closed: el claim se rechaza)"
            )
    return errors


def main() -> int:
    records = load_ledger(LEDGER_PATH)
    families = load_families()
    n_family, n_cluster, n_global, cluster_of = final_counts(records)
    print(f"DSR x3 por familia (N_global={n_global}; bar DSR_family > {DSR_BAR}):")
    for family_id in sorted(families):
        family = families[family_id]
        governance = family.get("governance") or {}
        candidates = governance.get("candidates", [])
        scope = family.get("deflation_scope", "family")
        header = (f"  {family_id} [N_family={n_family.get(family_id, 0)} "
                  f"N_cluster={n_cluster.get(cluster_of.get(family_id, ''), 0)} "
                  f"N_global={n_global}] scope={scope} "
                  f"claims_edge={governance.get('claims_edge')}")
        print(header)
        if not candidates:
            print("    (sin candidato => DSR nulo, claim_allowed=False - fail-closed)")
        for candidate in candidates:
            result = compute_candidate_dsr(
                candidate, n_family.get(family_id, 0),
                n_cluster.get(cluster_of.get(family_id, ""), n_global), n_global, scope,
            )
            if result["method"] == "computed":
                print(f"    {result['candidate']}: DSR family/cluster/global = "
                      f"{result['dsr_family']}/{result['dsr_cluster']}/{result['dsr_global']} "
                      f"(sigma menos favorable) -> claim_allowed={result['claim_allowed']}")
                for level, per_sigma in result["dsr_grid"].items():
                    grid = ", ".join(f"s={s:.6f}:{v}" for s, v in sorted(per_sigma.items()))
                    print(f"        {level}: {grid}")
            elif result["method"] == "bounded":
                print(f"    {result['candidate']}: DSR_family <= "
                      f"{result['dsr_family_upper_bound']} (cota publicada, "
                      f"n_used={candidate['published_dsr']['n_trials_used']} <= "
                      f"N_family) -> claim_allowed=False")
            elif result["method"] == "bounded_below":
                print(f"    {result['candidate']}: DSR_family >= "
                      f"{result['dsr_family_lower_bound']} (publicado con MAS trials que "
                      f"N_family: cota inferior, no habilita claim) -> claim_allowed=False")
            else:
                print(f"    {result['candidate']}: sin insumos -> DSR nulo, "
                      "claim_allowed=False (fail-closed)")
    errors = check_cutoff_policy(records) + check_governance(families, records)
    if errors:
        print(f"\n{len(errors)} VIOLACIONES:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nOK: politica de cutoff + gate DSR_family fail-closed sin violaciones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
