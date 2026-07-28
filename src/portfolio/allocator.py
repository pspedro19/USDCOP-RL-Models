"""Pre-registered inverse-volatility and constrained allocator (BL-27)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol, Sequence

import numpy as np


class AllocationError(ValueError):
    pass


class InfeasibleAllocation(AllocationError):
    """The registered optimization problem has no acceptable solution."""


@dataclass(frozen=True, slots=True)
class AllocationIncident:
    code: str
    severity: str
    detail: str


@dataclass(frozen=True, slots=True)
class AllocationResult:
    risk_budgets: dict[str, float]
    signed_weights: dict[str, float]
    fallback_level: int
    incident: str | None = None
    incidents: tuple[AllocationIncident, ...] = ()


@dataclass(frozen=True, slots=True)
class OptimizationRequest:
    sleeves: tuple[str, ...]
    provisional: Mapping[str, float]
    previous: Mapping[str, float]
    covariance: tuple[tuple[float, ...], ...]
    sleeve_asset: Mapping[str, str]
    sleeve_caps: Mapping[str, float]
    asset_caps: Mapping[str, float]
    liquidity_caps: Mapping[str, float]
    factor_loadings: Mapping[str, Mapping[str, float]]
    factor_bounds: Mapping[str, tuple[float, float]]
    gross_cap: float
    target_vol: float
    turnover_budget: float
    lambda_turnover: float


class BudgetOptimizer(Protocol):
    def solve(self, request: OptimizationRequest) -> Mapping[str, float]:
        """Return one non-negative budget per sleeve or raise infeasibility."""
        ...


class CvxpyBudgetOptimizer:
    """CVXPY adapter kept behind a port so policy and solver stay separable."""

    def __init__(self, *, solver: str = "CLARABEL") -> None:
        if not isinstance(solver, str) or not solver.strip():
            raise AllocationError("solver must be a non-empty string")
        self.solver = solver

    def solve(self, request: OptimizationRequest) -> Mapping[str, float]:
        try:
            import cvxpy as cp
        except ImportError as exc:  # pragma: no cover - depends on deployment extra
            raise AllocationError(
                "constrained allocation requires the 'portfolio' dependency extra"
            ) from exc

        sleeves = request.sleeves
        n_sleeves = len(sleeves)
        index = {sleeve: position for position, sleeve in enumerate(sleeves)}
        provisional = np.asarray(
            [request.provisional[sleeve] for sleeve in sleeves], dtype=float
        )
        previous = np.asarray(
            [request.previous[sleeve] for sleeve in sleeves], dtype=float
        )
        covariance = np.asarray(request.covariance, dtype=float)
        budgets = cp.Variable(n_sleeves, nonneg=True)
        constraints = [
            budgets
            <= np.asarray(
                [request.sleeve_caps[sleeve] for sleeve in sleeves], dtype=float
            ),
            cp.sum(budgets) <= request.gross_cap,
            cp.quad_form(budgets, cp.psd_wrap(covariance))
            <= request.target_vol**2,
            cp.norm1(budgets - previous) <= request.turnover_budget,
        ]
        for asset, asset_cap in request.asset_caps.items():
            members = [
                index[sleeve]
                for sleeve in sleeves
                if request.sleeve_asset[sleeve] == asset
            ]
            if members:
                constraints.append(cp.sum(budgets[members]) <= asset_cap)
        for sleeve, liquidity_cap in request.liquidity_caps.items():
            constraints.append(budgets[index[sleeve]] <= liquidity_cap)
        for factor, loadings in request.factor_loadings.items():
            vector = np.asarray(
                [loadings[sleeve] for sleeve in sleeves], dtype=float
            )
            lower, upper = request.factor_bounds[factor]
            exposure = vector @ budgets
            constraints.extend((exposure >= lower, exposure <= upper))

        objective = cp.Minimize(
            cp.sum_squares(budgets - provisional)
            + request.lambda_turnover * cp.norm1(budgets - previous)
        )
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=self.solver, warm_start=False, verbose=False)
        except Exception as exc:
            raise InfeasibleAllocation(
                f"solver {self.solver} failed: {type(exc).__name__}"
            ) from exc
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise InfeasibleAllocation(f"optimizer status is {problem.status}")
        if budgets.value is None:
            raise InfeasibleAllocation("optimizer returned no budget vector")
        return {
            sleeve: float(budgets.value[index[sleeve]])
            for sleeve in sleeves
        }


class AllocatorV1:
    MULTIPLIER_NAMES = (
        "forward",
        "liquidity",
        "diversification",
        "operations",
        "drawdown",
    )

    def __init__(
        self,
        *,
        sleeve_caps: Mapping[str, float],
        gross_cap: float,
        asset_caps: Mapping[str, float],
        optimizer: BudgetOptimizer | None = None,
    ) -> None:
        if (
            isinstance(gross_cap, bool)
            or not isinstance(gross_cap, (int, float))
            or not math.isfinite(float(gross_cap))
            or float(gross_cap) <= 0
        ):
            raise AllocationError("gross_cap must be finite and positive")
        self.gross_cap = float(gross_cap)
        self.sleeve_caps = self._validated_caps("sleeve", sleeve_caps)
        self.asset_caps = self._validated_caps("asset", asset_caps)
        self.optimizer = optimizer

    @staticmethod
    def _validated_caps(kind: str, raw: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(raw, Mapping):
            raise AllocationError(f"{kind}_caps must be a mapping")
        result: dict[str, float] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise AllocationError(f"{kind} cap keys must be non-empty strings")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise AllocationError(
                    f"{kind}:{key} cap must be finite and non-negative"
                )
            result[key] = float(value)
        return result

    def allocate(
        self,
        *,
        volatility: Mapping[str, float],
        side: Mapping[str, int],
        sleeve_asset: Mapping[str, str],
        multipliers: Mapping[str, Mapping[str, float]],
    ) -> AllocationResult:
        for name, values in (
            ("volatility", volatility),
            ("side", side),
            ("sleeve_asset", sleeve_asset),
            ("multipliers", multipliers),
        ):
            if not isinstance(values, Mapping):
                raise AllocationError(f"{name} must be a mapping")
        if not volatility:
            return AllocationResult({}, {}, 4, "NO_SLEEVES_TARGET_ZERO")
        sleeves = set(volatility)
        for name, values in (
            ("side", side),
            ("sleeve_asset", sleeve_asset),
            ("multipliers", multipliers),
        ):
            if set(values) != sleeves:
                missing = sorted(sleeves - set(values), key=repr)
                extra = sorted(set(values) - sleeves, key=repr)
                raise AllocationError(
                    f"{name} keys must exactly match volatility; "
                    f"missing={missing}, extra={extra}"
                )
        for sleeve in sleeves:
            if (
                not isinstance(sleeve, str)
                or not sleeve.strip()
                or not isinstance(sleeve_asset[sleeve], str)
                or not sleeve_asset[sleeve].strip()
            ):
                raise AllocationError("sleeve and asset ids must be non-empty strings")
        missing_caps = sorted(sleeves - set(self.sleeve_caps), key=repr)
        if missing_caps:
            raise AllocationError(f"missing sleeve cap(s): {missing_caps}")
        missing_asset_caps = sorted(
            {sleeve_asset[sleeve] for sleeve in sleeves} - set(self.asset_caps),
            key=repr,
        )
        if missing_asset_caps:
            raise AllocationError(f"missing asset cap(s): {missing_asset_caps}")

        inverse_vol: dict[str, float] = {}
        combined_multiplier: dict[str, float] = {}
        for sleeve, vol in volatility.items():
            if (
                isinstance(side[sleeve], bool)
                or type(side[sleeve]) is not int
                or side[sleeve] not in {-1, 0, 1}
            ):
                raise AllocationError(f"{sleeve}: side must be -1, 0 or 1")
            if (
                isinstance(vol, bool)
                or not isinstance(vol, (int, float))
                or not math.isfinite(float(vol))
                or float(vol) <= 0
            ):
                raise AllocationError(
                    f"{sleeve}: volatility must be finite and positive"
                )
            values = multipliers[sleeve]
            if not isinstance(values, Mapping) or set(values) != set(
                self.MULTIPLIER_NAMES
            ):
                raise AllocationError(
                    f"{sleeve}: multipliers must explicitly contain "
                    f"{self.MULTIPLIER_NAMES}"
                )
            multiplier = 1.0
            for name in self.MULTIPLIER_NAMES:
                raw_value = values[name]
                if isinstance(raw_value, bool) or not isinstance(
                    raw_value, (int, float)
                ):
                    raise AllocationError(f"{sleeve}.{name} must be numeric")
                value = float(raw_value)
                lower, upper = (
                    (0.70, 1.10)
                    if name == "diversification"
                    else (0.0, 1.0)
                )
                if not math.isfinite(value) or value < lower or value > upper:
                    raise AllocationError(
                        f"{sleeve}.{name} must be in [{lower:g},{upper:g}]"
                    )
                if name == "operations" and value not in {0.0, 1.0}:
                    raise AllocationError(f"{sleeve}.operations must be 0 or 1")
                multiplier *= value
            combined_multiplier[sleeve] = multiplier
            if side[sleeve] == 0:
                continue
            inverse_vol[sleeve] = 1.0 / float(vol)
        if not inverse_vol:
            zeros = {sleeve: 0.0 for sleeve in volatility}
            return AllocationResult(zeros, dict(zeros), 0)
        inverse_total = sum(inverse_vol.values())

        # Normalize inverse-vol once, before caps.  Every later operation may
        # only reduce risk; there is deliberately no normalize-after-clip.
        baseline = {
            sleeve: min(
                self.gross_cap * inv / inverse_total,
                self.sleeve_caps[sleeve],
            )
            for sleeve, inv in inverse_vol.items()
        }
        by_asset_members: dict[str, list[str]] = {}
        for sleeve in baseline:
            by_asset_members.setdefault(sleeve_asset[sleeve], []).append(sleeve)
        for asset, members in by_asset_members.items():
            total = sum(baseline[sleeve] for sleeve in members)
            cap = self.asset_caps[asset]
            if total > cap and total > 0:
                scale = cap / total
                for sleeve in members:
                    baseline[sleeve] *= scale

        provisional: dict[str, float] = {}
        for sleeve, base_budget in baseline.items():
            provisional[sleeve] = min(
                base_budget * combined_multiplier[sleeve],
                self.sleeve_caps[sleeve],
            )

        # Diversification may increase one provisional budget up to 1.10.
        # Re-apply every hard cap using reduction-only transforms; this is not
        # a normalize-after-clip and can never inflate another sleeve.
        for asset, members in by_asset_members.items():
            total = math.fsum(provisional[sleeve] for sleeve in members)
            cap = self.asset_caps[asset]
            if total > cap and total > 0:
                scale = cap / total
                for sleeve in members:
                    provisional[sleeve] *= scale
        gross = math.fsum(provisional.values())
        if gross > self.gross_cap and gross > 0:
            scale = self.gross_cap / gross
            provisional = {
                sleeve: budget * scale
                for sleeve, budget in provisional.items()
            }
        total = sum(provisional.values())
        if total <= 0:
            zeros = {sleeve: 0.0 for sleeve in volatility}
            return AllocationResult(zeros, dict(zeros), 0, "RISK_GATES_TARGET_ZERO")
        budgets = {
            sleeve: provisional.get(sleeve, 0.0)
            for sleeve in volatility
        }
        signed = {
            sleeve: budgets[sleeve] * (1 if side.get(sleeve, 0) > 0 else -1 if side.get(sleeve, 0) < 0 else 0)
            for sleeve in budgets
        }
        return AllocationResult(budgets, signed, 0)

    def allocate_constrained(
        self,
        *,
        volatility: Mapping[str, float],
        side: Mapping[str, int],
        sleeve_asset: Mapping[str, str],
        multipliers: Mapping[str, Mapping[str, float]],
        previous_budgets: Mapping[str, float],
        covariance: Sequence[Sequence[float]],
        target_vol: float,
        turnover_budget: float,
        turnover_relaxation_limit: float,
        lambda_turnover: float = 0.01,
        liquidity_caps: Mapping[str, float] | None = None,
        factor_loadings: Mapping[str, Mapping[str, float]] | None = None,
        factor_bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> AllocationResult:
        """Solve the registered convex problem and apply its four fallbacks.

        The optimizer is treated as an untrusted numerical dependency: every
        returned vector is checked again against the economic constraints.
        """
        if self.optimizer is None:
            raise AllocationError("a BudgetOptimizer is required")
        provisional_result = self.allocate(
            volatility=volatility,
            side=side,
            sleeve_asset=sleeve_asset,
            multipliers=multipliers,
        )
        sleeves = tuple(sorted(volatility))
        if set(previous_budgets) != set(sleeves):
            raise AllocationError("previous_budgets keys must exactly match sleeves")
        previous = self._validated_budget_map(
            "previous_budgets", previous_budgets, sleeves
        )
        target = self._positive_number("target_vol", target_vol)
        turnover = self._non_negative_number(
            "turnover_budget", turnover_budget
        )
        relaxation_limit = self._non_negative_number(
            "turnover_relaxation_limit", turnover_relaxation_limit
        )
        if relaxation_limit < turnover:
            raise AllocationError(
                "turnover_relaxation_limit cannot be below turnover_budget"
            )
        penalty = self._non_negative_number(
            "lambda_turnover", lambda_turnover
        )
        covariance_matrix = self._validated_covariance(covariance, len(sleeves))
        liquidity = self._validated_optional_caps(
            "liquidity", liquidity_caps or {}, sleeves
        )
        factors, bounds = self._validated_factors(
            factor_loadings or {}, factor_bounds or {}, sleeves
        )
        incidents: list[AllocationIncident] = []

        def request_for(
            candidate: Mapping[str, float], allowed_turnover: float
        ) -> OptimizationRequest:
            return OptimizationRequest(
                sleeves=sleeves,
                provisional=dict(candidate),
                previous=previous,
                covariance=covariance_matrix,
                sleeve_asset={sleeve: sleeve_asset[sleeve] for sleeve in sleeves},
                sleeve_caps={
                    sleeve: self.sleeve_caps[sleeve] for sleeve in sleeves
                },
                asset_caps=dict(self.asset_caps),
                liquidity_caps=liquidity,
                factor_loadings=factors,
                factor_bounds=bounds,
                gross_cap=self.gross_cap,
                target_vol=target,
                turnover_budget=allowed_turnover,
                lambda_turnover=penalty,
            )

        def attempt(
            candidate: Mapping[str, float], allowed_turnover: float
        ) -> dict[str, float] | None:
            request = request_for(candidate, allowed_turnover)
            try:
                solved = self.optimizer.solve(request)
                return self._validated_solution(request, solved)
            except InfeasibleAllocation:
                return None

        solved = attempt(provisional_result.risk_budgets, turnover)
        if solved is not None:
            return self._result(solved, side, 0, incidents)
        incidents.append(
            AllocationIncident(
                "ALLOCATOR_PRIMARY_INFEASIBLE",
                "WARNING",
                "registered turnover budget produced no feasible target",
            )
        )

        solved = attempt(
            provisional_result.risk_budgets, relaxation_limit
        )
        if solved is not None:
            incidents.append(
                AllocationIncident(
                    "ALLOCATOR_FALLBACK_1_TURNOVER_RELAXED",
                    "WARNING",
                    "turnover relaxed only to its pre-declared ceiling",
                )
            )
            return self._result(solved, side, 1, incidents)
        incidents.append(
            AllocationIncident(
                "ALLOCATOR_FALLBACK_1_INFEASIBLE",
                "WARNING",
                "declared turnover relaxation remained infeasible",
            )
        )

        for shrinkage in (0.75, 0.50, 0.25, 0.0):
            shrunk = {
                sleeve: provisional_result.risk_budgets[sleeve] * shrinkage
                for sleeve in sleeves
            }
            solved = attempt(shrunk, relaxation_limit)
            if solved is not None:
                incidents.append(
                    AllocationIncident(
                        "ALLOCATOR_FALLBACK_2_SHRINK_TO_ZERO",
                        "WARNING",
                        f"provisional target shrinkage={shrinkage:g}",
                    )
                )
                return self._result(solved, side, 2, incidents)
        incidents.append(
            AllocationIncident(
                "ALLOCATOR_FALLBACK_2_INFEASIBLE",
                "WARNING",
                "shrink-to-zero search remained infeasible",
            )
        )

        neutral = {
            sleeve: {
                "forward": 1.0,
                "liquidity": 1.0,
                "diversification": 1.0,
                "operations": 1.0,
                "drawdown": 1.0,
            }
            for sleeve in sleeves
        }
        baseline = self.allocate(
            volatility=volatility,
            side=side,
            sleeve_asset=sleeve_asset,
            multipliers=neutral,
        )
        solved = attempt(baseline.risk_budgets, relaxation_limit)
        if solved is not None:
            incidents.append(
                AllocationIncident(
                    "ALLOCATOR_FALLBACK_3_INVERSE_VOL",
                    "WARNING",
                    "returned to the pre-registered inverse-volatility baseline",
                )
            )
            return self._result(solved, side, 3, incidents)

        incidents.append(
            AllocationIncident(
                "ALLOCATOR_FALLBACK_4_TARGET_ZERO",
                "CRITICAL",
                "all registered feasible-target attempts failed",
            )
        )
        zero = {sleeve: 0.0 for sleeve in sleeves}
        return self._result(zero, side, 4, incidents)

    @staticmethod
    def _positive_number(name: str, value: float) -> float:
        number = AllocatorV1._non_negative_number(name, value)
        if number == 0:
            raise AllocationError(f"{name} must be positive")
        return number

    @staticmethod
    def _non_negative_number(name: str, value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise AllocationError(f"{name} must be finite and non-negative")
        return float(value)

    @classmethod
    def _validated_budget_map(
        cls, name: str, raw: Mapping[str, float], sleeves: tuple[str, ...]
    ) -> dict[str, float]:
        if not isinstance(raw, Mapping) or set(raw) != set(sleeves):
            raise AllocationError(f"{name} keys must exactly match sleeves")
        return {
            sleeve: cls._non_negative_number(
                f"{name}.{sleeve}", raw[sleeve]
            )
            for sleeve in sleeves
        }

    @classmethod
    def _validated_optional_caps(
        cls, name: str, raw: Mapping[str, float], sleeves: tuple[str, ...]
    ) -> dict[str, float]:
        if not isinstance(raw, Mapping) or not set(raw).issubset(sleeves):
            raise AllocationError(f"{name}_caps contains an unknown sleeve")
        return {
            sleeve: cls._non_negative_number(
                f"{name}_caps.{sleeve}", value
            )
            for sleeve, value in raw.items()
        }

    @staticmethod
    def _validated_covariance(
        raw: Sequence[Sequence[float]], size: int
    ) -> tuple[tuple[float, ...], ...]:
        try:
            matrix = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise AllocationError("covariance must be a numeric matrix") from exc
        if (
            matrix.shape != (size, size)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)
        ):
            raise AllocationError(
                "covariance must be finite, symmetric and sleeve-aligned"
            )
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues.min(initial=0.0) < -1e-10:
            raise AllocationError("covariance must be positive semidefinite")
        return tuple(tuple(float(value) for value in row) for row in matrix)

    @classmethod
    def _validated_factors(
        cls,
        raw_loadings: Mapping[str, Mapping[str, float]],
        raw_bounds: Mapping[str, tuple[float, float]],
        sleeves: tuple[str, ...],
    ) -> tuple[dict[str, dict[str, float]], dict[str, tuple[float, float]]]:
        if set(raw_loadings) != set(raw_bounds):
            raise AllocationError("factor loadings and bounds must share exact keys")
        loadings: dict[str, dict[str, float]] = {}
        bounds: dict[str, tuple[float, float]] = {}
        for factor, values in raw_loadings.items():
            if not isinstance(factor, str) or not factor.strip():
                raise AllocationError("factor ids must be non-empty strings")
            if not isinstance(values, Mapping) or set(values) != set(sleeves):
                raise AllocationError(
                    f"factor {factor} loadings must cover every sleeve"
                )
            converted: dict[str, float] = {}
            for sleeve, value in values.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise AllocationError(
                        f"factor {factor}.{sleeve} must be finite"
                    )
                converted[sleeve] = float(value)
            bound = raw_bounds[factor]
            if not isinstance(bound, tuple) or len(bound) != 2:
                raise AllocationError(f"factor {factor} bound must be a pair")
            lower = cls._finite_signed(f"factor {factor} lower", bound[0])
            upper = cls._finite_signed(f"factor {factor} upper", bound[1])
            if lower > upper:
                raise AllocationError(f"factor {factor} lower exceeds upper")
            loadings[factor] = converted
            bounds[factor] = (lower, upper)
        return loadings, bounds

    @staticmethod
    def _finite_signed(name: str, value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise AllocationError(f"{name} must be finite numeric")
        return float(value)

    @staticmethod
    def _validated_solution(
        request: OptimizationRequest, raw: Mapping[str, float]
    ) -> dict[str, float]:
        sleeves = request.sleeves
        try:
            budgets = AllocatorV1._validated_budget_map(
                "optimizer_result", raw, sleeves
            )
        except AllocationError as exc:
            raise InfeasibleAllocation(str(exc)) from exc
        tolerance = 1e-7
        for sleeve, budget in budgets.items():
            hard_cap = min(
                request.sleeve_caps[sleeve],
                request.liquidity_caps.get(sleeve, math.inf),
            )
            if budget > hard_cap + tolerance:
                raise InfeasibleAllocation(f"{sleeve} exceeds its hard cap")
        for asset, cap in request.asset_caps.items():
            total = math.fsum(
                budgets[sleeve]
                for sleeve in sleeves
                if request.sleeve_asset[sleeve] == asset
            )
            if total > cap + tolerance:
                raise InfeasibleAllocation(f"asset {asset} exceeds its cap")
        if math.fsum(budgets.values()) > request.gross_cap + tolerance:
            raise InfeasibleAllocation("gross cap exceeded")
        turnover = math.fsum(
            abs(budgets[sleeve] - request.previous[sleeve])
            for sleeve in sleeves
        )
        if turnover > request.turnover_budget + tolerance:
            raise InfeasibleAllocation("turnover budget exceeded")
        vector = np.asarray([budgets[sleeve] for sleeve in sleeves], dtype=float)
        covariance = np.asarray(request.covariance, dtype=float)
        variance = float(vector @ covariance @ vector)
        if variance < -tolerance or math.sqrt(max(variance, 0.0)) > (
            request.target_vol + tolerance
        ):
            raise InfeasibleAllocation("target volatility exceeded")
        for factor, loadings in request.factor_loadings.items():
            exposure = math.fsum(
                budgets[sleeve] * loadings[sleeve] for sleeve in sleeves
            )
            lower, upper = request.factor_bounds[factor]
            if exposure < lower - tolerance or exposure > upper + tolerance:
                raise InfeasibleAllocation(f"factor {factor} bound exceeded")
        return budgets

    @staticmethod
    def _result(
        budgets: Mapping[str, float],
        side: Mapping[str, int],
        fallback_level: int,
        incidents: Sequence[AllocationIncident],
    ) -> AllocationResult:
        copied = dict(budgets)
        signed = {
            sleeve: budget * (1 if side[sleeve] > 0 else -1 if side[sleeve] < 0 else 0)
            for sleeve, budget in copied.items()
        }
        incident_tuple = tuple(incidents)
        return AllocationResult(
            copied,
            signed,
            fallback_level,
            incident_tuple[-1].code if incident_tuple else None,
            incident_tuple,
        )

    @staticmethod
    def novelty_gate(max_correlation: float, delta_information_ratio: float) -> bool:
        if not math.isfinite(max_correlation) or not math.isfinite(delta_information_ratio):
            raise AllocationError("novelty inputs must be finite")
        return max_correlation < 0.60 or delta_information_ratio > 0.15
