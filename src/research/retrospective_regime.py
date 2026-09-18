"""Recover legacy regime categories for descriptive reporting, never for training.

The archived portable stores four coordinates of a five-state simplex. Recovering
the final coordinate is algebraic, not a refit or an independently recomputed HMM.
Current dataset guards deliberately continue to reject this undocumented contract.
"""

from __future__ import annotations

import hashlib

import numpy as np


def recover_regime_categories(stored, *, declared_k):
    """Validate four historical slots and return categories plus auditable evidence.

    Tolerance is for float32 storage roundoff only. Stored coordinates are never
    renormalized or mutated. A slightly negative residual is clipped to zero and its
    original excess is reported. Economic labels are deliberately not inferred.
    """
    if type(declared_k) is not int or not 2 <= declared_k <= 5:
        raise ValueError("declared_k must explicitly identify a model with 2..5 states")
    p = np.array(stored, dtype=np.float64, copy=True)
    if p.ndim != 2 or p.shape[1] != 4 or not len(p):
        raise ValueError("expected nonempty sessions x four stored coordinates")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("posterior coordinates must be finite and within [0, 1]")
    tolerance = 1e-6
    sums = p.sum(axis=1, dtype=np.float64)
    if (sums > 1 + tolerance).any():
        raise ValueError("posterior sum exceeds a probability simplex")
    if declared_k == 5:
        full = np.column_stack((p, np.maximum(0, 1 - sums)))
        method = "RESIDUAL_FIFTH_COORDINATE_DESCRIPTIVE_ONLY"
    else:
        if (p[:, declared_k:] != 0).any():
            raise ValueError("nonzero padding outside the declared model")
        if (np.abs(sums - 1) > tolerance).any():
            raise ValueError("full stored posterior sum must equal one within tolerance")
        full = p[:, :declared_k].copy()
        method = "FULL_STORED_POSTERIOR"
    old_ids = p.argmax(axis=1)
    ids = full.argmax(axis=1)
    ordered = np.sort(full, axis=1)
    return ids, {
        "scope": "retrospective_descriptive_reclassification",
        "method": method,
        "declared_k": declared_k,
        "stored_coordinates": 4,
        "n_sessions": len(p),
        "arithmetic": "float64 from archived coordinates; no renormalization",
        "roundoff_tolerance": tolerance,
        "stored_coordinates_float64_le_sha256": hashlib.sha256(
            p.astype("<f8").tobytes()
        ).hexdigest(),
        "max_sum_excess": float(max(0, sums.max() - 1)),
        "deficit_gt_tolerance_sessions": int(np.sum(1 - sums > tolerance)),
        "changed_argmax_sessions": int(np.sum(ids != old_ids)),
        "old_session_counts": np.bincount(old_ids, minlength=4).tolist(),
        "session_counts": np.bincount(ids, minlength=declared_k).tolist(),
        "near_tie_sessions": int(np.sum(ordered[:, -1] - ordered[:, -2] <= tolerance)),
        "tie_break": "lowest_state_id_numpy_argmax",
        "full_probabilities": full.tolist(),
        "state_ids": ids.tolist(),
        "training_inputs_modified": False,
        "independent_hmm_forward_recomputed": False,
        "new_policy_evaluated": False,
        "confirmatory": False,
    }
