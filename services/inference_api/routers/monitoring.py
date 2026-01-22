"""
Monitoring Router - Drift detection and monitoring endpoints

Sprint 3: COMP-88 - Dashboard improvements
Provides API endpoints for drift monitoring visualization.

DEPENDENCIES FOR REAL DATA (not demo/fallback):
==============================================

1. Reference Statistics File:
   - Path: config/norm_stats.json
   - Format: { "feature_name": { "mean": float, "std": float, "min": float, "max": float, "count": int } }
   - Generated from training data using scripts/compute_reference_stats.py

2. App State Initialization (in main.py lifespan):
   - app.state.drift_detector = FeatureDriftDetector(reference_stats_path=...)
   - app.state.multivariate_drift_detector = MultivariateDriftDetector(n_features=OBSERVATION_DIM)

3. Observation Feeding:
   - Call POST /api/v1/monitoring/drift/observe with each inference observation
   - Or integrate into inference pipeline via middleware

4. Reference Data for Multivariate:
   - Load historical observations via POST /api/v1/monitoring/drift/reference
   - Typically 500-1000 samples from training data
"""

from fastapi import APIRouter, Request, Query, Body
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import numpy as np

router = APIRouter(tags=["monitoring"])
logger = logging.getLogger(__name__)


@router.get("/drift")
async def get_drift_report(
    req: Request,
    features: Optional[str] = Query(None, description="Comma-separated feature names to check"),
    include_multivariate: bool = Query(True, description="Include multivariate drift methods"),
):
    """
    Get current drift detection report.

    Returns univariate (KS test) and multivariate (MMD, Wasserstein, PCA) drift analysis.

    Args:
        features: Optional comma-separated list of specific features to check
        include_multivariate: Whether to include multivariate drift methods

    Returns:
        DriftReport with:
        - features_checked: Number of features analyzed
        - features_drifted: Number of features with detected drift
        - overall_drift_score: Aggregate drift score (0-1)
        - alert_active: Whether drift alert is triggered
        - univariate_results: Per-feature KS test results
        - multivariate_results: MMD, Wasserstein, PCA results
        - aggregate: Overall status summary
    """
    try:
        # Try to get drift detector from app state
        detector = getattr(req.app.state, 'drift_detector', None)
        multivariate_detector = getattr(req.app.state, 'multivariate_drift_detector', None)

        if detector is None:
            # Return mock/demo data if detector not initialized
            return _get_demo_drift_report()

        # Get univariate results
        feature_list = features.split(',') if features else None
        univariate_results = detector.check_drift(features=feature_list)

        # Get multivariate results
        multivariate_results = {}
        if include_multivariate and multivariate_detector is not None:
            try:
                mv_results = multivariate_detector.check_multivariate_drift()
                multivariate_results = {
                    method: {
                        "method": result.method,
                        "score": result.score,
                        "threshold": result.threshold,
                        "is_drifted": result.is_drifted,
                        "details": result.details,
                    }
                    for method, result in mv_results.items()
                }
            except Exception as e:
                logger.warning(f"Multivariate drift check failed: {e}")

        # Calculate aggregate
        any_univariate_drift = any(r.is_drifted for r in univariate_results)
        any_multivariate_drift = any(
            r.get("is_drifted", False)
            for r in multivariate_results.values()
        )

        # Determine max severity
        severities = [r.drift_severity for r in univariate_results]
        severity_order = ['none', 'low', 'medium', 'high']
        max_severity = 'none'
        for sev in reversed(severity_order):
            if sev in severities:
                max_severity = sev
                break

        # Build response
        drifted_count = sum(1 for r in univariate_results if r.is_drifted)
        total_count = len(univariate_results)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "features_checked": total_count,
            "features_drifted": drifted_count,
            "overall_drift_score": drifted_count / total_count if total_count > 0 else 0,
            "alert_active": any_univariate_drift or any_multivariate_drift,
            "univariate_results": [
                {
                    "feature_name": r.feature_name,
                    "is_drifted": r.is_drifted,
                    "p_value": r.p_value,
                    "statistic": r.statistic,
                    "drift_severity": r.drift_severity,
                    "reference_mean": getattr(r, 'reference_mean', None),
                    "current_mean": getattr(r, 'current_mean', None),
                    "percent_change": getattr(r, 'percent_change', None),
                }
                for r in univariate_results
            ],
            "multivariate_results": multivariate_results,
            "aggregate": {
                "any_drifted": any_univariate_drift or any_multivariate_drift,
                "max_severity": max_severity,
                "methods_triggered": [
                    m for m, r in multivariate_results.items()
                    if r.get("is_drifted", False)
                ],
            },
        }

    except Exception as e:
        logger.error(f"Drift report failed: {e}")
        return _get_demo_drift_report()


@router.get("/drift/history")
async def get_drift_history(
    req: Request,
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve"),
    feature: Optional[str] = Query(None, description="Specific feature to get history for"),
):
    """
    Get historical drift detection results.

    Args:
        hours: Number of hours of history (1-168)
        feature: Optional specific feature name

    Returns:
        List of historical drift reports with timestamps.
    """
    try:
        detector = getattr(req.app.state, 'drift_detector', None)

        if detector is None or not hasattr(detector, 'get_history'):
            # Return mock history
            return {
                "history": [],
                "message": "Historical data not available",
                "hours_requested": hours,
            }

        history = detector.get_history(hours=hours, feature=feature)

        return {
            "history": history,
            "hours_requested": hours,
            "feature_filter": feature,
            "count": len(history),
        }

    except Exception as e:
        logger.error(f"Drift history failed: {e}")
        return {
            "history": [],
            "error": str(e),
            "hours_requested": hours,
        }


@router.get("/drift/reference")
async def get_reference_stats(req: Request):
    """
    Get reference statistics used for drift comparison.

    Returns the reference distribution statistics (mean, std, percentiles)
    for each feature.
    """
    try:
        detector = getattr(req.app.state, 'drift_detector', None)

        if detector is None:
            return {
                "reference_stats": {},
                "message": "Drift detector not initialized",
            }

        # Get reference stats if available
        if hasattr(detector, '_reference_stats'):
            return {
                "reference_stats": detector._reference_stats,
                "features_count": len(detector._reference_stats),
                "timestamp": datetime.utcnow().isoformat(),
            }

        return {
            "reference_stats": {},
            "message": "No reference stats loaded",
        }

    except Exception as e:
        logger.error(f"Reference stats failed: {e}")
        return {
            "reference_stats": {},
            "error": str(e),
        }


@router.post("/drift/reset")
async def reset_drift_windows(req: Request):
    """
    Reset drift detection windows.

    Clears current observation windows and resets drift detection state.
    Use after model updates or data pipeline changes.
    """
    try:
        detector = getattr(req.app.state, 'drift_detector', None)
        multivariate_detector = getattr(req.app.state, 'multivariate_drift_detector', None)

        reset_results = {"univariate": False, "multivariate": False}

        if detector is not None and hasattr(detector, 'reset'):
            detector.reset()
            reset_results["univariate"] = True

        if multivariate_detector is not None and hasattr(multivariate_detector, 'reset'):
            multivariate_detector.reset()
            reset_results["multivariate"] = True

        return {
            "success": any(reset_results.values()),
            "reset_results": reset_results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Drift reset failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@router.post("/drift/observe")
async def add_drift_observation(
    req: Request,
    observation: Dict[str, float] = Body(..., description="Feature observation as dict"),
):
    """
    Add an observation to drift detection windows.

    Call this endpoint after each inference to track feature distributions.
    Both univariate and multivariate detectors are updated.

    Args:
        observation: Dictionary mapping feature names to values
            Example: {"rsi_9": 45.2, "atr_pct": 0.023, ...}

    Returns:
        Confirmation with current window sizes.
    """
    try:
        detector = getattr(req.app.state, 'drift_detector', None)
        multivariate_detector = getattr(req.app.state, 'multivariate_drift_detector', None)

        results = {"univariate_added": False, "multivariate_added": False}

        # Add to univariate detector
        if detector is not None:
            detector.add_observation(observation)
            results["univariate_added"] = True
            results["univariate_window_size"] = len(next(iter(detector._windows.values()))) if detector._windows else 0

        # Add to multivariate detector (convert to array)
        if multivariate_detector is not None:
            try:
                from src.core.contracts import FEATURE_ORDER
                obs_array = np.array([observation.get(f, 0.0) for f in FEATURE_ORDER])
                multivariate_detector.add_observation(obs_array)
                results["multivariate_added"] = True
                results["multivariate_window_size"] = len(multivariate_detector._current_window)
            except Exception as e:
                logger.warning(f"Could not add to multivariate detector: {e}")

        return {
            "success": results["univariate_added"] or results["multivariate_added"],
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Add observation failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@router.post("/drift/observe/batch")
async def add_drift_observations_batch(
    req: Request,
    observations: List[Dict[str, float]] = Body(..., description="List of feature observations"),
):
    """
    Add multiple observations to drift detection windows in batch.

    More efficient than calling /observe multiple times.

    Args:
        observations: List of observation dictionaries

    Returns:
        Count of observations added.
    """
    try:
        detector = getattr(req.app.state, 'drift_detector', None)
        multivariate_detector = getattr(req.app.state, 'multivariate_drift_detector', None)

        added_count = 0

        for obs in observations:
            if detector is not None:
                detector.add_observation(obs)

            if multivariate_detector is not None:
                try:
                    from src.core.contracts import FEATURE_ORDER
                    obs_array = np.array([obs.get(f, 0.0) for f in FEATURE_ORDER])
                    multivariate_detector.add_observation(obs_array)
                except:
                    pass

            added_count += 1

        return {
            "success": added_count > 0,
            "observations_added": added_count,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Batch add failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@router.post("/drift/reference/multivariate")
async def set_multivariate_reference(
    req: Request,
    observations: List[List[float]] = Body(..., description="2D array of reference observations"),
):
    """
    Set reference data for multivariate drift detection.

    This initializes the multivariate detector with historical observations.
    Required for MMD, Wasserstein, and PCA methods to work.

    Args:
        observations: 2D array of shape (n_samples, n_features)
            Typically 500-1000 samples from training data.

    Returns:
        Confirmation with reference data stats.
    """
    try:
        multivariate_detector = getattr(req.app.state, 'multivariate_drift_detector', None)

        if multivariate_detector is None:
            return {
                "success": False,
                "error": "Multivariate drift detector not initialized",
            }

        # Convert to numpy array
        ref_data = np.array(observations)

        if len(ref_data.shape) != 2:
            return {
                "success": False,
                "error": f"Expected 2D array, got shape {ref_data.shape}",
            }

        if ref_data.shape[1] != multivariate_detector.n_features:
            return {
                "success": False,
                "error": f"Expected {multivariate_detector.n_features} features, got {ref_data.shape[1]}",
            }

        # Set reference data
        multivariate_detector.set_reference_data(ref_data)

        return {
            "success": True,
            "reference_samples": ref_data.shape[0],
            "reference_features": ref_data.shape[1],
            "pca_fitted": multivariate_detector._pca_components is not None,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Set multivariate reference failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@router.get("/drift/status")
async def get_drift_status(req: Request):
    """
    Get current drift detection system status.

    Returns information about detector initialization and readiness.
    """
    detector = getattr(req.app.state, 'drift_detector', None)
    multivariate_detector = getattr(req.app.state, 'multivariate_drift_detector', None)
    drift_service = getattr(req.app.state, 'drift_observation_service', None)
    persistence = getattr(req.app.state, 'drift_persistence_service', None)

    return {
        "univariate": {
            "initialized": detector is not None,
            "has_reference": detector is not None and hasattr(detector, 'reference_manager') and len(detector.reference_manager.feature_names) > 0,
            "features_tracked": len(detector._windows) if detector else 0,
            "min_samples": detector.min_samples if detector else None,
        } if detector else {"initialized": False},
        "multivariate": {
            "initialized": multivariate_detector is not None,
            "has_reference": multivariate_detector is not None and multivariate_detector._reference_data is not None,
            "n_features": multivariate_detector.n_features if multivariate_detector else None,
            "current_samples": len(multivariate_detector._current_window) if multivariate_detector else 0,
            "pca_fitted": multivariate_detector._pca_components is not None if multivariate_detector else False,
        } if multivariate_detector else {"initialized": False},
        "observation_service": drift_service.get_status() if drift_service else {"active": False},
        "persistence": {
            "enabled": persistence is not None and persistence._enabled,
        },
        "ready_for_detection": (
            detector is not None and
            multivariate_detector is not None and
            multivariate_detector._reference_data is not None
        ),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/drift/persist")
async def persist_drift_check(
    req: Request,
    model_id: Optional[str] = Query(None, description="Model ID for the check"),
    triggered_by: str = Query("manual", description="What triggered the check"),
):
    """
    Run drift check and persist results to database.

    This endpoint:
    1. Runs the drift detection
    2. Saves results to drift_checks table
    3. Creates alert if severity is medium/high

    Returns:
        Drift report with persistence confirmation.
    """
    try:
        # Get drift report
        detector = getattr(req.app.state, 'drift_detector', None)
        multivariate_detector = getattr(req.app.state, 'multivariate_drift_detector', None)
        persistence = getattr(req.app.state, 'drift_persistence_service', None)

        if detector is None:
            return {"error": "Drift detector not initialized", "persisted": False}

        # Run univariate check
        univariate_results = detector.check_drift()

        # Run multivariate check
        multivariate_results = {}
        if multivariate_detector is not None:
            try:
                mv_results = multivariate_detector.check_multivariate_drift()
                multivariate_results = {
                    method: {
                        "method": result.method,
                        "score": result.score,
                        "threshold": result.threshold,
                        "is_drifted": result.is_drifted,
                    }
                    for method, result in mv_results.items()
                }
            except Exception as e:
                logger.warning(f"Multivariate check failed: {e}")

        # Calculate summary
        drifted_count = sum(1 for r in univariate_results if r.is_drifted)
        total_count = len(univariate_results)
        drift_score = drifted_count / total_count if total_count > 0 else 0

        severities = [r.drift_severity for r in univariate_results]
        max_severity = 'high' if 'high' in severities else \
                       'medium' if 'medium' in severities else \
                       'low' if 'low' in severities else 'none'

        # Persist if service available
        persist_result = None
        if persistence is not None:
            persist_result = await persistence.save_drift_check(
                check_type="univariate",
                features_checked=total_count,
                features_drifted=drifted_count,
                drift_score=drift_score,
                max_severity=max_severity,
                univariate_results=[
                    {
                        "feature_name": r.feature_name,
                        "is_drifted": r.is_drifted,
                        "p_value": r.p_value,
                        "statistic": getattr(r, 'ks_statistic', 0),
                        "drift_severity": r.drift_severity,
                    }
                    for r in univariate_results
                ],
                multivariate_results=multivariate_results,
                model_id=model_id,
                triggered_by=triggered_by,
            )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "features_checked": total_count,
            "features_drifted": drifted_count,
            "overall_drift_score": drift_score,
            "max_severity": max_severity,
            "persisted": persist_result is not None,
            "check_id": persist_result.get("check_id") if persist_result else None,
            "alert_created": persist_result.get("alert_created", False) if persist_result else False,
        }

    except Exception as e:
        logger.error(f"Persist drift check failed: {e}")
        return {"error": str(e), "persisted": False}


@router.get("/drift/alerts")
async def get_active_alerts(req: Request):
    """Get all active drift alerts."""
    persistence = getattr(req.app.state, 'drift_persistence_service', None)

    if persistence is None:
        return {"alerts": [], "message": "Persistence service not available"}

    alerts = await persistence.get_active_alerts()
    return {
        "alerts": alerts,
        "count": len(alerts),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/drift/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    req: Request,
    alert_id: int,
    acknowledged_by: str = Query(..., description="User acknowledging the alert"),
):
    """Acknowledge a drift alert."""
    persistence = getattr(req.app.state, 'drift_persistence_service', None)

    if persistence is None:
        return {"success": False, "message": "Persistence service not available"}

    success = await persistence.acknowledge_alert(alert_id, acknowledged_by)
    return {
        "success": success,
        "alert_id": alert_id,
        "acknowledged_by": acknowledged_by,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/drift/alerts/{alert_id}/resolve")
async def resolve_alert(
    req: Request,
    alert_id: int,
    resolved_by: str = Query(..., description="User resolving the alert"),
    notes: Optional[str] = Query(None, description="Resolution notes"),
):
    """Resolve a drift alert."""
    persistence = getattr(req.app.state, 'drift_persistence_service', None)

    if persistence is None:
        return {"success": False, "message": "Persistence service not available"}

    success = await persistence.resolve_alert(alert_id, resolved_by, notes)
    return {
        "success": success,
        "alert_id": alert_id,
        "resolved_by": resolved_by,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _get_demo_drift_report() -> dict:
    """
    Generate demo/mock drift report for when detector is not available.
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "features_checked": 15,
        "features_drifted": 2,
        "overall_drift_score": 0.133,
        "alert_active": False,
        "univariate_results": [
            {
                "feature_name": "rsi_9",
                "is_drifted": False,
                "p_value": 0.45,
                "statistic": 0.12,
                "drift_severity": "none",
            },
            {
                "feature_name": "atr_pct",
                "is_drifted": True,
                "p_value": 0.008,
                "statistic": 0.34,
                "drift_severity": "medium",
                "percent_change": 15.2,
            },
            {
                "feature_name": "macd_hist",
                "is_drifted": False,
                "p_value": 0.23,
                "statistic": 0.18,
                "drift_severity": "none",
            },
            {
                "feature_name": "log_ret_5m",
                "is_drifted": False,
                "p_value": 0.67,
                "statistic": 0.08,
                "drift_severity": "none",
            },
            {
                "feature_name": "volatility_20",
                "is_drifted": True,
                "p_value": 0.003,
                "statistic": 0.42,
                "drift_severity": "high",
                "percent_change": 28.5,
            },
            {
                "feature_name": "adx_14",
                "is_drifted": False,
                "p_value": 0.31,
                "statistic": 0.15,
                "drift_severity": "none",
            },
            {
                "feature_name": "bb_width",
                "is_drifted": False,
                "p_value": 0.52,
                "statistic": 0.11,
                "drift_severity": "low",
            },
            {
                "feature_name": "obv_norm",
                "is_drifted": False,
                "p_value": 0.78,
                "statistic": 0.06,
                "drift_severity": "none",
            },
        ],
        "multivariate_results": {
            "mmd": {
                "method": "MMD",
                "score": 0.15,
                "threshold": 0.3,
                "is_drifted": False,
            },
            "wasserstein": {
                "method": "Wasserstein",
                "score": 0.22,
                "threshold": 0.4,
                "is_drifted": False,
            },
            "pca_reconstruction": {
                "method": "PCA Recon",
                "score": 0.08,
                "threshold": 0.15,
                "is_drifted": False,
            },
        },
        "aggregate": {
            "any_drifted": True,
            "max_severity": "high",
            "methods_triggered": [],
        },
        "_demo": True,
    }
