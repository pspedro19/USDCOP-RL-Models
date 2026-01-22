"""
NORM STATS CONTRACT
===================
Define el contrato para norm_stats.json.

Contract ID: CTR-NORM-STATS-001
Version: 2.0.0
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Final
from datetime import datetime, date
from pathlib import Path
import json
import hashlib
from pydantic import BaseModel, Field, field_validator


NORM_STATS_CONTRACT_VERSION: Final[str] = "2.0.0"


class FeatureStats(BaseModel):
    """Estadísticas de una feature individual."""
    mean: float = Field(..., description="Feature mean")
    std: float = Field(..., gt=0, description="Feature std (must be > 0)")
    min_val: Optional[float] = Field(None, alias="min", description="Observed minimum")
    max_val: Optional[float] = Field(None, alias="max", description="Observed maximum")

    model_config = {"populate_by_name": True}

    @field_validator("std")
    @classmethod
    def std_positive(cls, v):
        if v <= 0:
            raise ValueError("std must be positive")
        return v


class NormStatsMetadata(BaseModel):
    """Metadata obligatoria de norm_stats."""
    version: str = Field(NORM_STATS_CONTRACT_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    generated_at: datetime = Field(..., description="When stats were generated")
    sample_count: int = Field(..., gt=0, description="Number of samples used")
    data_start: date = Field(..., description="Start of data range")
    data_end: date = Field(..., description="End of data range")
    dataset_hash: Optional[str] = Field(None, pattern=r"^[a-f0-9]{64}$")


class NormStatsContract(BaseModel):
    """Contrato completo de norm_stats.json."""
    metadata: NormStatsMetadata = Field(..., alias="_metadata")
    features: Dict[str, FeatureStats]

    model_config = {"populate_by_name": True}

    @field_validator("features")
    @classmethod
    def validate_feature_count(cls, v):
        from src.core.contracts.feature_contract import FEATURE_ORDER
        normalized_features = [f for f in FEATURE_ORDER if f not in ("position", "time_normalized")]
        missing = [f for f in normalized_features if f not in v]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return v

    def compute_hash(self) -> str:
        """Computa hash determinístico."""
        canonical = {
            "features": {
                name: {"mean": stats.mean, "std": stats.std}
                for name, stats in sorted(self.features.items())
            }
        }
        content = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_metadata": self.metadata.model_dump(),
            "features": {
                name: stats.model_dump(exclude_none=True, by_alias=True)
                for name, stats in self.features.items()
            }
        }

    @classmethod
    def from_file(cls, path: str) -> "NormStatsContract":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class NormStatsContractError(Exception):
    pass


def load_norm_stats(path: str) -> Tuple["NormStatsContract", str]:
    """Carga norm_stats con validación y retorna hash."""
    try:
        contract = NormStatsContract.from_file(path)
        hash_value = contract.compute_hash()
        return contract, hash_value
    except Exception as e:
        raise NormStatsContractError(f"Failed to load norm_stats from {path}: {e}")


def save_norm_stats(
    features: Dict[str, Dict[str, float]],
    path: str,
    sample_count: int,
    data_start: date,
    data_end: date,
    dataset_hash: str = None,
) -> str:
    """Guarda norm_stats con validación."""
    metadata = NormStatsMetadata(
        version=NORM_STATS_CONTRACT_VERSION,
        generated_at=datetime.utcnow(),
        sample_count=sample_count,
        data_start=data_start,
        data_end=data_end,
        dataset_hash=dataset_hash,
    )

    feature_stats = {
        name: FeatureStats(**stats)
        for name, stats in features.items()
    }

    contract = NormStatsContract(
        _metadata=metadata,
        features=feature_stats,
    )

    output = contract.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    return contract.compute_hash()
